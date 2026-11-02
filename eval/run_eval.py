"""Evaluation (SPEC section 12). Prints the metrics table with explicit numerators and denominators.

    python eval/run_eval.py [--labels eval/labels] [--split heldout] [--json out.json]

Label file schema: eval/labels/SCHEMA.md. Only runs with runs.source = 'real' are evaluated;
fixture runs are refused unless --include-fixtures, and then every row is marked FIXTURE.
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db import connect, init_db, parse_iso  # noqa: E402

CORRECTION_FIELDS = ("brand", "name", "size", "price_pesewas")


@dataclass
class Ratio:
    num: int = 0
    den: int = 0

    def add(self, other: "Ratio") -> None:
        self.num += other.num
        self.den += other.den

    def fmt(self) -> str:
        return f"{self.num}/{self.den} ({100 * self.num / self.den:.0f}%)" if self.den else f"{self.num}/0 (n/a)"


@dataclass
class VideoResult:
    video: str
    run_id: str
    split: str
    source: str
    labelled: int = 0
    detection: Ratio = field(default_factory=Ratio)
    duplicates: Ratio = field(default_factory=Ratio)
    identification: Ratio = field(default_factory=Ratio)
    auto_precision: Ratio = field(default_factory=Ratio)
    auto_coverage: Ratio = field(default_factory=Ratio)
    spark_links: Ratio = field(default_factory=Ratio)
    corrections: int = 0
    published: bool = False
    time_to_shop_s: float | None = None
    cost_usd: float = 0.0
    calls: int = 0
    calls_without_cost: int = 0
    order_safety_failures: int = 0
    orders_confirmed: int = 0
    notes: list[str] = field(default_factory=list)


class LabelError(ValueError):
    pass


def load_label(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("video", "split", "run_id", "products", "track_to_label"):
        if key not in data:
            raise LabelError(f"{path.name}: missing '{key}'")
    if data["split"] not in ("tuning", "heldout"):
        raise LabelError(f"{path.name}: split must be tuning or heldout")
    ids = [p["label_id"] for p in data["products"]]
    if len(ids) != len(set(ids)):
        raise LabelError(f"{path.name}: duplicate label_id")
    for tid, lid in data["track_to_label"].items():
        if lid is not None and lid not in ids:
            raise LabelError(f"{path.name}: track {tid} maps to unknown label {lid}")
    for p in data["products"]:
        if p.get("price_pesewas") is not None and (not isinstance(p["price_pesewas"], int) or p["price_pesewas"] <= 0):
            raise LabelError(f"{path.name}: {p['label_id']} price_pesewas must be a positive integer")
    return data


def _tokens(text) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def identification_correct(product: dict, label: dict) -> bool:
    """Model's original guess (not the seller's edit): label name tokens all present, brand matches if labelled."""
    name_ok = bool(_tokens(label["name"])) and _tokens(label["name"]) <= _tokens(product["model_name"])
    brand_ok = not label.get("brand") or _tokens(label["brand"]) <= _tokens(product["model_brand"])
    return name_ok and brand_ok


def score_price_links(run_path: Path, label: dict) -> Ratio:
    """Prices Muse Spark linked to exactly the right tracks. 0/0 when the optional link stage did not run."""
    art = run_path / "artefacts"
    if not (art / "price_links.json").is_file():
        return Ratio()
    links = json.loads((art / "price_links.json").read_text(encoding="utf-8"))["links"]
    mentions = json.loads((art / "price_mentions.json").read_text(encoding="utf-8"))["mentions"]
    prices = {p["label_id"]: p.get("price_pesewas") for p in label["products"]}
    correct = 0
    for link in links:
        amount = mentions[link["mention_index"]]["amount_pesewas"]
        want = {t for t, lid in label["track_to_label"].items() if lid and prices[lid] == amount}
        correct += bool(want) and set(link["track_ids"]) == want
    return Ratio(correct, len(links))


def evaluate_run(conn, label: dict, runs_dir: Path | None = None) -> VideoResult:
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (label["run_id"],)).fetchone()
    if not run:
        raise LabelError(f"{label['video']}: run {label['run_id']} not in database")
    r = VideoResult(label["video"], run["id"], label["split"], run["source"], labelled=len(label["products"]))
    labels = {p["label_id"]: p for p in label["products"]}
    products = [dict(p) for p in conn.execute("SELECT * FROM products WHERE run_id = ? ORDER BY id", (run["id"],))]
    mapping = label["track_to_label"]
    unmapped = [p["track_id"] for p in products if p["track_id"] not in mapping]
    if unmapped:
        r.notes.append(f"{len(unmapped)} tracks not aligned in label file; treated as false detections")

    by_label: dict[str, list[dict]] = {}
    for p in products:
        lid = mapping.get(p["track_id"])
        if lid:
            by_label.setdefault(lid, []).append(p)

    r.detection = Ratio(len(by_label), len(labels))
    r.duplicates = Ratio(sum(1 for ps in by_label.values() if len(ps) > 1), len(by_label))
    r.identification = Ratio(sum(1 for lid, ps in by_label.items() if identification_correct(ps[0], labels[lid])),
                             len(by_label))
    auto = [p for p in products if p["match_status"] == "auto"]
    correct = 0
    for p in auto:
        lid = mapping.get(p["track_id"])
        if lid and labels[lid].get("price_pesewas") == p["model_price_pesewas"]:
            correct += 1
    r.auto_precision = Ratio(correct, len(auto))
    priced = [p for p in products if (lid := mapping.get(p["track_id"])) and labels[lid].get("price_pesewas")]
    r.auto_coverage = Ratio(sum(1 for p in priced if p["match_status"] == "auto"), len(priced))

    if runs_dir is not None:
        r.spark_links = score_price_links(Path(runs_dir) / run["id"], label)
    ids = [p["id"] for p in products]
    if ids:
        marks = ",".join("?" * len(ids))
        r.corrections = conn.execute(
            f"SELECT COUNT(*) FROM product_edits WHERE product_id IN ({marks}) AND field IN "
            f"({','.join('?' * len(CORRECTION_FIELDS))})", (*ids, *CORRECTION_FIELDS)).fetchone()[0]

    shop = conn.execute("SELECT * FROM shops WHERE run_id = ?", (run["id"],)).fetchone()
    if shop and shop["published_at"]:
        r.published = True
        r.time_to_shop_s = (parse_iso(shop["published_at"]) - parse_iso(run["created_at"])).total_seconds()
        for o in conn.execute("SELECT * FROM orders WHERE shop_id = ? AND status = 'confirmed'", (shop["id"],)):
            r.orders_confirmed += 1
            lines = json.loads(o["lines_json"])
            bad = (sum(l["unit_price_pesewas"] * l["qty"] for l in lines) != o["total_pesewas"]
                   or (o["confirmed_at"] and parse_iso(o["confirmed_at"]) > parse_iso(o["expires_at"]))
                   or any(l["product_id"] not in ids for l in lines))
            r.order_safety_failures += int(bool(bad))

    for c in conn.execute("SELECT cost_usd FROM model_calls WHERE run_id = ?", (run["id"],)):
        r.calls += 1
        if c["cost_usd"] is None:
            r.calls_without_cost += 1
        else:
            r.cost_usd += c["cost_usd"]
    return r


def format_table(results: list[VideoResult], title: str) -> str:
    agg = {k: Ratio() for k in ("detection", "duplicates", "identification", "auto_precision", "auto_coverage",
                                "spark_links")}
    for r in results:
        for k in agg:
            agg[k].add(getattr(r, k))
    n = len(results)
    published = sum(r.published for r in results)
    times = [r.time_to_shop_s for r in results if r.time_to_shop_s is not None]
    calls = sum(r.calls for r in results)
    no_cost = sum(r.calls_without_cost for r in results)
    cost = sum(r.cost_usd for r in results)
    cost_txt = (f"${cost / n:.4f} per video" if n else "n/a") + (f" (+{no_cost}/{calls} calls with unknown cost)" if no_cost else "")
    fixture = any(r.source != "real" for r in results)
    rows = [
        ("Detection recall", agg["detection"].fmt(), "labelled products found by >=1 track", ">=80%"),
        ("Duplicate rate", agg["duplicates"].fmt(), "found products with >1 track", "<=10%"),
        ("Identification accuracy", agg["identification"].fmt(), "found products, model brand+name correct", ">=80%"),
        ("Auto-match precision", agg["auto_precision"].fmt(), "auto-matched products with correct price", ">=90%"),
        ("Auto-match coverage", agg["auto_coverage"].fmt(), "priced products auto-matched", "report"),
        ("Muse Spark price links", agg["spark_links"].fmt(), "spoken prices linked to exactly the right tracks", "report"),
        ("Seller corrections", str(sum(r.corrections for r in results)), f"edits across {n} videos", "lower"),
        ("End-to-end completion", Ratio(published, n).fmt(), "videos with a published shop", "report"),
        ("Time to shop", f"median {sorted(times)[len(times) // 2]:.0f}s" if times else "n/a (none published)",
         f"{len(times)} published runs", "report"),
        ("Cost per video", cost_txt, "sum of model_calls.cost_usd", "report"),
        ("Order safety failures", str(sum(r.order_safety_failures for r in results)),
         f"of {sum(r.orders_confirmed for r in results)} confirmed orders", "0"),
    ]
    out = [f"## {title}" + ("  ** FIXTURE DATA - NOT A RESULT **" if fixture else ""),
           f"Videos: {n}; labelled products: {sum(r.labelled for r in results)}", "",
           "| Metric | Value | Denominator | Target |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    for r in results:
        for note in r.notes:
            out.append(f"- {r.video}: {note}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/labels")
    ap.add_argument("--split", choices=["tuning", "heldout", "all"], default="all")
    ap.add_argument("--include-fixtures", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    s = get_settings()
    conn = connect(s.db_path)
    init_db(conn)
    results = []
    for path in sorted(Path(args.labels).glob("*.json")):
        label = load_label(path)
        if args.split != "all" and label["split"] != args.split:
            continue
        r = evaluate_run(conn, label, s.runs_dir)
        if r.source != "real" and not args.include_fixtures:
            print(f"skipping {r.video}: run is a {r.source}, not a real model run", file=sys.stderr)
            continue
        results.append(r)
    if not results:
        print("No labelled real runs to evaluate. Add eval/labels/<video>.json for processed videos.")
        return 1
    for split in ("tuning", "heldout"):
        subset = [r for r in results if r.split == split]
        if subset:
            print(format_table(subset, f"{split} set"), end="\n\n")
    if args.json:
        Path(args.json).write_text(json.dumps([r.__dict__ for r in results], default=lambda o: o.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
