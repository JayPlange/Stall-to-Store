"""Runs, draft products, seller review and publishing.

Invariants:
- Only approved products with a price appear on a shop page.
- Any seller edit to an approved product returns it to draft (it must be re-approved).
- Every seller edit is logged in product_edits (the "seller corrections" metric).
"""
import hmac
import json
import re
import secrets
import sqlite3
from dataclasses import dataclass

from app.artefacts import new_run_id
from app.db import iso, utcnow
from app.money import PriceError, parse_cedis
from app.pipeline.associate import MentionDecision, TrackDecision, name_similarity
from app.models.types import PriceMention

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,38})[a-z0-9]$")
RESERVED_SLUGS = {"admin", "api", "static", "health", "review", "dev", "new"}
EDITABLE = ("brand", "name", "size")
MAX_TEXT = 120


class CatalogueError(ValueError):
    pass


@dataclass
class ProductProposal:
    track_id: str
    brand: str | None
    name: str | None
    size: str | None
    keyframe_path: str | None
    model_confidence: float | None = None
    cutout_path: str | None = None


# -- runs -----------------------------------------------------------------
def create_run(conn: sqlite3.Connection, source: str = "real", run_id: str | None = None,
               video_path: str | None = None, meta: dict | None = None) -> dict:
    run_id = run_id or new_run_id()
    token = secrets.token_urlsafe(24)
    conn.execute("INSERT INTO runs(id, created_at, video_path, status, stage, source, review_token, meta_json)"
                 " VALUES (?,?,?,?,?,?,?,?)",
                 (run_id, iso(utcnow()), video_path, "created", None, source, token, json.dumps(meta or {})))
    conn.commit()
    return get_run(conn, run_id)


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def run_by_token(conn: sqlite3.Connection, token: str) -> dict | None:
    if not token or len(token) > 64:
        return None
    row = conn.execute("SELECT * FROM runs WHERE review_token = ?", (token,)).fetchone()
    # Constant-time confirm after the indexed lookup.
    if row and hmac.compare_digest(row["review_token"], token):
        return dict(row)
    return None


def set_run_status(conn: sqlite3.Connection, run_id: str, status: str, stage: str | None = None,
                   error: str | None = None) -> None:
    conn.execute("UPDATE runs SET status = ?, stage = ?, error = ? WHERE id = ?", (status, stage, error, run_id))
    conn.commit()


# -- drafts from the pipeline ---------------------------------------------
def store_draft(conn: sqlite3.Connection, run_id: str, proposals: list[ProductProposal],
                mentions: list[PriceMention], decisions: list[MentionDecision],
                track_decisions: dict[str, TrackDecision]) -> dict[str, int]:
    """Write products, mentions and candidates for a run. Returns track_id -> product id."""
    if conn.execute("SELECT 1 FROM products WHERE run_id = ?", (run_id,)).fetchone():
        raise CatalogueError("run already has draft products")
    now = iso(utcnow())
    ids: dict[str, int] = {}
    for p in proposals:
        td = track_decisions.get(p.track_id)
        status = td.match_status if td else "no_price"
        price = td.price_pesewas if td and status == "auto" else None
        cur = conn.execute(
            "INSERT INTO products(run_id, track_id, brand, name, size, keyframe_path, cutout_path,"
            " price_pesewas, price_source, match_status, match_confidence, match_note, status,"
            " model_brand, model_name, model_size, model_price_pesewas, model_confidence, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'draft',?,?,?,?,?,?)",
            (run_id, p.track_id, p.brand, p.name, p.size, p.keyframe_path, p.cutout_path, price,
             "auto" if price else "none", status, td.confidence if td else None, td.note if td else None,
             p.brand, p.name, p.size, price, p.model_confidence, now))
        ids[p.track_id] = cur.lastrowid
    mention_ids = []
    for i, m in enumerate(mentions):
        td_auto = next((ids[tid] for tid in decisions[i].auto_track_ids if tid in ids), None) \
            if i < len(decisions) else None
        cur = conn.execute(
            "INSERT INTO price_mentions(run_id, text, amount_pesewas, t_start, t_end, referent, source,"
            " assigned_product_id) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, m.text, m.amount_pesewas, m.t_start, m.t_end, m.referent, m.source, td_auto))
        mention_ids.append(cur.lastrowid)
    for d in decisions:
        for rank, c in enumerate(d.candidates, start=1):
            if c.track_id in ids:
                conn.execute("INSERT INTO candidates(mention_id, product_id, rank, score) VALUES (?,?,?,?)",
                             (mention_ids[d.mention_index], ids[c.track_id], rank, c.score))
    conn.execute("UPDATE runs SET status = 'draft_ready', stage = 'associate' WHERE id = ?", (run_id,))
    conn.commit()
    return ids


# -- reading --------------------------------------------------------------
def list_products(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM products WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def get_product(conn: sqlite3.Connection, run_id: str, product_id: int) -> dict:
    row = conn.execute("SELECT * FROM products WHERE id = ? AND run_id = ?", (product_id, run_id)).fetchone()
    if not row:
        raise CatalogueError("unknown product")
    return dict(row)


def suggestions(conn: sqlite3.Connection, product_id: int) -> list[dict]:
    """Every price heard in this run, the likeliest for this product first. The seller decides."""
    p = conn.execute("SELECT run_id, brand, name FROM products WHERE id = ?", (product_id,)).fetchone()
    rows = conn.execute("SELECT id, text, amount_pesewas, t_start, source, referent FROM price_mentions"
                        " WHERE run_id = ? ORDER BY id", (p["run_id"],)).fetchall()
    label = " ".join(x for x in (p["brand"], p["name"]) if x)
    return sorted((dict(r) for r in rows), key=lambda m: -name_similarity(m["referent"], label))


# -- seller edits ---------------------------------------------------------
def _log_edit(conn, product_id: int, field: str, old, new) -> None:
    conn.execute("INSERT INTO product_edits(product_id, field, old_value, new_value, created_at) VALUES (?,?,?,?,?)",
                 (product_id, field, None if old is None else str(old), None if new is None else str(new),
                  iso(utcnow())))


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(value.split())
    if len(value) > MAX_TEXT:
        raise CatalogueError(f"text longer than {MAX_TEXT} characters")
    return value or None


def update_product(conn: sqlite3.Connection, run_id: str, product_id: int, *, brand: str | None = None,
                   name: str | None = None, size: str | None = None, price: str | None = None) -> dict:
    """Apply seller corrections. Fields left as None are unchanged; '' clears text fields."""
    product = get_product(conn, run_id, product_id)
    changes: dict[str, object] = {}
    for field, value in (("brand", brand), ("name", name), ("size", size)):
        if value is not None:
            cleaned = _clean_text(value)
            if cleaned != product[field]:
                changes[field] = cleaned
    if price is not None and price.strip():
        try:
            pesewas = parse_cedis(price)
        except PriceError as exc:
            raise CatalogueError(str(exc)) from exc
        if pesewas != product["price_pesewas"]:
            changes["price_pesewas"] = pesewas
            changes["price_source"] = "seller"
    if not changes:
        return product
    for field, value in changes.items():
        if field != "price_source":
            _log_edit(conn, product_id, field, product[field], value)
    if product["status"] == "approved":
        changes["status"] = "draft"
    changes["updated_at"] = iso(utcnow())
    cols = ", ".join(f"{k} = ?" for k in changes)
    conn.execute(f"UPDATE products SET {cols} WHERE id = ? AND run_id = ?", (*changes.values(), product_id, run_id))
    conn.commit()
    return get_product(conn, run_id, product_id)


def use_suggestion(conn: sqlite3.Connection, run_id: str, product_id: int, mention_id: int) -> dict:
    get_product(conn, run_id, product_id)
    row = conn.execute("SELECT amount_pesewas FROM price_mentions WHERE id = ? AND run_id = ?",
                       (mention_id, run_id)).fetchone()
    if not row:
        raise CatalogueError("that price was not heard in this video")
    pesewas = row["amount_pesewas"]
    return update_product(conn, run_id, product_id, price=f"{pesewas // 100}.{pesewas % 100:02d}")


def set_status(conn: sqlite3.Connection, run_id: str, product_id: int, status: str) -> dict:
    if status not in ("draft", "approved", "hidden"):
        raise CatalogueError("invalid status")
    product = get_product(conn, run_id, product_id)
    if status == "approved":
        if not product["name"]:
            raise CatalogueError("a product needs a name before approval")
        if not product["price_pesewas"]:
            raise CatalogueError("a product needs a price before approval")
    if product["status"] != status:
        _log_edit(conn, product_id, "status", product["status"], status)
        conn.execute("UPDATE products SET status = ?, updated_at = ? WHERE id = ?",
                     (status, iso(utcnow()), product_id))
        conn.commit()
    return get_product(conn, run_id, product_id)


# -- publishing -----------------------------------------------------------
def normalise_whatsapp(value: str) -> str:
    """Digits only, international format. Ghana local '0XXXXXXXXX' becomes '233XXXXXXXXX'."""
    digits = re.sub(r"[\s\-().]", "", value or "")
    if digits.startswith("+"):
        digits = digits[1:]
    elif digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 10:
        digits = "233" + digits[1:]
    if not digits.isdigit() or not 8 <= len(digits) <= 15:
        raise CatalogueError("enter a WhatsApp number in international format, e.g. +233 20 000 0001")
    return digits


def validate_slug(slug: str) -> str:
    slug = (slug or "").strip().lower()
    if not SLUG_RE.match(slug) or "--" in slug or slug in RESERVED_SLUGS:
        raise CatalogueError("shop link must be 3 to 40 letters, numbers or hyphens")
    return slug


def get_shop_for_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM shops WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_shop_by_slug(conn: sqlite3.Connection, slug: str) -> dict | None:
    row = conn.execute("SELECT s.*, r.source FROM shops s JOIN runs r ON r.id = s.run_id"
                       " WHERE s.slug = ? AND s.published_at IS NOT NULL", (slug,)).fetchone()
    return dict(row) if row else None


def publish(conn: sqlite3.Connection, run_id: str, seller_name: str, seller_whatsapp: str, slug: str) -> dict:
    name = _clean_text(seller_name)
    if not name:
        raise CatalogueError("enter the shop or seller name")
    wa = normalise_whatsapp(seller_whatsapp)
    slug = validate_slug(slug)
    approved = conn.execute("SELECT COUNT(*) FROM products WHERE run_id = ? AND status = 'approved'"
                            " AND price_pesewas IS NOT NULL AND name IS NOT NULL", (run_id,)).fetchone()[0]
    if approved == 0:
        raise CatalogueError("approve at least one product before publishing")
    other = conn.execute("SELECT run_id FROM shops WHERE slug = ?", (slug,)).fetchone()
    if other and other["run_id"] != run_id:
        raise CatalogueError("that shop link is taken")
    now = iso(utcnow())
    if get_shop_for_run(conn, run_id):
        conn.execute("UPDATE shops SET slug = ?, seller_name = ?, seller_whatsapp = ?, published_at = ?"
                     " WHERE run_id = ?", (slug, name, wa, now, run_id))
    else:
        conn.execute("INSERT INTO shops(run_id, slug, seller_name, seller_whatsapp, published_at)"
                     " VALUES (?,?,?,?,?)", (run_id, slug, name, wa, now))
    conn.execute("UPDATE runs SET status = 'published', stage = 'publish' WHERE id = ?", (run_id,))
    conn.commit()
    return get_shop_for_run(conn, run_id)


def shop_products(conn: sqlite3.Connection, shop: dict) -> list[dict]:
    """The ONLY query that feeds the public shop page."""
    rows = conn.execute("SELECT * FROM products WHERE run_id = ? AND status = 'approved'"
                        " AND price_pesewas IS NOT NULL AND name IS NOT NULL ORDER BY id",
                        (shop["run_id"],)).fetchall()
    return [dict(r) for r in rows]
