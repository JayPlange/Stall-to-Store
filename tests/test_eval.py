import json

import pytest

from app.catalogue import publish, set_status, update_product
from app.fixtures import load_fixture
from app.orders import service as orders
from eval.run_eval import LabelError, Ratio, evaluate_run, format_table, load_label, score_price_links


def _label(run_id, **kw):
    base = {"video": "fx", "split": "tuning", "run_id": run_id,
            "products": [{"label_id": "milk", "brand": "FIXTURE", "name": "Evaporated Milk", "price_pesewas": 1200},
                         {"label_id": "soap", "brand": "FIXTURE", "name": "Bar Soap", "price_pesewas": 800},
                         {"label_id": "lotion", "brand": "FIXTURE", "name": "Body Lotion", "price_pesewas": 3500},
                         {"label_id": "rice", "brand": "FIXTURE", "name": "Rice", "price_pesewas": 4000}],
            "track_to_label": {"t1": "milk", "t3": "soap", "t4": "soap", "t5": "lotion", "t6": None, "t2": None}}
    base.update(kw)
    return base


def test_ratio_empty_is_honest():
    assert Ratio(0, 0).fmt() == "0/0 (n/a)"


def test_metrics_on_fixture_run(conn, settings):
    run = load_fixture(conn, settings.runs_dir)
    r = evaluate_run(conn, _label(run["id"]))
    assert r.source == "fixture"
    assert (r.detection.num, r.detection.den) == (3, 4)          # rice never tracked
    assert (r.duplicates.num, r.duplicates.den) == (1, 3)        # soap has two tracks
    assert (r.identification.num, r.identification.den) == (3, 3)
    # auto: t1 (1200 ok), t2 (unaligned -> wrong), t3, t4 (800 ok) => 3/4
    assert (r.auto_precision.num, r.auto_precision.den) == (3, 4)
    # priced aligned products: t1, t3, t4, t5 -> 3 auto
    assert (r.auto_coverage.num, r.auto_coverage.den) == (3, 4)
    assert any("not aligned" in n for n in r.notes)              # t7 missing from mapping
    assert "FIXTURE DATA - NOT A RESULT" in format_table([r], "tuning set")


def test_corrections_publish_and_order_safety(conn, settings):
    run = load_fixture(conn, settings.runs_dir)
    pid = conn.execute("SELECT id FROM products WHERE run_id = ? AND track_id = 't1'", (run["id"],)).fetchone()[0]
    update_product(conn, run["id"], pid, name="Evap Milk")
    set_status(conn, run["id"], pid, "approved")
    shop = publish(conn, run["id"], "Ama", "+233240000000", "ama")
    o = orders.propose_order(conn, shop, {pid: 1}, ttl_minutes=15)
    orders.seller_confirm(conn, shop["id"], o["reference"])
    r = evaluate_run(conn, _label(run["id"]))
    assert r.corrections == 1 and r.published and r.time_to_shop_s is not None
    assert r.orders_confirmed == 1 and r.order_safety_failures == 0
    conn.execute("UPDATE orders SET total_pesewas = 1")
    conn.commit()
    assert evaluate_run(conn, _label(run["id"])).order_safety_failures == 1


def test_label_validation(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps(_label("r", split="train")))
    with pytest.raises(LabelError):
        load_label(p)
    p.write_text(json.dumps(_label("r", track_to_label={"t1": "nope"})))
    with pytest.raises(LabelError):
        load_label(p)


def test_spark_links_scored_on_exact_tracks(tmp_path):
    art = tmp_path / "artefacts"
    art.mkdir()
    (art / "price_mentions.json").write_text(json.dumps({"mentions": [{"amount_pesewas": 100}, {"amount_pesewas": 200}]}))
    (art / "price_links.json").write_text(json.dumps({"links": [
        {"mention_index": 0, "track_ids": ["a", "b"]}, {"mention_index": 1, "track_ids": ["a"]}]}))
    label = {"products": [{"label_id": "x", "price_pesewas": 100}, {"label_id": "y", "price_pesewas": 200}],
             "track_to_label": {"a": "x", "b": "x", "c": "y"}}
    result = score_price_links(tmp_path, label)
    assert (result.num, result.den) == (1, 2)
    assert (score_price_links(tmp_path / "missing", label).den) == 0
