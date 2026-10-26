import pytest

from app.catalogue import (CatalogueError, get_product, normalise_whatsapp, publish, set_status, shop_products,
                           update_product, validate_slug)
from tests.conftest import make_draft


def test_approval_requires_name_and_price(conn):
    d = make_draft(conn, [{"name": "Milk"}, {"name": "", "price": 100}])
    rid = d["run"]["id"]
    with pytest.raises(CatalogueError, match="price"):
        set_status(conn, rid, d["ids"][0], "approved")
    conn.execute("UPDATE products SET name = NULL WHERE id = ?", (d["ids"][1],))
    with pytest.raises(CatalogueError, match="name"):
        set_status(conn, rid, d["ids"][1], "approved")


def test_seller_edit_is_logged_and_reverts_approval(conn):
    d = make_draft(conn, [{"name": "Milk", "price": 1200}])
    rid, pid = d["run"]["id"], d["ids"][0]
    set_status(conn, rid, pid, "approved")
    p = update_product(conn, rid, pid, price="13")
    assert p["price_pesewas"] == 1300 and p["price_source"] == "seller" and p["status"] == "draft"
    fields = [r["field"] for r in conn.execute("SELECT field FROM product_edits WHERE product_id = ?", (pid,))]
    assert "price_pesewas" in fields


def test_repeated_identical_edit_is_a_noop(conn):
    d = make_draft(conn, [{"name": "Milk", "price": 1200}])
    rid, pid = d["run"]["id"], d["ids"][0]
    update_product(conn, rid, pid, name="Milk", price="12.00")
    set_status(conn, rid, pid, "approved")
    set_status(conn, rid, pid, "approved")
    edits = conn.execute("SELECT COUNT(*) FROM product_edits WHERE product_id = ?", (pid,)).fetchone()[0]
    assert edits == 1  # only the single status change
    assert get_product(conn, rid, pid)["status"] == "approved"


def test_invalid_price_rejected(conn):
    d = make_draft(conn, [{"name": "Milk", "price": 1200}])
    with pytest.raises(CatalogueError):
        update_product(conn, d["run"]["id"], d["ids"][0], price="-5")


def test_product_from_other_run_is_unknown(conn):
    a = make_draft(conn, [{"name": "Milk", "price": 1200}])
    b = make_draft(conn, [{"name": "Soap", "price": 100}])
    with pytest.raises(CatalogueError, match="unknown"):
        update_product(conn, a["run"]["id"], b["ids"][0], name="hijack")


def test_publish_requires_an_approved_product(conn):
    d = make_draft(conn, [{"name": "Milk", "price": 1200}])
    with pytest.raises(CatalogueError, match="approve"):
        publish(conn, d["run"]["id"], "Ama", "+233240000000", "ama-shop")


def test_shop_shows_only_approved(published, conn):
    names = [p["name"] for p in shop_products(conn, published["shop"])]
    assert names == ["Milk", "Soap"]


def test_slug_taken_by_other_run(published, conn):
    d = make_draft(conn, [{"name": "X", "price": 100}])
    set_status(conn, d["run"]["id"], d["ids"][0], "approved")
    with pytest.raises(CatalogueError, match="taken"):
        publish(conn, d["run"]["id"], "Other", "+233240000001", "auntie-ama")


@pytest.mark.parametrize("raw,digits", [("+233 20 000 0001", "233200000001"), ("0200000001", "233200000001"),
                                        ("00233200000001", "233200000001")])
def test_whatsapp_normalise(raw, digits):
    assert normalise_whatsapp(raw) == digits


@pytest.mark.parametrize("raw", ["", "abc", "+12", "+2332000000018901"])
def test_whatsapp_rejects(raw):
    with pytest.raises(CatalogueError):
        normalise_whatsapp(raw)


@pytest.mark.parametrize("slug", ["ab", "-abc", "abc-", "a--b", "admin", "Has Space", "x" * 41, "../x"])
def test_bad_slugs(slug):
    with pytest.raises(CatalogueError):
        validate_slug(slug)
