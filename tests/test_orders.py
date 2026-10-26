"""SPEC section 9 required order-safety tests, plus totals and input validation."""
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from app.catalogue import set_status, update_product
from app.db import utcnow
from app.orders import service as orders
from app.orders.links import order_message, wa_link

TTL = 15


def propose(conn, published, lines, **kw):
    return orders.propose_order(conn, published["shop"], lines, ttl_minutes=TTL, **kw)


def test_total_computed_from_approved_prices(conn, published):
    milk, soap = published["ids"][:2]
    o = propose(conn, published, {milk: 2, soap: 3})
    assert o["total_pesewas"] == 2 * 1200 + 3 * 550
    assert [l["unit_price_pesewas"] for l in o["lines"]] == [1200, 550]
    assert o["status"] == "proposed" and o["reference"].startswith("S2S-")


def test_duplicate_lines_merge_and_zero_qty_ignored(conn, published):
    milk, soap = published["ids"][:2]
    o = propose(conn, published, [(milk, 1), (milk, 2), (soap, 0)])
    assert len(o["lines"]) == 1 and o["lines"][0]["qty"] == 3


@pytest.mark.parametrize("lines,code", [({}, "empty_order"), ({1: -1}, "invalid_quantity"),
                                        ({1: 51}, "invalid_quantity"), ([("x", 1)], "invalid_line")])
def test_invalid_lines(conn, published, lines, code):
    lines = {published["ids"][0]: list(lines.values())[0]} if isinstance(lines, dict) and lines else lines
    with pytest.raises(orders.OrderError) as e:
        propose(conn, published, lines)
    assert e.value.code == code


def test_ordering_hidden_product_rejected(conn, published):
    with pytest.raises(orders.OrderError) as e:
        propose(conn, published, {published["ids"][2]: 1})
    assert e.value.code == "unavailable_product"


def test_ordering_draft_or_unknown_product_rejected(conn, published):
    for pid in (published["ids"][3], 999_999):
        with pytest.raises(orders.OrderError):
            propose(conn, published, {pid: 1})


def test_confirm_without_proposal(conn, published):
    with pytest.raises(orders.OrderError) as e:
        orders.confirm_order(conn, "not-a-hash", "233200000000")
    assert e.value.code == "unknown_proposal"
    with pytest.raises(orders.OrderError) as e:
        orders.seller_confirm(conn, published["shop"]["id"], "S2S-NOPE00")
    assert e.value.code == "unknown_reference"


def test_expired_proposal(conn, published):
    o = propose(conn, published, {published["ids"][0]: 1}, customer_wa="233200000000", channel="agent")
    with pytest.raises(orders.OrderError) as e:
        orders.confirm_order(conn, o["proposal_hash"], "233200000000", now=utcnow() + timedelta(minutes=TTL + 1))
    assert e.value.code == "expired"
    assert orders.get_order(conn, o["proposal_hash"])["status"] == "expired"
    with pytest.raises(orders.OrderError):  # stays dead
        orders.confirm_order(conn, o["proposal_hash"], "233200000000")


def test_another_customers_proposal(conn, published):
    o = propose(conn, published, {published["ids"][0]: 1}, customer_wa="233200000000", channel="agent")
    with pytest.raises(orders.OrderError) as e:
        orders.confirm_order(conn, o["proposal_hash"], "233299999999")
    assert e.value.code == "not_your_proposal"
    # A click-to-chat proposal has no bound customer, so no customer can confirm it via the agent path.
    web = propose(conn, published, {published["ids"][0]: 1})
    with pytest.raises(orders.OrderError):
        orders.confirm_order(conn, web["proposal_hash"], "233200000000")


def test_duplicate_confirmation_is_idempotent(conn, published):
    o = propose(conn, published, {published["ids"][0]: 1}, customer_wa="233200000000", channel="agent")
    first = orders.confirm_order(conn, o["proposal_hash"], "233200000000")
    second = orders.confirm_order(conn, o["proposal_hash"], "233200000000")
    assert first["status"] == second["status"] == "confirmed"
    assert first["already_confirmed"] is False and second["already_confirmed"] is True
    assert first["confirmed_at"] == second["confirmed_at"]


def test_price_changed_after_proposal(conn, published):
    rid, milk = published["run"]["id"], published["ids"][0]
    o = propose(conn, published, {milk: 2})
    update_product(conn, rid, milk, price="15")
    set_status(conn, rid, milk, "approved")
    with pytest.raises(orders.OrderError) as e:
        orders.seller_confirm(conn, published["shop"]["id"], o["reference"])
    assert e.value.code == "stale_snapshot"
    assert orders.get_order(conn, o["proposal_hash"])["status"] == "proposed"


def test_product_hidden_after_proposal(conn, published):
    rid, milk = published["run"]["id"], published["ids"][0]
    o = propose(conn, published, {milk: 1})
    set_status(conn, rid, milk, "hidden")
    with pytest.raises(orders.OrderError) as e:
        orders.seller_confirm(conn, published["shop"]["id"], o["reference"])
    assert e.value.code == "stale_snapshot"


def test_tampered_total_is_caught(conn, published):
    o = propose(conn, published, {published["ids"][0]: 1})
    conn.execute("UPDATE orders SET total_pesewas = 1 WHERE id = ?", (o["id"],))
    conn.commit()
    with pytest.raises(orders.OrderError) as e:
        orders.seller_confirm(conn, published["shop"]["id"], o["reference"])
    assert e.value.code == "stale_snapshot"


def test_cancelled_cannot_be_confirmed(conn, published):
    o = propose(conn, published, {published["ids"][0]: 1})
    orders.seller_cancel(conn, published["shop"]["id"], o["reference"])
    with pytest.raises(orders.OrderError) as e:
        orders.seller_confirm(conn, published["shop"]["id"], o["reference"])
    assert e.value.code == "order_cancelled"


def test_reference_scoped_to_shop(conn, published):
    o = propose(conn, published, {published["ids"][0]: 1})
    with pytest.raises(orders.OrderError):
        orders.seller_confirm(conn, published["shop"]["id"] + 1, o["reference"])


def test_wa_link_contains_reference_and_total(conn, published):
    o = propose(conn, published, {published["ids"][0]: 2})
    link = wa_link(published["shop"]["seller_whatsapp"], order_message(published["shop"], o))
    parsed = urlparse(link)
    assert parsed.netloc == "wa.me" and parsed.path == "/233240000000"
    text = parse_qs(parsed.query)["text"][0]
    assert o["reference"] in text and "Total: GH₵24.00" in text and "2 x Acme Milk @ GH₵12.00" in text
    with pytest.raises(ValueError):
        wa_link("+233", "x")
