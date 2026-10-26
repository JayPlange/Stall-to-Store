"""Order proposals and confirmation (SPEC section 9). Shared by Stage A (click-to-chat) and Stage B (agent).

- propose_order computes totals from approved prices only and stores a price snapshot with an expiry.
  It writes nothing else.
- Confirmation is the only write path to 'confirmed'. It requires a stored, unexpired proposal whose
  snapshot still matches current approved prices, and it is idempotent.
- Callers pass product ids and quantities; prices never come from a client or a model.
"""
import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta

from app.db import iso, parse_iso, utcnow

REF_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


class OrderError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def new_reference() -> str:
    return "S2S-" + "".join(secrets.choice(REF_ALPHABET) for _ in range(6))


def _normalise_lines(lines, max_lines: int, max_qty: int) -> dict[int, int]:
    items = lines.items() if isinstance(lines, dict) else lines
    merged: dict[int, int] = {}
    for pid, qty in items:
        try:
            pid, qty = int(pid), int(qty)
        except (TypeError, ValueError):
            raise OrderError("invalid_line", "product and quantity must be whole numbers")
        if qty == 0:
            continue
        if qty < 0:
            raise OrderError("invalid_quantity", "quantity cannot be negative")
        merged[pid] = merged.get(pid, 0) + qty
    if not merged:
        raise OrderError("empty_order", "choose at least one product")
    if len(merged) > max_lines:
        raise OrderError("too_many_lines", f"at most {max_lines} different products per order")
    for qty in merged.values():
        if qty > max_qty:
            raise OrderError("invalid_quantity", f"at most {max_qty} of each product")
    return merged


def _approved_products(conn: sqlite3.Connection, run_id: str, ids) -> dict[int, sqlite3.Row]:
    ids = list(ids)
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM products WHERE run_id = ? AND id IN ({marks}) AND status = 'approved'"
                        f" AND price_pesewas IS NOT NULL", (run_id, *ids)).fetchall()
    return {r["id"]: r for r in rows}


def _label(p) -> str:
    return " ".join(x for x in (p["brand"], p["name"], p["size"]) if x)


def propose_order(conn: sqlite3.Connection, shop: dict, lines, *, ttl_minutes: int, channel: str = "click_to_chat",
                  customer_wa: str | None = None, max_lines: int = 30, max_qty: int = 50,
                  now: datetime | None = None) -> dict:
    if not shop.get("published_at"):
        raise OrderError("shop_unpublished", "this shop is not published")
    now = now or utcnow()
    wanted = _normalise_lines(lines, max_lines, max_qty)
    products = _approved_products(conn, shop["run_id"], wanted)
    missing = [pid for pid in wanted if pid not in products]
    if missing:
        # Unknown, hidden, draft or other-shop products all look the same to the customer.
        raise OrderError("unavailable_product", "one or more products are not available")
    snapshot = []
    for pid, qty in wanted.items():
        p = products[pid]
        snapshot.append({"product_id": pid, "label": _label(p), "qty": qty,
                         "unit_price_pesewas": p["price_pesewas"], "line_total_pesewas": p["price_pesewas"] * qty})
    total = sum(line["line_total_pesewas"] for line in snapshot)
    created, expires = iso(now), iso(now + timedelta(minutes=ttl_minutes))
    for _ in range(5):
        reference = new_reference()
        digest = hashlib.sha256(json.dumps(
            {"shop": shop["id"], "ref": reference, "lines": snapshot, "total": total, "created": created,
             "customer": customer_wa, "nonce": secrets.token_hex(16)}, sort_keys=True).encode()).hexdigest()[:40]
        try:
            conn.execute(
                "INSERT INTO orders(shop_id, reference, customer_wa, channel, status, lines_json, total_pesewas,"
                " proposal_hash, created_at, expires_at) VALUES (?,?,?,?, 'proposed', ?,?,?,?,?)",
                (shop["id"], reference, customer_wa, channel, json.dumps(snapshot), total, digest, created, expires))
            conn.commit()
            return get_order(conn, digest)
        except sqlite3.IntegrityError:
            conn.rollback()
    raise OrderError("reference_collision", "could not allocate an order reference")


def get_order(conn: sqlite3.Connection, proposal_hash: str) -> dict | None:
    row = conn.execute("SELECT * FROM orders WHERE proposal_hash = ?", (proposal_hash,)).fetchone()
    return _decode(row)


def get_by_reference(conn: sqlite3.Connection, shop_id: int, reference: str) -> dict | None:
    row = conn.execute("SELECT * FROM orders WHERE shop_id = ? AND reference = ?",
                       (shop_id, (reference or "").strip().upper())).fetchone()
    return _decode(row)


def list_orders(conn: sqlite3.Connection, shop_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM orders WHERE shop_id = ? ORDER BY id DESC LIMIT 50", (shop_id,)).fetchall()
    return [_decode(r) for r in rows]


def _decode(row) -> dict | None:
    if not row:
        return None
    order = dict(row)
    order["lines"] = json.loads(order["lines_json"])
    return order


def snapshot_problems(conn: sqlite3.Connection, run_id: str, order: dict) -> list[str]:
    """Empty list when every line still matches a current approved price."""
    lines = order["lines"]
    current = _approved_products(conn, run_id, [line["product_id"] for line in lines])
    problems = []
    for line in lines:
        p = current.get(line["product_id"])
        if p is None:
            problems.append(f"{line['label']} is no longer available")
        elif p["price_pesewas"] != line["unit_price_pesewas"]:
            problems.append(f"{line['label']} price changed")
    if sum(line["unit_price_pesewas"] * line["qty"] for line in lines) != order["total_pesewas"]:
        problems.append("stored total does not match lines")
    return problems


def _confirm(conn: sqlite3.Connection, order: dict, now: datetime) -> dict:
    if order["status"] == "confirmed":
        return {**order, "already_confirmed": True}
    if order["status"] != "proposed":
        raise OrderError(f"order_{order['status']}", f"this order is {order['status']}")
    if now > parse_iso(order["expires_at"]):
        conn.execute("UPDATE orders SET status = 'expired' WHERE id = ? AND status = 'proposed'", (order["id"],))
        conn.commit()
        raise OrderError("expired", "this order proposal has expired; please order again")
    run_id = conn.execute("SELECT run_id FROM shops WHERE id = ?", (order["shop_id"],)).fetchone()["run_id"]
    problems = snapshot_problems(conn, run_id, order)
    if problems:
        raise OrderError("stale_snapshot", "; ".join(problems))
    cur = conn.execute("UPDATE orders SET status = 'confirmed', confirmed_at = ? WHERE id = ? AND status = 'proposed'",
                       (iso(now), order["id"]))
    conn.commit()
    updated = get_order(conn, order["proposal_hash"])
    return {**updated, "already_confirmed": cur.rowcount == 0}


def confirm_order(conn: sqlite3.Connection, proposal_hash: str, customer_wa: str,
                  now: datetime | None = None) -> dict:
    """Stage B path: the customer who received the proposal confirms it."""
    order = get_order(conn, proposal_hash or "")
    if order is None:
        raise OrderError("unknown_proposal", "no such order proposal")
    if not customer_wa or order["customer_wa"] != customer_wa:
        raise OrderError("not_your_proposal", "this proposal belongs to another customer")
    return _confirm(conn, order, now or utcnow())


def seller_confirm(conn: sqlite3.Connection, shop_id: int, reference: str, now: datetime | None = None) -> dict:
    """Stage A path: the seller confirms a click-to-chat order they received by reference."""
    order = get_by_reference(conn, shop_id, reference)
    if order is None:
        raise OrderError("unknown_reference", "no order with that reference in this shop")
    return _confirm(conn, order, now or utcnow())


def seller_cancel(conn: sqlite3.Connection, shop_id: int, reference: str) -> dict:
    order = get_by_reference(conn, shop_id, reference)
    if order is None:
        raise OrderError("unknown_reference", "no order with that reference in this shop")
    if order["status"] == "confirmed":
        raise OrderError("order_confirmed", "confirmed orders cannot be cancelled here")
    conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ? AND status = 'proposed'", (order["id"],))
    conn.commit()
    return get_by_reference(conn, shop_id, reference)
