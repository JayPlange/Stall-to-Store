"""WhatsApp click-to-chat links (Stage A).

A wa.me link only opens WhatsApp with the text prefilled. Nothing is delivered until the customer
presses Send; the UI and README must say so.
"""
from urllib.parse import quote

from app.money import format_cedis


def order_message(shop: dict, order: dict) -> str:
    lines = [f"Hello {shop['seller_name']}, I would like to order from your shop.",
             f"Ref: {order['reference']}"]
    for line in order["lines"]:
        lines.append(f"- {line['qty']} x {line['label']} @ {format_cedis(line['unit_price_pesewas'])}"
                     f" = {format_cedis(line['line_total_pesewas'])}")
    lines.append(f"Total: {format_cedis(order['total_pesewas'])}")
    lines.append("Please confirm.")
    return "\n".join(lines)


def wa_link(phone_digits: str, text: str) -> str:
    if not phone_digits.isdigit():
        raise ValueError("phone must be digits only")
    return f"https://wa.me/{phone_digits}?text={quote(text, safe='')}"
