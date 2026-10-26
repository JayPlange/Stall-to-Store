"""Cedis <-> pesewas. Integer pesewas everywhere; no floats touch money."""
import re
from decimal import Decimal, InvalidOperation

MAX_PESEWAS = 100_000_000  # GH₵1,000,000


class PriceError(ValueError):
    pass


_STRIP = re.compile(r"(?i)\s|gh₵|ghs|gh¢|ghc|₵|¢|cedis?|,")


def parse_cedis(value: str | int | None) -> int:
    """'12', '12.5', 'GH₵ 1,200.00' -> pesewas. Rejects zero, negatives, >2 decimals."""
    if value is None:
        raise PriceError("price is required")
    text = _STRIP.sub("", str(value))
    if not text:
        raise PriceError("price is required")
    if not re.fullmatch(r"\d+(\.\d{1,2})?", text):
        raise PriceError("price must be a number of cedis with at most 2 decimals")
    try:
        pesewas = int(Decimal(text) * 100)
    except InvalidOperation as exc:
        raise PriceError("invalid price") from exc
    if pesewas <= 0:
        raise PriceError("price must be greater than zero")
    if pesewas > MAX_PESEWAS:
        raise PriceError("price is too large")
    return pesewas


def format_cedis(pesewas: int | None) -> str:
    if pesewas is None:
        return "—"
    return f"GH₵{pesewas // 100:,}.{pesewas % 100:02d}"


def cedis_input_value(pesewas: int | None) -> str:
    return "" if pesewas is None else f"{pesewas // 100}.{pesewas % 100:02d}"
