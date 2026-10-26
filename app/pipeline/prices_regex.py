"""Stage 5a fallback (SPEC section 5): digit amounts near a currency word. Referents are left to the seller."""
import re

from app.models.types import PriceMention, Transcript
from app.money import PriceError, parse_cedis

_AMOUNT = re.compile(
    r"(?i)(?:gh₵|ghs|ghc|₵)\s*(\d+(?:\.\d{1,2})?)"
    r"|(\d+(?:\.\d{1,2})?)\s*(?:cedis?|ghs|gh₵)\b"
    r"|(\d+)\s*(?:cedis?\s*)?(\d{2})\s*pesewas\b")


def find_amounts(text: str) -> list[tuple[str, int]]:
    out = []
    for m in _AMOUNT.finditer(text):
        raw = m.group(1) or m.group(2) or (f"{m.group(3)}.{m.group(4)}" if m.group(3) else None)
        try:
            out.append((m.group(0).strip(), parse_cedis(raw)))
        except PriceError:
            continue
    return out


def regex_mentions(transcript: Transcript) -> list[PriceMention]:
    mentions = []
    for unit_i, unit in enumerate(transcript.units or []):
        for quote, pesewas in find_amounts(unit.text):
            timed = transcript.granularity != "none"
            mentions.append(PriceMention(quote, pesewas, unit.t_start if timed else None,
                                         unit.t_end if timed else None, referent=None, source="regex"))
    if not transcript.units:
        mentions = [PriceMention(q, p, None, None, None, "regex") for q, p in find_amounts(transcript.text)]
    return mentions
