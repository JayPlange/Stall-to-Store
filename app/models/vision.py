"""Stages 5a, 5b, 6 adapter: Muse Spark via Chat Completions.

Documented (dev.meta.ai/docs, read 2026-10-26): POST /v1/chat/completions; images as content parts
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}} in user messages, max 50 per
request; response_format {"type": "json_schema", "json_schema": {"name", "schema", "strict"}};
result JSON string in choices[0].message.content; usage has prompt/completion/total_tokens.
Documented but not yet observed live: treat any deviation as AdapterNotVerified, never guess.
"""
import base64
import json
from pathlib import Path

from app.models.base import AdapterNotVerified, CallContext, MetaClient
from app.models.pricing import spark_cost
from app.models.types import Crop, PriceLink, PriceMention, ProductGuess, Transcript
from app.money import PriceError, parse_cedis

MAX_IMAGES_PER_CALL = 8

IDENTIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"products": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"index": {"type": "integer"}, "brand": {"type": ["string", "null"]},
                       "name": {"type": ["string", "null"]}, "size": {"type": ["string", "null"]},
                       "confidence": {"type": "number"}},
        "required": ["index", "brand", "name", "size", "confidence"]}}},
    "required": ["products"],
}

PRICES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"mentions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"unit_start": {"type": "integer"}, "unit_end": {"type": "integer"},
                       "quote": {"type": "string"}, "amount_cedis": {"type": "string"},
                       "referent": {"type": ["string", "null"]}},
        "required": ["unit_start", "unit_end", "quote", "amount_cedis", "referent"]}}},
    "required": ["mentions"],
}


LINKS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"links": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"price_index": {"type": "integer"}, "product_numbers": {"type": "array", "items": {"type": "integer"}},
                       "confidence": {"type": "number"}, "reason": {"type": "string"}},
        "required": ["price_index", "product_numbers", "confidence", "reason"]}}},
    "required": ["links"],
}


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _structured_content(body: object) -> dict:
    try:
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        keys = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
        raise AdapterNotVerified(f"chat completion shape differs from docs; top-level keys: {keys}") from exc


def _response_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


def identify(client: MetaClient, ctx: CallContext, crops: list[Crop], model: str) -> list[ProductGuess]:
    guesses: list[ProductGuess] = []
    for start in range(0, len(crops), MAX_IMAGES_PER_CALL):
        batch = crops[start:start + MAX_IMAGES_PER_CALL]
        content: list[dict] = [{"type": "text", "text": (
            "Each image is one packaged product from a market stall in Ghana, numbered from 0 in order. "
            "For each, read the brand, product name and pack size printed on it. Use null for anything "
            "not legible; do not guess. confidence is 0 to 1.")}]
        content += [{"type": "image_url", "image_url": {"url": _data_url(Path(c.image_path))}} for c in batch]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "response_format": _response_format("products", IDENTIFY_SCHEMA)}
        result = client.request(ctx, model, "POST", "/chat/completions", json_body=body,
                                estimate_cost=lambda b: spark_cost(model, b.get("usage") if isinstance(b, dict) else None))
        parsed = _structured_content(result.body)
        by_index = {p.get("index"): p for p in parsed.get("products", []) if isinstance(p, dict)}
        for i, crop in enumerate(batch):
            p = by_index.get(i, {})
            guesses.append(ProductGuess(crop.track_id, p.get("brand"), p.get("name"), p.get("size"),
                                        p.get("confidence")))
    return guesses


def parse_prices(client: MetaClient, ctx: CallContext, transcript: Transcript, model: str) -> list[PriceMention]:
    """Model picks spans and amounts; timings always come from our transcript units, never the model."""
    if not transcript.units:
        return []
    lines = "\n".join(f"[{i}] {u.text}" for i, u in enumerate(transcript.units))
    prompt = ("A market seller is narrating prices while filming products. Transcript units are numbered.\n"
              f"{lines}\n\nList every price the seller states. unit_start/unit_end are the unit numbers "
              "spanning the statement, quote is the exact words, amount_cedis is the amount in Ghana cedis "
              "as digits (e.g. '12' or '12.50'), referent is the words naming what the price is for, or null.")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "response_format": _response_format("price_mentions", PRICES_SCHEMA)}
    result = client.request(ctx, model, "POST", "/chat/completions", json_body=body,
                                estimate_cost=lambda b: spark_cost(model, b.get("usage") if isinstance(b, dict) else None))
    parsed = _structured_content(result.body)
    mentions = []
    n = len(transcript.units)
    for m in parsed.get("mentions", []):
        try:
            amount = parse_cedis(m["amount_cedis"])
            a, b = int(m["unit_start"]), int(m["unit_end"])
        except (KeyError, TypeError, ValueError, PriceError):
            continue  # unparseable mention: dropped, visible in the raw artefact
        if not (0 <= a <= b < n):
            continue
        timed = transcript.granularity != "none"
        mentions.append(PriceMention(
            text=m.get("quote") or " ".join(u.text for u in transcript.units[a:b + 1]),
            amount_pesewas=amount,
            t_start=transcript.units[a].t_start if timed else None,
            t_end=transcript.units[b].t_end if timed else None,
            referent=m.get("referent"), source="model"))
    return mentions


def match_prices(client: MetaClient, ctx: CallContext, sheet_path: Path, transcript_text: str,
                 mentions: list[PriceMention], products: list[dict], model: str) -> list[PriceLink]:
    """Step 2: link each spoken price to products on a numbered contact sheet.

    products: [{"number", "track_id", "label"}]. The model only chooses numbers; amounts always come from the
    mentions, so it cannot introduce or change a price. Unknown mentions and numbers are discarded.
    """
    listed = "\n".join(f"{p['number']}: {p['label'] or 'unreadable label'}" for p in products)
    spoken = "\n".join(f"[{i}] \"{m.text}\" for: {m.referent or 'not stated'}" for i, m in enumerate(mentions))
    prompt = ("This image shows numbered products from a market stall in Ghana, cut out of the seller's video.\n"
              f"Products as read from their labels:\n{listed}\n\n"
              f"What the seller said:\n{transcript_text}\n\nPrices the seller stated:\n{spoken}\n\n"
              "For each price, give the numbers of the products it belongs to. Speech recognition mishears brand "
              "names, so compare how names sound and what the product is, and look at the images. One price can "
              "apply to several numbered items when they are the same product. Use an empty list when no product "
              "shown matches. confidence is 0 to 1. Do not change any amount.")
    content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": _data_url(sheet_path)}}]
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "response_format": _response_format("price_links", LINKS_SCHEMA)}
    result = client.request(ctx, model, "POST", "/chat/completions", json_body=body,
                            estimate_cost=lambda b: spark_cost(model, b.get("usage") if isinstance(b, dict) else None))
    parsed = _structured_content(result.body)
    by_number = {p["number"]: p["track_id"] for p in products}
    links = {i: PriceLink(i, [], None) for i in range(len(mentions))}
    for item in parsed.get("links", []):
        i = item.get("price_index") if isinstance(item, dict) else None
        if i in links:
            ids = [by_number[n] for n in item.get("product_numbers", []) if n in by_number]
            links[i] = PriceLink(i, list(dict.fromkeys(ids)), item.get("confidence"), item.get("reason"))
    return list(links.values())
