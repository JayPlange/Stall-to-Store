"""Stage 6: price association (SPEC section 7). Pure, deterministic, model-free.

The scorer proposes candidates; it is never the source of truth. Auto-matches are still shown to
the seller, and nothing is published without seller approval.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.models.types import Candidate, Granularity, PriceLink, PriceMention, RankedCandidate, Track

WORD_PRE_S = 1.5
WORD_POST_S = 1.0
TOP_K = 4
# Placeholder thresholds: tune on tuning videos only (SPEC section 7), never on held-out videos.
AUTO_MARGIN = 0.15
AUTO_MIN_SCORE = 0.35
NAME_MATCH_RATIO = 0.6
NAME_BONUS = 0.3
PLURAL_REFERENTS = {"these", "those", "them", "ones", "all", "each", "both"}
# Words that say nothing about which product is meant.
FILLER_WORDS = {"the", "and", "for", "with", "this", "that", "here", "there", "have", "has", "which", "goes",
                "you", "know", "mean", "also", "over", "our", "from", "are", "was", "one"}
# Category words shared across brands: "water" must not link Bel-Aqua to a "Water Kettle". They can't prove a
# match alone, so a name match needs a brand-like word. Reviewed on tuning videos only, like the thresholds.
GENERIC_PRODUCT_WORDS = {"water", "kettle", "baby", "wipe", "wipes", "milk", "powder", "tea", "soap", "mineral",
                         "premium", "still", "classic", "blend", "concentrated", "dairy", "non", "creamer",
                         "fabric", "conditioner", "softener", "softeners", "washing", "multi", "purpose",
                         "scourer", "scorer", "cream", "lotion", "oil"}
MIN_ID_CONFIDENCE = 0.3
LINK_AUTO_CONFIDENCE = 0.8  # placeholder: tune on tuning videos only
# Merging SAM prompts: one request per noun phrase can find the same object twice.
MIN_TRACK_HITS = 5
DUPLICATE_OVERLAP = 0.6
MIN_SHARED_FRAMES = 3


@dataclass
class MentionDecision:
    mention_index: int
    candidates: list[Candidate]
    auto_track_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class TrackDecision:
    track_id: str
    match_status: str  # auto | needs_seller | no_price
    price_pesewas: int | None = None
    confidence: float | None = None
    mention_index: int | None = None
    note: str | None = None
    suggested: list[int] = field(default_factory=list)  # mention indices offered to the seller


def mention_window(m: PriceMention, granularity: Granularity) -> tuple[float, float] | None:
    if granularity == "none" or m.t_start is None or m.t_end is None:
        return None
    if granularity == "word":
        return (m.t_start - WORD_PRE_S, m.t_end + WORD_POST_S)
    return (m.t_start, m.t_end)


def _tokens(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _distinctive(text: str | None) -> list[str]:
    ignored = PLURAL_REFERENTS | FILLER_WORDS | GENERIC_PRODUCT_WORDS
    return [t for t in _tokens(text) if len(t) >= 3 and t not in ignored]


def name_similarity(referent: str | None, name: str | None) -> float:
    """Best fuzzy match between brand-like words. Filler and category words never count."""
    ref, nm = _distinctive(referent), _distinctive(name)
    if not ref or not nm:
        return 0.0
    # Speech splits brand names ("soft care" for "Softcare"), so try adjacent words joined too.
    spoken = ref + [a + b for a, b in zip(ref, ref[1:])]
    return max(SequenceMatcher(None, r, n).ratio() for r in spoken for n in nm)


def _inside(a, b) -> float:
    """Share of box a that lies inside box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0, w) * max(0, h) / area if area > 0 else 0.0


def _duplicates(track: Track, other: Track) -> bool:
    boxes = {h.frame_idx: h.box_xyxy for h in other.frames}
    shared = [(h.box_xyxy, boxes[h.frame_idx]) for h in track.frames if h.frame_idx in boxes]
    if len(shared) < MIN_SHARED_FRAMES:
        return False
    return sum(_inside(a, b) for a, b in shared) / len(shared) >= DUPLICATE_OVERLAP


def merge_tracks(tracks: list[Track]) -> tuple[list[Track], dict[str, str]]:
    """Combine tracks from several prompts, earlier prompts winning. Returns kept tracks and why others left.

    Tracks from the same prompt are never merged: SAM already separated them, so three sachets stay three.
    """
    kept, dropped = [], {}
    for track in tracks:
        prompt = track.track_id.split("o")[0]
        twin = next((k for k in kept if k.track_id.split("o")[0] != prompt and _duplicates(track, k)), None)
        if len(track.frames) < MIN_TRACK_HITS:
            dropped[track.track_id] = f"only {len(track.frames)} frames"
        elif twin:
            dropped[track.track_id] = f"same object as {twin.track_id}"
        else:
            kept.append(track)
    return kept, dropped


def apply_price_links(decisions: list[MentionDecision], tracks: dict[str, TrackDecision],
                      mentions: list[PriceMention], links: list[PriceLink]) -> None:
    """Fold Muse Spark's proposals into the decisions. Amounts always come from the mentions.

    A confident link auto-matches, still subject to seller approval. Anything uncertain or contradictory goes to
    the seller, and Publishing never depends on this step.
    """
    proposed: dict[str, list[tuple[int, PriceLink]]] = {}
    for link in links:
        if link.confidence is not None and link.confidence >= LINK_AUTO_CONFIDENCE:
            for tid in link.track_ids:
                proposed.setdefault(tid, []).append((link.mention_index, link))
        else:
            for tid in link.track_ids:
                if tid in tracks and link.mention_index not in tracks[tid].suggested:
                    tracks[tid].suggested.append(link.mention_index)
    for tid, items in proposed.items():
        td = tracks.get(tid)
        if td is None:
            continue
        amounts = {mentions[i].amount_pesewas for i, _ in items}
        if td.match_status == "auto":
            amounts.add(td.price_pesewas)
        if len(amounts) > 1:
            td.match_status, td.price_pesewas, td.confidence = "needs_seller", None, None
            td.note = "Different prices were linked to this product: please check"
            td.suggested.extend(i for i, _ in items if i not in td.suggested)
            continue
        index, link = items[-1]
        td.match_status, td.price_pesewas, td.confidence = "auto", mentions[index].amount_pesewas, link.confidence
        td.mention_index = index
        td.note = f"Matched by Muse Spark: {link.reason}" if link.reason else "Matched by Muse Spark"
    for d in decisions:
        d.auto_track_ids = [tid for tid, td in tracks.items() if td.match_status == "auto" and td.mention_index == d.mention_index]


def is_identifiable(brand: str | None, name: str | None, confidence: float | None) -> bool:
    """False only when the model read nothing and is confident there is nothing to read (e.g. floor tile)."""
    return bool(brand or name) or confidence is None or confidence >= MIN_ID_CONFIDENCE


def score_track(track: Track, window: tuple[float, float], fps: float, frame_w: int, frame_h: int,
                referent: str | None, product_name: str | None) -> Candidate | None:
    t0, t1 = window
    hits = [h for h in track.frames if t0 <= h.t <= t1]
    if not hits:
        return None
    expected = max(1, round((t1 - t0) * fps))
    visibility = min(1.0, len(hits) / expected)
    cx_f, cy_f = frame_w / 2, frame_h / 2
    max_dist = (cx_f ** 2 + cy_f ** 2) ** 0.5 or 1.0
    cents, sizes = [], []
    for h in hits:
        x0, y0, x1, y1 = h.box_xyxy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cents.append(1 - min(1.0, ((cx - cx_f) ** 2 + (cy - cy_f) ** 2) ** 0.5 / max_dist))
        sizes.append(max(0.0, (x1 - x0) * (y1 - y0)) / max(1, frame_w * frame_h))
    centrality = sum(cents) / len(cents)
    size = min(1.0, (sum(sizes) / len(sizes)) * 4)  # a box covering a quarter of the frame saturates
    bonus = NAME_BONUS if name_similarity(referent, product_name) >= NAME_MATCH_RATIO else 0.0
    score = 0.5 * visibility + 0.25 * centrality + 0.15 * size + bonus
    return Candidate(track.track_id, round(score, 4), round(visibility, 4), round(centrality, 4),
                     round(size, 4), bonus)


def candidates_for(m: PriceMention, tracks: list[Track], granularity: Granularity, fps: float,
                   frame_w: int, frame_h: int, names: dict[str, str | None]) -> list[Candidate]:
    window = mention_window(m, granularity)
    if window is None:
        # No timing: every track is a candidate, nothing is auto-matched.
        return [Candidate(t.track_id, 0.0) for t in tracks]
    scored = [c for t in tracks
              if (c := score_track(t, window, fps, frame_w, frame_h, m.referent, names.get(t.track_id)))]
    scored.sort(key=lambda c: (-c.score, -c.visibility, -c.centrality, -c.size, c.track_id))
    return scored[:TOP_K]


def _is_plural(referent: str | None) -> bool:
    return any(t in PLURAL_REFERENTS for t in _tokens(referent))


def _same_product(a: str | None, b: str | None) -> bool:
    return bool(a and b and _tokens(a) == _tokens(b))


def decide_mention(idx: int, m: PriceMention, cands: list[Candidate], granularity: Granularity,
                   names: dict[str, str | None], ranked: list[RankedCandidate] | None = None) -> MentionDecision:
    if granularity == "none" or m.t_start is None:
        return MentionDecision(idx, cands, reason="no timing")
    if not cands:
        return MentionDecision(idx, cands, reason="no track visible")
    top = cands[0]
    runner = cands[1].score if len(cands) > 1 else 0.0
    if top.score < AUTO_MIN_SCORE:
        return MentionDecision(idx, cands, reason="weak evidence")
    if ranked:
        model_top = min(ranked, key=lambda r: r.rank).track_id
        if model_top != top.track_id:
            return MentionDecision(idx, cands, reason="model disagrees")
    same = [c.track_id for c in cands[1:] if _same_product(names.get(c.track_id), names.get(top.track_id))]
    if _is_plural(m.referent) and same:
        # "these ones, 12 cedis" across identical items: one price, several tracks.
        return MentionDecision(idx, cands, [top.track_id, *same], reason="plural, identical items")
    if top.score - runner < AUTO_MARGIN:
        return MentionDecision(idx, cands, reason="candidates too close")
    return MentionDecision(idx, cands, [top.track_id], reason="clear winner")


def associate(mentions: list[PriceMention], tracks: list[Track], granularity: Granularity, fps: float,
              frame_w: int, frame_h: int, names: dict[str, str | None],
              ranked: dict[int, list[RankedCandidate]] | None = None) -> tuple[list[MentionDecision], dict[str, TrackDecision]]:
    """Many-to-one matching. Returns per-mention decisions and a decision for every track."""
    ranked = ranked or {}
    decisions = []
    for i, m in enumerate(mentions):
        cands = candidates_for(m, tracks, granularity, fps, frame_w, frame_h, names)
        decisions.append(decide_mention(i, m, cands, granularity, names, ranked.get(i)))

    out = {t.track_id: TrackDecision(t.track_id, "no_price") for t in tracks}
    # Process in time order so the latest clear price wins.
    order = sorted(range(len(mentions)), key=lambda i: (mentions[i].t_start is None, mentions[i].t_start or 0, i))
    for i in order:
        d, m = decisions[i], mentions[i]
        for c in d.candidates:
            td = out.get(c.track_id)
            if td and i not in td.suggested:
                td.suggested.append(i)
        for tid in d.auto_track_ids:
            td = out[tid]
            conf = next((c.score for c in d.candidates if c.track_id == tid), None)
            if td.match_status == "auto" and td.price_pesewas != m.amount_pesewas:
                td.note = (f"conflict: earlier mention said {td.price_pesewas} pesewas; "
                           f"kept latest {m.amount_pesewas}")
            td.match_status, td.price_pesewas, td.confidence, td.mention_index = "auto", m.amount_pesewas, conf, i
    for td in out.values():
        if td.match_status == "no_price" and td.suggested:
            td.match_status = "needs_seller"
    return decisions, out
