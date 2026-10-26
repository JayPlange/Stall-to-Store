"""Stage 6: price association (SPEC section 7). Pure, deterministic, model-free.

The scorer proposes candidates; it is never the source of truth. Auto-matches are still shown to
the seller, and nothing is published without seller approval.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.models.types import Candidate, Granularity, PriceMention, RankedCandidate, Track

WORD_PRE_S = 1.5
WORD_POST_S = 1.0
TOP_K = 4
# Placeholder thresholds: tune on tuning videos only (SPEC section 7), never on held-out videos.
AUTO_MARGIN = 0.15
AUTO_MIN_SCORE = 0.35
NAME_MATCH_RATIO = 0.6
NAME_BONUS = 0.3
PLURAL_REFERENTS = {"these", "those", "them", "ones", "all", "each", "both"}


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


def name_similarity(referent: str | None, name: str | None) -> float:
    ref, nm = _tokens(referent), _tokens(name)
    if not ref or not nm:
        return 0.0
    best = 0.0
    for r in ref:
        if len(r) < 3 or r in PLURAL_REFERENTS:
            continue
        for n in nm:
            best = max(best, SequenceMatcher(None, r, n).ratio())
    return best


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
