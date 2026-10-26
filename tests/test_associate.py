from app.fixtures import FIXTURE_MENTIONS, FIXTURE_TRACKS, FPS, FRAME_H, FRAME_W, fixture_tracks
from app.models.types import FrameHit, PriceMention, RankedCandidate, Track
from app.pipeline.associate import associate, candidates_for, mention_window

NAMES = {tid: name for tid, _b, name, *_ in FIXTURE_TRACKS}


def _track(tid, t0, t1, box, fps=2.0):
    return Track(tid, tid, [FrameHit(i, i / fps, box) for i in range(int(t0 * fps), int(t1 * fps) + 1)])


def test_windows():
    m = PriceMention("x", 100, 3.0, 4.0)
    assert mention_window(m, "word") == (1.5, 5.0)
    assert mention_window(m, "segment") == (3.0, 4.0)
    assert mention_window(m, "none") is None


def test_fixture_decisions():
    _, per = associate(FIXTURE_MENTIONS, fixture_tracks(), "word", FPS, FRAME_W, FRAME_H, NAMES)
    assert per["t1"].match_status == "auto" and per["t1"].price_pesewas == 1200
    assert per["t2"].match_status == "auto" and per["t2"].price_pesewas == 500
    # "these ones, 8 cedis" across two identical soaps: one price, both tracks.
    assert per["t3"].price_pesewas == per["t4"].price_pesewas == 800
    # Lotion and toothpaste equally visible: too close, goes to the seller.
    assert per["t5"].match_status == per["t6"].match_status == "needs_seller"
    assert per["t5"].price_pesewas is None
    assert per["t7"].match_status == "no_price"


def test_no_timing_sends_everything_to_seller():
    tracks = fixture_tracks()
    mentions = [PriceMention("12 cedis", 1200, None, None)]
    decisions, per = associate(mentions, tracks, "none", FPS, FRAME_W, FRAME_H, NAMES)
    assert len(decisions[0].candidates) == len(tracks)
    assert all(td.match_status == "needs_seller" for td in per.values())


def test_latest_clear_price_wins_and_flags_conflict():
    t = _track("a", 0, 10, (400, 200, 900, 600))
    ms = [PriceMention("ten", 1000, 2.0, 2.5), PriceMention("no, twelve", 1200, 6.0, 6.5)]
    _, per = associate(ms, [t], "word", 2.0, 1280, 720, {"a": "Milk"})
    assert per["a"].price_pesewas == 1200 and "conflict" in per["a"].note


def test_model_disagreement_blocks_auto_match():
    a = _track("a", 0, 5, (440, 160, 840, 600))
    b = _track("b", 3, 4, (0, 0, 100, 100))
    m = [PriceMention("12", 1200, 2.0, 3.0)]
    _, per = associate(m, [a, b], "word", 2.0, 1280, 720, {"a": "Milk", "b": "Soap"})
    assert per["a"].match_status == "auto"
    ranked = {0: [RankedCandidate("b", 1, 0.9), RankedCandidate("a", 2, 0.1)]}
    _, per = associate(m, [a, b], "word", 2.0, 1280, 720, {"a": "Milk", "b": "Soap"}, ranked)
    assert per["a"].match_status == "needs_seller"


def test_referent_name_bonus():
    a = _track("a", 0, 5, (300, 200, 600, 500))
    b = _track("b", 0, 5, (680, 200, 980, 500))
    m = PriceMention("the sardines are 15", 1500, 2.0, 3.0, referent="sardines")
    cands = candidates_for(m, [a, b], "word", 2.0, 1280, 720, {"a": "Milk", "b": "Sardines"})
    assert cands[0].track_id == "b" and cands[0].name_bonus > 0


def test_track_not_visible_is_not_a_candidate():
    a = _track("a", 10, 12, (0, 0, 100, 100))
    assert candidates_for(PriceMention("x", 100, 1.0, 2.0), [a], "word", 2.0, 1280, 720, {}) == []
