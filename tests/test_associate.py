from PIL import Image

from app.fixtures import FIXTURE_MENTIONS, FIXTURE_TRACKS, FPS, FRAME_H, FRAME_W, fixture_tracks
from app.models.types import FrameHit, PriceLink, PriceMention, RankedCandidate, Track
from app.pipeline.keyframes import make_contact_sheet, ranked_frames
from app.pipeline.overlay import box_at
from app.pipeline.associate import (MentionDecision, TrackDecision, apply_price_links, associate, candidates_for, is_identifiable, mention_window, merge_tracks,
                                    unusable_tracks,
                                    name_similarity)

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


def test_filler_words_cannot_match():
    # Real failure: "the" in the Vim narration scored 1.0 against "The Earl Grey".
    assert name_similarity("vim scorer the vim multi-purpose scorer", "TWININGS The Earl Grey Our Classic Blend") < 0.6


def test_generic_words_cannot_match_alone():
    assert name_similarity("Bel Aqua premium mineral water", "AILYONS Water Kettle") < 0.6
    assert name_similarity("the kettle", "AILYONS Water Kettle") == 0.0


def test_misheard_and_split_brands_still_match():
    assert name_similarity("cremila milk powder", "Kremela Non-Dairy Creamer") >= 0.6
    assert name_similarity("soft care baby wipes", "Softcare Baby Wipes") == 1.0
    assert name_similarity("twinnings Earl Grey", "TWININGS The Earl Grey") >= 0.9


def test_only_empty_low_confidence_tracks_are_filtered():
    assert not is_identifiable(None, None, 0.1)
    assert not is_identifiable(None, None, 0.3)  # real floor tile, second run
    assert is_identifiable("Comfort", None, 0.4)
    assert is_identifiable(None, "Baby Wipes", 0.1)
    assert is_identifiable(None, None, 0.8)      # unreadable real product: seller still sees it
    assert is_identifiable(None, None, None)     # no evidence either way: never drop silently


def test_photo_frame_avoids_obstruction_and_edges(tmp_path):
    img = tmp_path / "f.jpg"
    Image.effect_noise((200, 200), 80).convert("RGB").save(img)
    box = (50, 50, 150, 150)
    track = Track("a", "a", [FrameHit(0, 0.0, box), FrameHit(1, 0.5, box), FrameHit(2, 1.0, (0, 50, 100, 150))])
    ranked = ranked_frames(track, lambda i: img, {0: [(50, 50, 120, 150)]})
    assert [h.frame_idx for h, _ in ranked][0] == 1           # clear view wins over a covered one
    assert ranked[-1][0].frame_idx in (0, 2)                  # covered or edge-touching ranks last


def test_overlay_only_draws_where_sam_detected():
    hits = [FrameHit(0, 0.0, (0, 0, 10, 10)), FrameHit(1, 0.5, (10, 0, 20, 10)), FrameHit(20, 10.0, (50, 0, 60, 10))]
    track = Track("a", "a", hits)
    box, nearest = box_at(track, 0.25)
    assert box == (5.0, 0.0, 15.0, 10.0) and nearest.frame_idx in (0, 1)  # glides across a short gap
    assert box_at(track, 5.0) is None                                      # object gone: nothing invented
    assert box_at(track, 10.1)[0] == (50, 0, 60, 10)                       # brief hold around a real hit
    assert box_at(track, 12.0) is None


def test_merge_drops_cross_prompt_duplicates_and_tiny_tracks():
    box = (100, 100, 300, 300)
    first = Track("p0o1", "x", [FrameHit(i, i / 2, box) for i in range(10)])
    side = Track("p1o2", "x", [FrameHit(i, i / 2, (150, 150, 250, 250)) for i in range(10)])
    other = Track("p1o0", "x", [FrameHit(i, i / 2, (400, 100, 500, 300)) for i in range(10)])
    tiny = Track("p1o3", "x", [FrameHit(i, i / 2, (10, 10, 20, 20)) for i in range(3)])
    kept, dropped = merge_tracks([first, side, other, tiny])
    assert [t.track_id for t in kept] == ["p0o1", "p1o0"]
    assert dropped == {"p1o2": "same object as p0o1", "p1o3": "only 3 frames"}


def test_merge_keeps_touching_items_from_the_same_prompt():
    box = (100, 100, 300, 300)
    a = Track("p0o3", "x", [FrameHit(i, i / 2, box) for i in range(10)])
    b = Track("p0o4", "x", [FrameHit(i, i / 2, box) for i in range(10)])
    assert merge_tracks([a, b]) == ([a, b], {})


def _links_setup():
    mentions = [PriceMention("300", 30000, 0.0, 9.0), PriceMention("200", 20000, 0.0, 9.0)]
    tracks = {"a": TrackDecision("a", "no_price"), "b": TrackDecision("b", "no_price")}
    decisions = [MentionDecision(0, []), MentionDecision(1, [])]
    return decisions, tracks, mentions


def test_confident_link_uses_the_spoken_amount():
    decisions, tracks, mentions = _links_setup()
    apply_price_links(decisions, tracks, mentions, [PriceLink(0, ["a"], 0.9, "sounds alike")])
    assert tracks["a"].match_status == "auto" and tracks["a"].price_pesewas == 30000
    assert "Muse Spark" in tracks["a"].note and decisions[0].auto_track_ids == ["a"]


def test_uncertain_link_goes_to_the_seller():
    decisions, tracks, mentions = _links_setup()
    apply_price_links(decisions, tracks, mentions, [PriceLink(0, ["a"], 0.5, "maybe")])
    assert tracks["a"].match_status == "no_price" and tracks["a"].price_pesewas is None
    assert tracks["a"].suggested == [0]


def test_conflicting_links_go_to_the_seller():
    decisions, tracks, mentions = _links_setup()
    links = [PriceLink(0, ["a"], 0.9), PriceLink(1, ["a"], 0.9)]
    apply_price_links(decisions, tracks, mentions, links)
    assert tracks["a"].match_status == "needs_seller" and tracks["a"].price_pesewas is None


def test_link_disagreeing_with_a_clear_match_goes_to_the_seller():
    decisions, tracks, mentions = _links_setup()
    tracks["a"] = TrackDecision("a", "auto", 20000, 0.9, 1)
    apply_price_links(decisions, tracks, mentions, [PriceLink(0, ["a"], 0.95)])
    assert tracks["a"].match_status == "needs_seller"


def test_one_price_can_cover_several_items():
    decisions, tracks, mentions = _links_setup()
    apply_price_links(decisions, tracks, mentions, [PriceLink(0, ["a", "b"], 0.9)])
    assert tracks["a"].price_pesewas == tracks["b"].price_pesewas == 30000


def test_contact_sheet_numbers_every_product(tmp_path):
    img = tmp_path / "p.jpg"
    Image.new("RGB", (120, 80), "red").save(img)
    out = make_contact_sheet([(1, img), (2, img), (3, img), (4, img), (5, img)], tmp_path / "sheet.jpg")
    with Image.open(out) as s:
        assert s.size == (1200, 600)


def test_unusable_tracks_drops_background_and_parts_only():
    box = (100, 100, 300, 300)
    whole = Track("a", "x", [FrameHit(i, i / 2, box) for i in range(20)])
    flap = Track("b", "x", [FrameHit(i, i / 2, (120, 120, 200, 180)) for i in range(5)])
    floor = Track("c", "x", [FrameHit(i, i / 2, (0, 0, 50, 50)) for i in range(10)])
    sachet = Track("d", "x", [FrameHit(i, i / 2, (120, 120, 200, 180)) for i in range(5)])
    guesses = {"a": {"brand": "AILYON", "name": "Water Kettle", "confidence": 0.8},
               "b": {"brand": "AILYONS", "name": None, "confidence": 0.35},
               "c": {"brand": None, "name": None, "confidence": 0.3},
               "d": {"brand": "Comfort", "name": None, "confidence": 0.5}}
    out = unusable_tracks([whole, flap, floor, sachet], guesses)
    assert out == {"b": "part of a", "c": "nothing readable"}  # a different brand held in front stays


def test_named_item_inside_another_track_stays():
    box = (100, 100, 300, 300)
    big = Track("a", "x", [FrameHit(i, i / 2, box) for i in range(20)])
    small = Track("b", "x", [FrameHit(i, i / 2, (120, 120, 200, 180)) for i in range(5)])
    guesses = {"a": {"brand": "Comfort", "name": "Blue Ultra", "confidence": 0.9},
               "b": {"brand": "Comfort", "name": "Blue Ultra", "confidence": 0.9}}
    assert unusable_tracks([big, small], guesses) == {}
