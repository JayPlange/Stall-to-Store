from pathlib import Path

import pytest

from app.artefacts import UnsafePathError, new_run_id, resolve_artefact, run_dir, safe_join, validate_run_id
from app.models.types import PriceMention, Transcript, Unit
from app.money import PriceError, format_cedis, parse_cedis
from app.pipeline.prices_regex import find_amounts, regex_mentions


@pytest.mark.parametrize("text,expected", [
    ("12", 1200), ("12.5", 1250), ("12.50", 1250), ("GH₵ 1,200.00", 120000), ("5 cedis", 500), ("0.50", 50)])
def test_parse_cedis(text, expected):
    assert parse_cedis(text) == expected


@pytest.mark.parametrize("text", ["", "0", "-3", "12.345", "abc", "1e3", None, "99999999", "12,50", "12,5", "GH₵ 12,50", "1.200,00"])
def test_parse_cedis_rejects(text):
    with pytest.raises(PriceError):
        parse_cedis(text)


def test_format_cedis():
    assert format_cedis(123456) == "GH₵1,234.56"
    assert format_cedis(5) == "GH₵0.05"


def test_run_ids_validate():
    assert validate_run_id(new_run_id())
    for bad in ["../etc", "r_1", "", "r_20261026T120000_zzzzzzzz", "r_20261026T120000_0123abcd/.."]:
        with pytest.raises(UnsafePathError):
            validate_run_id(bad)


def test_safe_join_blocks_traversal(tmp_path: Path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "..", "secret")
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "a", "../../b")
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, str(tmp_path.anchor) + "Windows")
    assert safe_join(tmp_path, "a", "b.txt") == (tmp_path / "a" / "b.txt").resolve()


def test_resolve_artefact_rejects_outside_and_missing(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x")
    root = tmp_path / "runs"
    root.mkdir()
    assert resolve_artefact(root, "../outside.txt") is None
    assert resolve_artefact(root, "nope.jpg") is None
    rd = run_dir(root, new_run_id(), create=True)
    (rd / "keyframes" / "t1.jpg").write_bytes(b"x")
    assert resolve_artefact(root, f"{rd.name}/keyframes/t1.jpg") is not None


def test_transcript_requires_timings_for_timed_granularity():
    with pytest.raises(ValueError):
        Transcript("x", "word", [Unit("x", None, None)])
    assert Transcript("x", "none", [Unit("x")]).granularity == "none"
    assert Transcript("x", "segment", [Unit("x", 0.0, 1.0)]).units[0].t_end == 1.0


def test_price_mention_rejects_non_integer_amount():
    with pytest.raises(ValueError):
        PriceMention("x", 12.5, None, None)  # type: ignore[arg-type]


def test_regex_prices():
    assert [p for _, p in find_amounts("the milk is 12 cedis and soap GH₵5.50")] == [1200, 550]
    t = Transcript("x", "segment", [Unit("tomato 5 cedis", 1.0, 2.0), Unit("hello", 2.0, 3.0)])
    mentions = regex_mentions(t)
    assert len(mentions) == 1 and mentions[0].amount_pesewas == 500 and mentions[0].t_start == 1.0
    assert regex_mentions(Transcript("x", "none", [Unit("12 cedis")]))[0].t_start is None
