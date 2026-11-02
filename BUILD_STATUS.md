# Build status

Last updated: 2026-11-02. One real clip has gone through the full path. All figures below come from that one clip,
which is a tuning clip, so treat them as a first benchmark rather than measured accuracy.

## Gates

| Gate | Status |
|---|---|
| G1a: real outputs usable from code | **PASS** for SAM 3.1, Muse Voice Transcribe and Muse Spark (text and vision) |
| G1b: detection recall ≥80%, auto-match precision ≥90% | **PASS on one tuning clip:** 7/7 detection with two SAM prompts; 3/3 precision before Muse Spark |
| G2: eval prints the table | **Partial:** table runs on one real video; SPEC's gate asks for two |
| G3: real video to approved shop, real order reaches the seller | **PASS with a limit:** order S2S-KPU3XE went through WhatsApp Web, the seller received it, and it was confirmed. Not yet tried on a phone browser |
| G4: full flow, rough demo exists | Flow done across separate steps; no demo recorded yet |

## Live calls (10, about $0.11 estimated)

Estimates use Meta's published prices. The dashboard is the source of truth for billing.

| Call | Result | Est. cost |
|---|---|---|
| List models | 200, 1.8 s | none reported |
| Muse Spark chat | 200, 3.2 s; 144 of 154 output tokens were reasoning | $0.0007 |
| Transcribe, synthetic speech | 200, 6.8 s; one turn; "cedis" heard as "sadies" | $0.0005 |
| SAM "product package" | 200, 35.6 s; 8 tracks, 135 frames | $0.027 |
| Transcribe, real clip | 200, 16.7 s; "Ghana cedis" correct; one 65 s turn | $0.0034 |
| Muse Spark price extraction | 200, 15.3 s; 7 of 7 prices | $0.0124 |
| Muse Spark naming, 8 crops | 200, 19.5 s; correct names; nothing invented for floor tile | $0.0170 |
| SAM "bottle" | 200, 13.8 s; found Vim and Bel-Aqua, plus 2 tracks dropped by merging | $0.027 |
| Muse Spark naming, 2 crops | 200, 13.5 s; Vim and Bel-Aqua named | $0.0080 |
| Muse Spark price linking | 200, 11.4 s; 7 of 7 exact | $0.0135 |

`model_calls.run_id` for this clip's smoke calls was set afterwards, so `run_eval.py` reports the right cost.

## Benchmark: 1 video, 7 products

| Metric | Result | Note |
|---|---|---|
| Detection recall | 7/7 | Needed two SAM prompts; "product package" alone found 5 of 7 product types |
| Duplicate rate | 1/7 | Three real Comfort sachets are three tracks for one product |
| Identification | 6/7 | Scored on the first Comfort crop, which was too blurry to name |
| Auto-match precision / coverage, before Spark | 3/3, 3/9 tracks | As stored in the draft |
| Muse Spark price links | 7/7 | With Spark applied in a dry run: 9 of 9 tracks auto-match with correct prices |
| Seller corrections | 9 | Includes names, prices and sizes |
| Time to shop | about 40 minutes | Includes development during the run, so not representative |
| Cost per video | about $0.11 | Includes an extra SAM search and a repeat naming call |
| Order safety failures | 0 of 1 | |

## Open items

1. **Organisers:** SPEC says the repository starts on 27 October, but the first commit (`34dfff0`) is dated 26 October,
   kickoff day. Ask whether that is allowed. Commit timestamps have been preserved and will not be altered.
2. **Licence:** not chosen. The owner decides.
3. **Submission requirements:** the organisers' screenshots were not available in the build session; check against them.
4. **Publication:** no remote exists. Push `main` only, never `--all` or `--mirror`.
5. **Demo:** the 3-minute video is not recorded.

## Findings

- **Timing:** Muse Voice Transcribe has no word timestamps, and real narration came back as one turn. Timing cannot
  link prices to products on this clip, so linking relies on names and Muse Spark.
- **SAM prompts:** one noun phrase per request. "product package" missed the bottle and scourer; "bottle" found them.
  It also returned a false floor-tile track, which identification correctly left unnamed.
- **Parsing:** `meta-sam-parser` decodes every mask in pure Python, which took 18 minutes. Masks are now decoded only
  for the frames used, and parsing takes 0.17 s. The official parser still reads frames, ids, boxes and confidence.
- **Reasoning tokens:** they dominate Muse Spark's cost. Measure across more clips before relying on the estimate.
- **Word matching:** on this clip it needed filler and category words removed; "the" once caused a wrong match.
- **Video:** the clip is 640×360, the owner's recording limit, so product photos are soft.

## Known gaps

- `link` ran on an existing draft; a full run from `ingest` through `link` and `associate` has not been done in one go.
- `associate` treats a whole-clip turn as real timing. Keyword hints to transcription are untested.
- Identical items are separate listings. Muse Image clean-up is not built.
- Word lists, thresholds and the 0.8 link cut-off are tuned on one clip.
- Review links are bearer URLs with no login or CSRF token. Demo only.

## Local setup

Python 3.12 venv at `~/.venvs/stall-to-store`; ffmpeg/ffprobe on `PATH`; 127 tests pass.
