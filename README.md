# Stall to Store

A market seller films their stall for under a minute while saying prices out loud, and gets back a live
shop page whose products, photos and prices come from that video, with orders arriving on WhatsApp.

Meta Global AI Developer Hackathon, Indie track, theme: Build for Your World.

## Built during the event

**Everything in this repository was built during the hackathon, starting after kickoff on 26 October 2026.** Nothing is
copied from earlier projects, and the commit history shows the work as it happened.

- The video pipeline: ingest, multi-prompt SAM 3.1 tracking, transcription, price extraction, product naming, and
  Muse Spark price matching
- Seller review, the shop page, and WhatsApp click-to-chat ordering with price and order safeguards
- The tracking video, the evaluation script, and the test suite

The only outside code is the open-source libraries in `requirements.txt`, including Meta's `meta-sam-parser`.
See [SPEC.md](SPEC.md) for the design and [BUILD_STATUS.md](BUILD_STATUS.md) for current results.

## Run it

Requires Python 3.12 and ffmpeg on `PATH`. On Windows, keep the venv at a short path (e.g. `~/.venvs/stall-to-store`) if pip hits the 260-character path limit.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m uvicorn app.main:create_app --factory --no-access-log --port 8000
```

Create your local settings file by copying `.env.example` to `.env` (Windows: `copy .env.example .env`;
macOS/Linux: `cp .env.example .env`). `.env` is gitignored. `MODEL_API_KEY` is needed only for live model calls;
the app, the fixture and the tests run without it.

Open http://127.0.0.1:8000. To try the review screen and shop without model access, set
`STS_DEV_FIXTURES=true`. The "Load development fixture" button then creates a hand-written fixture run,
shown with a red **DEVELOPMENT FIXTURE** banner on every page.

Tests (no network, no API key, no paid calls):

```bash
.venv/Scripts/python -m pytest
```

## How it works

1. **Ingest** (ffmpeg, local): `audio.wav` (24 kHz mono), `normalised.mp4` (2 fps, the file SAM receives),
   frames extracted from that same file, `metadata.json`.
2. **Transcribe** with Muse Voice Transcribe, 3. **Track** products with SAM 3.1, 4. **Keyframes** (area × sharpness),
5a. **Price mentions** from the transcript with Muse Spark (regex fallback), 5b. **Identify** products from crops with Muse Spark.
6. **Link and associate** prices with products: Muse Spark compares a numbered sheet of product photos with the
   narration, and deterministic scoring checks the result. Spoken amounts are never changed, and anything uncertain
   goes to the seller.
7. **Seller review** (`/r/{private-token}`): correct names and prices, pick a suggested price, approve or hide.
   **Nothing unapproved is ever published.**
8. **Shop page** (`/s/{slug}`): the customer picks quantities; the backend computes the total from approved
   prices (integer pesewas) and stores a proposal with a price snapshot and reference code.
9. **Ordering (Stage A, click-to-chat):** the customer gets a `wa.me` link with the order prefilled.
   **This is click-to-chat, not an automated agent: nothing is sent until the customer presses Send in
   WhatsApp.** The seller confirms the reference on the review page; confirmation re-checks expiry and that
   prices have not changed since the proposal.

Uploading a video runs every stage in the background, with a progress screen. The draft is usually ready in about
3 minutes. Stages can also be run one at a time:

```bash
python -m app.pipeline.cli transcribe <run_id> --live
```

The `overlay` stage draws SAM's real tracking over the footage for the review page. Live stages need `STS_ALLOW_LIVE_MODEL_CALLS=true` and `MODEL_API_KEY`. Every call is logged to `model_calls`,
and its sanitised raw response is saved under `runs/{run_id}/raw/`. Calls are never retried automatically.

## Project status

A single upload now runs the whole pipeline and produces a draft for the seller to approve. It has been run on one
real video using Meta's live API, so the results below are a small first benchmark, not a measured accuracy.

**Verified with live Meta calls on a real seller-style video:**

| Model | Job | Result |
|---|---|---|
| SAM 3.1 (`sam-3.1`) | Find and track products | Two searches ("product package", "bottle") found all 7 product types across 135 frames |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Hear the narration | All 7 products and prices heard; "Ghana cedis" correct; one long turn, so no useful timing |
| Muse Spark (`muse-spark-1.3`) | Read prices, name products, link prices to products | 7 of 7 prices extracted and linked to the right products, on all three runs where it matched |

**End to end on real data:**
- One upload produced a full draft in about 3 minutes, with no manual steps. The photos and tracking video finish after
  that, at about 11 to 14 minutes in total.
- A published shop took an order through WhatsApp Web. The seller's WhatsApp received it, and it was confirmed on the
  review page. The customer side has not yet been tried in a phone browser.

**Tested locally (pytest, no network, no API key):** upload validation, ffmpeg ingest, track merging, price
association, seller review and publishing, shop page, order safeguards, evaluation maths, and the whole pipeline
against mocked responses, including failure and double-click handling.

**Known limits:**
- All results come from one clip, which was also used for tuning. Each run varies a little: SAM's object count and the
  transcript both changed between runs.
- Word lists, confidence cut-offs and match thresholds are tuned on this clip.
- Filters can remove a real item that is too small or blurry to read. On one run this removed one of three Comfort
  sachets. The removal is recorded in the run's files, but the seller is not told.
- The clip is 640×360, so product photos are soft.
- The hosted version is not live. Review links are private URLs with no login or CSRF protection: demo only.

**Planned, not started:** Muse Image background clean-up, grouping identical items into one listing, and a WhatsApp
agent (Muse Glimmer or Muse Spark).

The development fixture is hand-written test data and is never model output. No result in this repository
is presented as model output unless it came from a logged live call.

## Evaluation

`python eval/run_eval.py` prints the SPEC section 12 table for labelled real runs (label schema:
[eval/labels/SCHEMA.md](eval/labels/SCHEMA.md)). It shows each metric's numerator and denominator,
reports precision together with coverage, and skips fixture runs. Add `--labels eval/labels_automatic` for the
automatic run below.

**First benchmark: 1 video, 7 products (tuning clip, no held-out videos):**

| Metric | Automatic run (no seller edits) | Target |
|---|---|---|
| Detection recall | 7/7 | ≥80% |
| Duplicate rate | 1/7 (two Comfort sachets count as one product with two tracks) | ≤10% |
| Identification accuracy | 5/7 ("AILYON" missing its S; one Comfort card had no product name) | ≥80% |
| Auto-match precision | 8/8 | ≥90% |
| Auto-match coverage | 8/8 | report |
| Muse Spark price links | 7/7 | report |
| Cost per video | about $0.12 estimated | report |

Duplicate rate and identification miss their targets on this run. The separate first run, which went on to publishing,
recorded 1/1 end-to-end completion and 0 order safety failures across 1 confirmed order.

## Privacy

Film products and hands, not faces. Footage, run artefacts, the database, `.env` and seller or customer
numbers are gitignored and never committed.
