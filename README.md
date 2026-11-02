# Stall to Store

A market seller films their stall for under a minute while saying prices out loud, and gets back a live
shop page whose products, photos and prices come from that video, with orders arriving on WhatsApp.

Meta Global AI Developer Hackathon, Indie track. **All code in this repository was written during the
event, starting 26 October 2026.** Nothing is copied from earlier projects. See [SPEC.md](SPEC.md) for the
full design and [BUILD_STATUS.md](BUILD_STATUS.md) for what works today versus what is still blocked.

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

Model stages run one at a time, only when asked:

```bash
python -m app.pipeline.cli transcribe <run_id> --live
```

The `overlay` stage draws SAM's real tracking over the footage for the review page. Live stages also need `STS_ALLOW_LIVE_MODEL_CALLS=true` and `MODEL_API_KEY`. Every call is logged to `model_calls`,
and its sanitised raw response is saved under `runs/{run_id}/raw/`. Calls are never retried automatically.

## Project status

One real seller-style video has gone through the whole path: video, tracking, narration, prices, names, seller
review, a published shop, and an order that reached WhatsApp and was confirmed. That is one tuning clip, so the
results below are a small first benchmark, not a measured accuracy.

**Verified with live Meta calls:**

| Model | Job | Real result on the clip |
|---|---|---|
| SAM 3.1 (`sam-3.1`) | Find and track products | Two searches ("product package", "bottle") found all 7 product types across 135 frames |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Hear the narration | All 7 products and prices heard; "Ghana cedis" correct; one 65 s turn, so no useful timing |
| Muse Spark (`muse-spark-1.3`) | Read prices, name products, link prices to products | 7 of 7 prices extracted; product names read from photos; 7 of 7 prices linked to the right products |

**Tested locally (pytest, no network, no API key):** upload validation, ffmpeg ingest, price association, seller
review and publishing, shop page, order safeguards, evaluation maths, and adapter behaviour against mocked responses.

**Verified by hand on the real clip:** a published shop, and order S2S-KPU3XE sent through WhatsApp Web, received by
the seller's WhatsApp, and confirmed on the review page. The customer side has not yet been tried on a phone browser.

**Known limits:**
- Results come from one clip. SAM thresholds, the word lists and Muse Spark's confidence cut-off are tuned on it.
- Muse Spark links ran on this clip after the draft existed. The full pipeline has not yet run them from start to finish.
- The clip is 640×360, so product photos are soft.
- Review links are private URLs with no login or CSRF protection: demo only.

**Planned, not started:** Muse Image background clean-up, grouping identical items into one listing, and a WhatsApp
agent (Muse Glimmer or Muse Spark).

The development fixture is hand-written test data and is never model output. No result in this repository
is presented as model output unless it came from a logged live call.

## Evaluation

`python eval/run_eval.py` prints the SPEC section 12 table for labelled real runs (label schema:
[eval/labels/SCHEMA.md](eval/labels/SCHEMA.md)). It shows each metric's numerator and denominator,
reports precision together with coverage, and skips fixture runs.

**First benchmark: 1 video, 7 products (tuning clip, no held-out videos yet):**

| Metric | Result | Target |
|---|---|---|
| Detection recall | 7/7 | ≥80% |
| Duplicate rate | 1/7 (three separate Comfort sachets count as one product with three tracks) | ≤10% |
| Identification accuracy | 6/7 (the first Comfort crop was too blurry to name) | ≥80% |
| Auto-match precision, before Muse Spark | 3/3 | ≥90% |
| Auto-match coverage, before Muse Spark | 3/9 tracks | report |
| Muse Spark price links | 7/7 | report |
| Seller corrections | 9 | lower |
| End-to-end completion | 1/1 | report |
| Time to shop | about 40 minutes, including building and testing during the run | report |
| Cost per video | about $0.11 estimated, including one extra SAM search and a repeat naming call | report |
| Order safety failures | 0 of 1 confirmed orders | 0 |

## Privacy

Film products and hands, not faces. Footage, run artefacts, the database, `.env` and seller or customer
numbers are gitignored and never committed.
