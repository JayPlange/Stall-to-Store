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
6. **Associate** prices with products: a deterministic time-window scorer proposes candidates; auto-matches
   only when one candidate clearly wins; everything else goes to the seller.
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

They also need `STS_ALLOW_LIVE_MODEL_CALLS=true` and `MODEL_API_KEY`. Every call is logged to `model_calls`,
and its sanitised raw response is saved under `runs/{run_id}/raw/`. Calls are never retried automatically.

## Project status

**Tested locally (pytest, no network, no API key):** upload validation, ffmpeg ingest, price association,
seller review and publishing, shop page, click-to-chat order proposals and confirmation safeguards, evaluation
maths, and the adapters' request building and response parsing against mocked responses.

**Not yet verified end to end:** Stage A click-to-chat ordering is tested locally only. The `wa.me` link has NOT
yet been opened on a real phone, and an order message has NOT yet been confirmed as received by a seller. This is
an open G3 blocker (see [BUILD_STATUS.md](BUILD_STATUS.md)).

**Verified with live Meta calls:**

| Call | Result |
|---|---|
| List models | HTTP 200; `sam-3.1`, `muse-voice-transcribe-1.0`, `muse-spark-1.3` available to our key |
| Muse Spark (`muse-spark-1.3`) text chat | HTTP 200, billed; response and `usage` shape match Meta's docs |

**Built from Meta's docs but NOT yet tested live:**

| Stage | Model | How it is called |
|---|---|---|
| Track | SAM 3.1 (`sam-3.1`) | Responses API, one noun phrase per request, normalised 2 fps MP4; parsed with Meta's `meta-sam-parser` |
| Transcribe | Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | `ENDPOINTING` mode for turn-level timings (no word timings exist) |
| Prices, identify | Muse Spark (`muse-spark-1.3`) | Chat Completions with JSON schema; images as data URLs |

**Planned, not started:** candidate ranking with Muse Spark, Muse Image clean-up, and a WhatsApp agent
(Muse Glimmer or Muse Spark).

The development fixture is hand-written test data and is never model output. No result in this repository
is presented as model output unless it came from a logged live call.

## Evaluation

`python eval/run_eval.py` prints the SPEC section 12 table for labelled real runs (label schema:
[eval/labels/SCHEMA.md](eval/labels/SCHEMA.md)). It shows each metric's numerator and denominator,
reports precision together with coverage, and skips fixture runs.

## Privacy

Film products and hands, not faces. Footage, run artefacts, the database, `.env` and seller or customer
numbers are gitignored and never committed.
