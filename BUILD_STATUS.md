# Build status

Last updated: 2026-10-26 (Day 0). **No live Meta call has been made yet.** Every adapter below follows Meta's published
docs, but none has been confirmed against a real response.

## Gate G1a, one model at a time

Each row passes only on its own real response. A working `models` or `chat` call proves account access only. It does
not show that SAM video segmentation or transcription works.

| Check | Command (one paid call each) | Proves | Status |
|---|---|---|---|
| Key and billing | `smoke_meta.py models --yes` | Auth works; account not payment-restricted | Not run: no `MODEL_API_KEY` |
| Muse Spark text | `smoke_meta.py chat --yes` | Chat Completions shape and `usage` fields | Not run |
| Muse Voice Transcribe | `smoke_meta.py transcribe --yes --audio runs/_smoke/speech_sample.wav` | Turn timings (`startMs`/`endMs`) come back | Not run |
| **SAM 3.1 video** | `smoke_meta.py sam --yes --video <clip.mp4> --prompt "bottle"` | Tracks, boxes and masks on real footage. **The hardest risk.** | Not run; needs a real clip |
| Muse Spark vision | first `identify` on real keyframes | Product names from crops | Not run |

`runs/_smoke/speech_sample.wav` is a local Windows TTS clip saying three prices. It is synthetic speech for checking
the response format, not seller narration. Each smoke call writes a sanitised raw response to `runs/_smoke/raw/`, a
line to `runs/_smoke/smoke_log.jsonl`, and a row in `model_calls`.

## Blockers

1. **No `MODEL_API_KEY`** on this machine (process, user and machine env checked by name only). The account's
   payment-authorisation restriction is unresolved.
2. **No footage or labels yet.** G1b, G2 and G3 cannot start without them.
3. **Open question for the organisers:** SPEC says "The repository starts on 27 October", but the first commit
   (`f89643b`) is dated 26 October, kickoff day, when "Hacking starts now" was announced. Ask in `#questions` whether
   kickoff-day commits are allowed. History has not been and will not be rewritten.

## Findings from Meta's docs (read 2026-10-26, dev.meta.ai/docs)

- **Transcription has no word-level timestamps.** The docs say timings are turn-level only (`turns[].startMs/endMs`),
  and `turns` is empty in the default `PUSH_TO_TALK` mode. The adapter originally used that mode (copied from the API
  example); that would have sent every price to the seller untimed. It now requests `ENDPOINTING` and maps turns to
  `segment` granularity, so price windows are whole turns (SPEC section 7).
- **SAM 3.1**:
  - Request: `POST /v1/responses` with `input_text` (ONE noun phrase per request; lists return zero matches) and
    `input_video` (URL or base64 data URL). MP4 or MOV with H.264, H.265 or MPEG-4; up to 15,000 frames and 16
    objects per frame.
  - Output: text with one line per frame (`<Nf>`), then object id, box and base85 mask. Object ids are stable while
    an object stays in view.
  - Parsing uses Meta's official `meta-sam-parser==0.1.0` (SAM License; source read before install: pure parsing,
    no I/O or network). Its boxes are half-open pixel bounds (x2+1).
- **Muse Spark accepts MP4 video including its audio** (`input_video`, Responses API). This is a possible fallback for
  both transcription and identification. Not used yet.
- **Prices** (estimates only; re-check before batches):
  - SAM: $0.20 per 1,000 frames, minimum 50 per request.
  - Transcribe: $0.18 per audio hour, in whole seconds.
  - Muse Spark 1.3: $1.25 input / $4.25 output per 1M tokens.
  - Muse Image: $0.01 per image.
  - The docs describe no spend cap.
- Model API base is `https://api.meta.ai/v1`, with bearer `MODEL_API_KEY`. 402 is `billing_error`. Do not retry
  400/401/402/403/404/413.

## Design decisions made today

- **One normalised video feeds SAM and our frames.** Ingest writes `normalised.mp4` (2 fps, long side at most
  1,280 px, H.264, faststart, no audio) and extracts frames from that same file. SAM frame N is therefore our frame N
  at t = N/2, in the same pixel space. A test checks the frame counts match.
- **Estimated cost per 60 s video:**
  - SAM: 120 frames, about $0.024 per prompt.
  - Transcription: about $0.003.
  - Muse Spark (prices plus identification): token-dependent, measured on the first live run.
  - `track` defaults to a single prompt (`"product package"`); each extra prompt is billed again.
- `model_calls.cost_usd` holds the estimate computed from documented usage and prices. It is never a model's claim
  and stays NULL when units are missing.
- Audio is extracted at 24 kHz mono 16-bit, the transcriber's documented native rate.

## Milestones

| Area | State | Evidence |
|---|---|---|
| A. Foundation (FastAPI, settings, SQLite v1, JSON logs, run IDs, safe paths, upload limits, health) | Done | tests/test_basics.py, tests/test_web.py |
| B. Domain (approval, hidden, shops, snapshots, proposals, wa.me) | Done | tests/test_catalogue.py, tests/test_orders.py |
| C. Seller review UI (Jinja + HTMX, mobile) | Done, checked in a browser at 375 px on the fixture | |
| D. Shop page and Stage A click-to-chat ordering | Done, checked in a browser on the fixture | |
| E. Pipeline: ingest, keyframes, cutouts from SAM masks, association, regex prices, stage CLI | Done offline | tests/test_cli.py, tests/test_sam_transcribe.py |
| E. Adapters: SAM, transcription, Muse Spark | Built from docs; **unconfirmed live** | mocked-transport tests only |
| F. Evaluation script and label schema | Done; no real labels | tests/test_eval.py |
| G. Safety tests | Done | 106 tests pass |
| Stage B agent, Muse Image, Glimmer, public intake | Not started (P2/P3) | |

## Verified local facts

- Python 3.12.10 venv at `~/.venvs/stall-to-store`. It sits outside the project because this folder's path is longer
  than Windows' 260-character limit inside pip. ffmpeg/ffprobe 9.0 are on `PATH`.

## Known gaps

- Association thresholds are placeholders. Tune them on tuning videos only.
- Tracks from different SAM prompts are not merged. Watch the duplicate-rate metric if you use more than one prompt.
- Videos are sent inline (20 MB cap on the normalised file). The Files API upload path is not built.
- Contact-sheet ranking (Step 2) is not built.
- Review links are bearer URLs, with no login and no CSRF token. Acceptable for the demo only.
