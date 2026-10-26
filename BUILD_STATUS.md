# Build status

Last updated: 2026-10-26 (Day 0).

## Gates

| Gate | Status |
|---|---|
| G1a: one real call to each model from code | **BLOCKED**: no `MODEL_API_KEY` on this machine (process, user and machine env checked by name only). Zero live calls made. |
| G1b: detection recall / auto-match precision on a real video | Not started (needs G1a and footage) |
| G2: eval runs on 2 videos | Script built and tested; no labelled real videos yet |
| G3: real video → approved shop → real order reaches seller | App flow works on fixture data; no real video processed by models |

## Blockers

1. **Meta API access.** `MODEL_API_KEY` is not set, and the account's payment-authorisation restriction is unresolved. Next steps:
   1. Set the key in `.env` with `STS_ALLOW_LIVE_MODEL_CALLS=true`.
   2. Run `python scripts/smoke_meta.py models --yes`, then `chat --yes`, then `transcribe --yes`, one at a time.
   3. Write the transcribe mapping from the saved raw response.
2. **SAM 3.1 request format.** The pages read on 2026-10-26 list `sam-3.1` and say video files go through `POST /v1/files`. They do not document the tracking request or response. `app/models/segment.py` makes no call until this is verified.
3. **Muse Voice Transcribe response schema.** The `TranscribeResponse` fields are not in the pages read, and the Schemas page returned 404. `transcribe.normalise` raises `AdapterNotVerified` and lists the observed keys. Word-timestamp support is unknown.
4. **No footage or labels yet.**

## Verified facts

- Python 3.12.10 (via `py -3.12`). Python 3.14 is the default `python` on this machine.
- ffmpeg/ffprobe 9.0 (gyan.dev full build) on `PATH`. Ingest is tested on generated video (6 s → 12–13 frames at 2 fps plus a 16 kHz WAV).
- The venv lives at `~/.venvs/stall-to-store` because the workspace path exceeds Windows' 260-character limit inside pip.
- Meta docs moved: `llama.developer.meta.com/docs` now redirects to `dev.meta.ai/docs`. From those docs, read 2026-10-26:
  - Model API base is `https://api.meta.ai/v1`, with bearer auth and env var `MODEL_API_KEY`.
  - Chat completions: `POST /v1/chat/completions`. Image parts use `image_url` (https or data URL), at most 50 per request. Structured output uses `response_format.json_schema`. The reply is in `choices[0].message.content`. `usage` has prompt, completion and total tokens.
  - Transcription: `POST /v1/asr/transcribe`, multipart (`request` JSON plus `audio` WAV), at most 10 minutes and 32 MB.
  - Errors: envelope `{"error": {message, type, param, code}}`. 402 is `billing_error`. Do not retry 400/401/402/403/404/413.
  - Model ids listed: `muse-spark-1.3`, `muse-image-1.0`, `muse-voice-transcribe-1.0`, `sam-3.1`. No Model API id is listed for Muse Glimmer (open weights).
- Spec conflict: SPEC says the repo starts on 27 October, but the first commit is dated 26 October (kickoff day). Fix SPEC or the README if that matters.

## Milestones

| Area | State | Evidence |
|---|---|---|
| A. Foundation: FastAPI, settings, SQLite schema v1, JSON logs, run IDs, safe paths, upload limits, health | Done | tests/test_basics.py, tests/test_web.py |
| B. Domain: products, mentions, candidates, approval, hidden, shops, snapshots, proposals, wa.me links | Done | tests/test_catalogue.py, tests/test_orders.py |
| C. Seller review UI (Jinja + HTMX, mobile) | Done, checked in a browser at 375 px on the fixture | Save & approve, suggested prices, hide/restore, publish, order confirm |
| D. Shop page and Stage A ordering | Done, checked in a browser on the fixture | Backend total, reference code, "not sent yet" notice |
| E. Pipeline: ingest, keyframes, cutouts, association, regex prices, stage CLI, adapters | Ingest/association/keyframes done; model adapters gated | tests/test_associate.py, tests/test_models.py, tests/test_cli.py |
| F. Evaluation script and label schema | Done; no real labels yet | tests/test_eval.py |
| G. Safety tests (SPEC section 9 list, invalid uploads, unsafe paths, repeats) | Done | 98 tests pass |
| Stage B agent, Muse Image, Glimmer, public intake | Not started (P2/P3 by design) | — |

## Known gaps

- Association thresholds (`AUTO_MARGIN=0.15`, `AUTO_MIN_SCORE=0.35`) are placeholders. Tune them on tuning videos only.
- `cost_usd` is always NULL until real billing units are known. Eval reports these calls as "unknown cost".
- Contact-sheet ranking (Step 2) is not built (`vision.rank_candidates` raises).
- Review links are bearer URLs (secret token, `Referrer-Policy: no-referrer`), with no login and no CSRF token. Fine for the demo, not for production.
- Review edits are not blocked after publishing; edited products drop to draft and disappear from the shop until re-approved.
