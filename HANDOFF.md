# Handoff: Stall to Store

State as of 2026-11-02. Read [SPEC.md](SPEC.md) (design) and [BUILD_STATUS.md](BUILD_STATUS.md) (results) first.
Owner: Edwin (Justice Edwin Mensah Plange).

## What it does

A market seller films their stall and says the prices aloud. Meta models propose products and prices, the seller
corrects and approves them on a private page, and a shop goes live. A customer picks items, the backend computes the
total from approved integer-pesewa prices, and a WhatsApp click-to-chat message goes to the seller, who confirms it by
reference code. The customer must press Send.

## Code map

- **Web:** `app/main.py` (`create_app`), `app/web/routes.py`, templates in `app/web/templates/`. Upload `/`, seller
  review `/r/{token}`, shop `/s/{slug}`.
- **Data:** `app/schema.sql`, `app/catalogue.py`, `app/orders/service.py`, `app/orders/links.py`, `app/money.py`.
- **Pipeline:** `python -m app.pipeline.cli <stage> <run_id> [--live]`. Stages: `ingest`, `transcribe`, `prices`,
  `track`, `keyframes`, `identify`, `link`, `associate`, `cutouts`, `overlay`. Paid stages need `--live`.
- **Logic:** `app/pipeline/associate.py` (matching, merging prompts, applying Muse Spark links),
  `keyframes.py` (photos, contact sheet), `overlay.py` (tracking video).
- **Meta adapters:** `app/models/` (`base.py` gates, logs and saves every call; never retries).
- **Checks:** `scripts/smoke_meta.py`, `eval/run_eval.py`, label in `eval/labels/stall-test-01.json`.

## Setup

- Python 3.12, ffmpeg on `PATH`. Venv: `C:/Users/Edwin/.venvs/stall-to-store`.
- `.env` holds `MODEL_API_KEY` and is gitignored. Enable paid calls per command:
  `STS_ALLOW_LIVE_MODEL_CALLS=true python ...`.
- Run: `python -m uvicorn app.main:create_app --factory --no-access-log --port 8000`.
- Tests: `python -m pytest` (127 tests, no network or key).

## Where things stand

- The real clip is `footage/stall-test-01.mp4`, and its run is in `runs/` and `var/` (all gitignored, not in any ZIP).
- Published local shop `/s/malloy-stall` shows 7 products at the spoken prices. Order S2S-KPU3XE is confirmed.
- The seller's private review link is in the session, not in this file. Get it from `runs.review_token` in the local
  database if needed. Never publish it.
- `tracking.mp4` is on the review page. Results and limits are in BUILD_STATUS.md.
- Edwin will keep using this clip. Do not ask for 1080p or new videos.
- The GitHub requirements screenshots from the kickoff meeting were never received. Ask Edwin to resend them.

## Next, in order

1. Commit the current documentation and evaluation changes when Edwin approves.
2. Edwin decides the licence and asks organisers about the code-start date.
3. Edwin creates the public GitHub repo. Push `main` only, after approval, then check it in a logged-out browser.
4. Record the 3-minute demo (SPEC section 14) with the tracking video and the real shop and order.
5. If time allows, and each paid call is approved: Muse Image clean-up, keyword hints, grouping Comfort.

**Deadlines:** build freeze 4 Nov, 6:00pm Accra. Submission 5 Nov, 2:00am Accra. Viral engagement until 8 Nov, 11:59pm
ET. Live final 9 Nov.

## Do not

- Claim results beyond the one clip, or present fixtures as model output.
- Invent model outputs or prices. Amounts come from speech; the seller approves every price.
- Commit `.env`, `.private/`, `footage/`, `runs/`, `var/` or `.claude/`, or print key values.
- Make paid calls without Edwin's approval, or retry them automatically.
- Push, add a remote, deploy, rewrite history, alter timestamps, or choose a licence without Edwin's decision.
- Push `--all` or `--mirror`, or touch the private backups: branch `backup/pre-rewrite-2026-10-26`,
  `refs/original/refs/heads/main`, and `C:\Users\Edwin\stall-to-store-backups\pre-rewrite-2026-10-26.bundle`.
