# Handoff: Stall to Store

State as of 2026-11-02. Read [SPEC.md](SPEC.md) (design) and [BUILD_STATUS.md](BUILD_STATUS.md) (results) first.
Owner: Edwin (Justice Edwin Mensah Plange). Indie track, theme Build for Your World.

## What it does

A market seller films their stall and says the prices aloud. One upload runs Meta's models and produces a draft. The
seller corrects and approves it, and a shop goes live. A customer picks items, the backend computes the total from
approved integer-pesewa prices, and a WhatsApp click-to-chat message goes to the seller, who confirms it by reference
code. The customer must press Send.

## Code map

- **Web:** `app/main.py` (`create_app`), `app/web/routes.py`, templates in `app/web/templates/`. Upload `/`, seller
  review `/r/{token}`, shop `/s/{slug}`.
- **Pipeline:** `app/pipeline/orchestrate.py` runs every stage in the background; `stages.py` holds the stage code,
  shared with the command line `python -m app.pipeline.cli <stage> <run_id> [--live]`.
- **Logic:** `associate.py` (matching, merging prompts, filtering junk, Muse Spark links), `keyframes.py` (photos,
  contact sheet), `overlay.py` (tracking video).
- **Data:** `app/schema.sql`, `app/catalogue.py`, `app/orders/service.py`, `app/orders/links.py`, `app/money.py`.
- **Meta adapters:** `app/models/`. `base.py` gates, logs and saves every call, and never retries.
- **Checks:** `scripts/smoke_meta.py`, `eval/run_eval.py`, labels in `eval/labels/` and `eval/labels_automatic/`.

## Setup

- Python 3.12, ffmpeg on `PATH`. Venv: `C:/Users/Edwin/.venvs/stall-to-store`.
- `.env` holds `MODEL_API_KEY` and is gitignored. Paid calls stay off unless the server starts with
  `STS_ALLOW_LIVE_MODEL_CALLS=true`. With that on, each upload spends about $0.12.
- Any public server must also set `STS_UPLOAD_CODE`.
- Run: `python -m uvicorn app.main:create_app --factory --no-access-log --port 8000`.
- Tests: `python -m pytest` (138 tests, no network or key).

## Where things stand

- The real clip is `footage/stall-test-01.mp4`. Runs are in `runs/` and `var/`, which are gitignored.
- Published local shop `/s/malloy-stall` shows 7 products at the spoken prices, and order S2S-KPU3XE is confirmed. That
  is the demo shop. Two later automatic runs are drafts for testing only; do not publish them.
- Review links are in `runs.review_token` in the local database. Never publish them.
- The owner keeps using this clip. Do not ask for new videos or 1080p.
- Decided: no licence, and no need to ask the organisers about the 26 October commits.

## Next, in order

1. Edwin creates the empty public GitHub repo. Push `main` only, after his approval, then check it logged out.
2. Hosting: Vercel and Netlify do not fit. Prepare a container for Fly.io if Edwin approves the cost. He signs in himself.
3. Write the demo script, then Edwin records the 3-minute video. It must clearly state what was built during the event,
   or the project risks disqualification.
4. Edwin submits at the CV platform and lists everyone who worked on it.
5. Post on X for the Most Viral prize, and submit that link.

**Deadlines:** submissions close 4 Nov, 9:00 PM ET (5 Nov, 2:00 AM Accra). Tue 3 Nov is the last full day for help.
Viral engagement until 8 Nov end of day. Live finals 9 Nov, 12:00 PM ET.

## Do not

- Claim results beyond this one clip, or present fixtures as model output.
- Invent model outputs or prices. Amounts come from speech, and the seller approves every price.
- Commit `.env`, `.private/`, `footage/`, `runs/`, `var/` or `.claude/`, or print key values.
- Leave paid calls on, or retry them automatically. Edwin has said to use the credits where they improve results.
- Push, add a remote, deploy, rewrite history or alter timestamps without Edwin's approval.
- Push `--all` or `--mirror`, or touch the private backups: branch `backup/pre-rewrite-2026-10-26`,
  `refs/original/refs/heads/main`, and `C:\Users\Edwin\stall-to-store-backups\pre-rewrite-2026-10-26.bundle`.
