# Build status

Last updated: 2026-11-02. One upload now runs the whole pipeline. All figures come from one clip, which was also used
for tuning, so treat them as a first benchmark rather than measured accuracy.

## Gates

| Gate | Status |
|---|---|
| G1a: real outputs usable from code | **PASS** for SAM 3.1, Muse Voice Transcribe and Muse Spark (text and vision) |
| G1b: detection recall ≥80%, auto-match precision ≥90% | **PASS on one tuning clip:** 7/7 detection with two SAM prompts; 8/8 precision |
| G2: eval prints the table | **Partial:** table runs on one real video; SPEC's gate asks for two. The owner has decided to keep using this clip |
| G3: real video to approved shop, real order reaches the seller | **PASS with a limit:** order S2S-KPU3XE went through WhatsApp Web, the seller received it, and it was confirmed. Not yet tried in a phone browser |
| G4: full flow, rough demo exists | Full flow works from one upload; no demo recorded yet |

## End-to-end runs on the real clip

| | Run 1 (`r_20261102T123812`) | Run 2 (`r_20261102T131555`, after filters) |
|---|---|---|
| Stages | 9 of 9, no failures | 9 of 9, no failures |
| Draft cards | 11 (2 junk) | 8 (0 junk) |
| Product types priced correctly | 7 of 7 | 7 of 7 |
| Draft ready | 2¼ minutes | 3 minutes |
| Everything finished | nearly 14 minutes | about 11½ minutes |
| Estimated cost | $0.126 | $0.118 |

Run 1's junk was a floor tile and a flap of the kettle box, both given the kettle's price. Run 2's filter removed the
floor tile again, and also one small, blurry Comfort sachet the model could not read. The run log records this, but the
seller is not told.

## Benchmark: automatic run, 1 video, 7 products

| Metric | Result | Note |
|---|---|---|
| Detection recall | 7/7 | Needs two SAM prompts; "product package" alone found 5 of 7 product types |
| Duplicate rate | 1/7 | **Misses target.** Two Comfort sachets are two tracks for one product |
| Identification | 5/7 | **Misses target.** "AILYON" lost its S; one Comfort card had no product name |
| Auto-match precision / coverage | 8/8, 8/8 | |
| Muse Spark price links | 7/7 | |
| Cost per video | about $0.12 | Estimate |

From `python eval/run_eval.py --labels eval/labels_automatic`. Run `python eval/run_eval.py` for the earlier manual run,
which reached publishing: 9 seller corrections, 1/1 completion, 0 order safety failures of 1 confirmed order.

## Live calls: 24, about $0.35 estimated

Estimates use Meta's published prices. The dashboard is the source of truth for billing. Reasoning tokens drive Muse
Spark's cost and vary by run. Raw sanitised responses are in the gitignored `runs/` folder.

## Decisions

- **Licence:** none, by the owner's decision. The guide requires a public repository and names no licence.
- **26 October commits:** the owner accepts this. The first commit (`34dfff0`) is 17:40 GMT on 26 October, inside the
  event window, which opened at 17:00 GMT. The guide's "Building period" is 27 Oct to 4 Nov. Commit timestamps have been
  preserved and will not be altered.
- **New videos:** none. The owner's phone records this clip's quality, and this is the clip the project uses.

## Remaining

1. Create the public repo. Push `main` only, never `--all` or `--mirror`, then check it in a logged-out browser.
2. Hosting: not built. Vercel and Netlify do not fit (ffmpeg, long runs, stored files). Fly.io is recommended, subject to
   its pricing. Any public server must set `STS_UPLOAD_CODE`.
3. Record the 3-minute demo. It must clearly show what was built during the event.
4. Submit on the CV platform by 4 Nov, 9:00 PM ET, and list everyone who worked on it.

## Findings

- **Timing:** Muse Voice Transcribe has no word timestamps, and real narration came back as one turn on every run.
  Timing cannot link prices to products on this clip, so linking relies on spoken names and Muse Spark.
- **Run-to-run variation:** SAM returned 8 or 9 first-prompt tracks, and "Ailyons" was heard as "Allianz", "Elion's" and
  "Elyon's". Muse Spark still matched the kettle each time.
- **SAM prompts:** one noun phrase per request. "bottle" found the Vim and Bel-Aqua that "product package" missed.
- **Parsing:** `meta-sam-parser` decodes every mask in pure Python, which took 18 minutes. Masks are now decoded only
  when needed. The tracking video still decodes about 500 of them, which is why it takes about 8 to 10 minutes.
- **Word matching:** "the" once caused a wrong match, so filler and category words no longer count.

## Known gaps

- The `associate` stage treats a whole-clip turn as real timing. Muse Spark's links compensate on this clip.
- Identical items become separate listings. Keyword hints to transcription are untested. Muse Image is not built.
- Word lists, cut-offs and thresholds are tuned on one clip.
- Review links are bearer URLs with no login or CSRF token. Demo only.
- A server restart during a run leaves that run marked as processing.

## Local setup

Python 3.12 venv at `~/.venvs/stall-to-store`; ffmpeg/ffprobe on `PATH`; 138 tests pass.
