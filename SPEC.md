# Stall to Store: technical specification

Meta Global AI Developer Hackathon, Indie track, theme: Build for Your World.
Hacking started at kickoff on 26 October 2026 ("Hacking starts now"). Build freeze: 4 November, 6:00pm Accra. Submission cutoff: 5 November, 2:00am Accra (9:00pm ET on 4 November).

**Judging rubric (from the kickoff, replaces the equal weighting in the written guide):** Impact 30%, Use of Meta AI tooling 25%, Demo 25%, Technical depth 20%. Same rubric in Stage 2 and the live final on 9 November.

**Anti-projects to stay clear of:** simple image analysers, Streamlit apps, dashboards as the main feature, basic RAG. The review screen is a step in the flow, never the product. No Streamlit anywhere.

**Submission:** CV platform entry, a 3-minute video showing only work built during the event, a public GitHub repo that opens in a logged-out browser. Viral prize engagement counts until 8 November, 11:59pm ET (4:59am Accra on 9 November).

## 1. The one-sentence product

A market seller films their stall for under a minute while saying prices out loud, and gets back a live shop page whose products, photos and prices come from that video, with orders arriving on WhatsApp.

## 2. Ground rules

1. **Real outputs only.** Curated footage, good lighting and seller confirmation are fine. Hand-edited or fabricated results presented as model output are not.
2. **Every model has a necessary job.** No model is added only to increase the count.
3. **Nothing reaches the shop page unconfirmed.** Every published price has been approved by the seller on the review screen.
4. **All new code.** The repository starts on 27 October. Nothing is copied from KasaFlow. The README states what was built during the event.
5. **The public repo holds code only.** Footage, API keys and seller details stay out of git.
6. **Meta API details are unconfirmed.** Section 6 describes our internal interfaces, not real endpoints. Before writing any adapter, prove from Python: authentication, that the endpoint is available to this account, the actual response shape, the usage units billed, and the failure behaviour. No pipeline stage is written against an assumed response.
7. **API access is the first gate.** The Meta account currently shows a payment-authorisation restriction. A working playground chat does not prove that SAM 3.1 video or transcription calls work from code. Clear the restriction and make one real call to each model before anything else.
8. **There is no hard spending cap.** The billing control available is an email notification at a threshold, not a guaranteed limit. Make deliberate test calls, log every call's usage, never retry automatically in a loop, and check current billing terms before running a batch.
9. **Privacy by filming, not by models.** Film products and the seller's hands, not faces. If a face appears, cut that footage from anything published. Segmentation is not a privacy control.

## 3. Scope

**In scope (the demo path)**
- One product category: packaged goods (provisions, cosmetics, toiletries) with readable brands.
- Video: 20 to 60 seconds, phone footage, seller narrating prices in English (Twi is a stretch goal, see section 11).
- Draft catalogue from video, seller review and correction, published shop page.
- Ordering by WhatsApp: browse, propose order, explicit confirmation, seller notified.
- An evaluation script with real numbers.

**Out of scope**
Payments, delivery, inventory sync, multiple sellers per account, loose produce, fabric, a seller dashboard beyond the review screen, and analytics.

## 4. Architecture

```
 Seller phone video (20 to 60s)
        |
        v
 [1 Ingest]  ffmpeg: audio.wav + frames at 2 fps + metadata
        |
        +-------------------------------+
        v                               v
 [2 Transcribe]                   [3 Track]
 Muse Voice Transcribe            SAM 3.1, text prompts on video
 words + timestamps               tracks: id, frames, boxes, masks
        |                               |
        |                         [4 Keyframes]
        |                         best frame per track (area, sharpness)
        |                               |
        v                               v
 [5a Price mentions]              [5b Identify]
 Muse Spark parses transcript     Muse Spark names each keyframe crop:
 into {text, amount, t_start,     brand, product, size, confidence
 t_end, referent words}                 |
        |                               |
        +---------------+---------------+
                        v
               [6 Associate]
               temporal scoring proposes matches,
               Muse Spark verifies on a numbered contact sheet,
               low confidence goes to the seller
                        |
                        v
               [7 Cutouts]  real pixels from SAM masks
               optional Muse Image background clean-up, with a guard
                        |
                        v
               [8 Draft catalogue] --> [9 Seller review] --> [10 Publish]
                                                                  |
                                                                  v
                                       Shop page /s/{slug}   "Order on WhatsApp"
                                                                  |
                                                                  v
                                       [11 Ordering agent]  WhatsApp Cloud API
                                       Muse Glimmer (bonus) or Muse Spark
                                       tools: list, get, propose_order, confirm_order
                                                                  |
                                                                  v
                                       Order stored, seller notified on WhatsApp
```

Every stage reads and writes JSON artefacts in `runs/{run_id}/`. Any stage can be re-run alone, and every result can be inspected, which is what makes the demo honest and debugging fast.

## 5. Model responsibilities

| Stage | Model | Job | Why it is needed | Fallback |
|---|---|---|---|---|
| 2 | Muse Voice Transcribe | Transcript with whatever timing it returns (word, segment or none) | Lets prices be matched by time | Muse Spark on the audio, if it accepts audio; otherwise seller-assisted price entry |
| 3 | SAM 3.1 | Find, track and mask every product across the video | Each product counted once; real cut-out photos | Muse Spark on sampled frames with no tracking, de-duplicated by name (weaker) |
| 5a | Muse Spark | Turn narration into structured price mentions | Speech is messy: "these ones, 12 cedis" | Regex for amounts plus the seller fixing referents |
| 5b | Muse Spark | Name each product from its crop | Brand and size for the listing | Seller types the name |
| 6 | Muse Spark | Rank and verify candidate price matches on a numbered contact sheet | The multimodal moment of the demo | Not required for publishing: candidates go to the seller unverified |
| 7 | Muse Image (optional) | Clean product photo backgrounds | Nicer shop page | Plain cut-out on white; cut this first |
| 11 | Muse Glimmer (bonus) | Ordering agent with tool calls | Open-weights bonus entry; cheap self-hostable agent | Muse Spark runs the agent |

Log every model call to a `model_calls` table: run_id, stage, model id, provider, latency, token or unit counts, estimated cost. This verifies the open-weights entry and tracks spending.

## 6. Interfaces (adapters, confirm against docs on Day 1)

All Meta calls go through `app/models/`. Each adapter saves the provider's **raw response** to `runs/{run_id}/raw/{stage}.json`, then normalises it into our internal dataclasses. If a provider returns something different from what we expect, only the adapter changes. Do not assume a video call behaves like a sequence of frame calls: verify which operation SAM 3.1 actually supports.

```python
# app/models/transcribe.py
def transcribe(audio_path: str) -> Transcript
# Transcript(text, granularity: 'word'|'segment'|'none', units: list[Unit])
# Unit(text, t_start: float | None, t_end: float | None, confidence: float | None)

# app/models/segment.py
def track(video_path: str, prompts: list[str]) -> list[Track]
# Track(track_id, label, frames: list[FrameHit])
# FrameHit(frame_idx, t, box_xyxy, mask_path, score)

# app/models/vision.py
def identify(crops: list[Crop]) -> list[ProductGuess]   # brand, name, size, confidence
def parse_prices(transcript: Transcript) -> list[PriceMention]
def rank_candidates(sheet_path: str, mention: PriceMention,
                    candidates: list[Candidate]) -> list[RankedCandidate]   # optional step

# app/models/image.py (optional)
def clean_background(cutout_path: str) -> str | None    # None if the guard rejects it

# app/models/agent.py
def agent_turn(history: list[Msg], tools: list[ToolSpec]) -> AgentReply
```

**Known from public sources, still to confirm:** SAM 3.1 is called as `sam-3.1` through the Responses API with bearer authentication and returns boxes, masks and frame-by-frame tracking. Meta's docs say OpenAI-compatible clients work with the Meta Model API. Muse Voice Transcribe's model id is `muse-voice-transcribe-1.0`.

**Unknown until Day 1:** SAM 3.1 price, maximum video length and resolution; whether Muse Spark accepts video and audio directly through the API and in what format; whether the transcription output includes word timestamps; Glimmer's model id and tool-calling format on the partner host.

## 7. Price association (the core algorithm)

**Inputs:** tracks with per-frame boxes, price mentions with whatever timing the transcript provides.

The deterministic scorer **proposes candidates; it is never the source of truth.** The largest object in the centre of the frame is not necessarily the one the seller means.

**Step 1, candidates by time.** For each price mention, list every track visible within a window around it (`t_start - 1.5s` to `t_end + 1.0s` with word timing; the whole segment with segment timing). Keep the top few candidates by a simple score: visibility in the window, then centrality, then size, plus a bonus when the referent words fuzzy-match a product name. With no timing at all, every track is a candidate and the step falls through to the seller.

**Step 2, rank with narration and visual evidence.** Muse Spark sees a contact sheet (keyframes with numbered boxes) and the transcript lines, and ranks the candidates for each mention with a confidence. This step is optional: if the model cannot take the inputs or costs too much, skip it.

**Step 3, decide.** Matching is many-to-one, not one-to-one:
- One price may apply to several tracks ("these ones, 12 cedis" across identical items).
- A product mentioned twice keeps the latest clear price and flags the conflict.
- **Auto-matched** only when the top candidate clearly beats the rest and, if Step 2 ran, Spark agrees. Still shown to the seller.
- **Needs seller** when candidates are close, the model disagrees, or timing is missing.
- **No price** for tracks no mention reached; the seller types a price or hides the item.

Thresholds are tuned on tuning videos only, never on held-out videos. Publishing never depends on Step 2.

## 8. Data model (SQLite)

```
runs(id, created_at, video_path, status, stage, error)
products(id, run_id, track_id, brand, name, size, keyframe_path, cutout_path,
         image_path, price_pesewas, price_source[auto|seller|none],
         match_confidence, status[draft|approved|hidden])
price_mentions(id, run_id, text, amount_pesewas, t_start, t_end, assigned_product_id)
shops(id, run_id, slug, seller_name, seller_whatsapp, published_at)
orders(id, shop_id, customer_wa, status[proposed|confirmed|cancelled|expired],
       lines_json, total_pesewas, proposal_hash, created_at, confirmed_at)
model_calls(id, run_id, stage, model, provider, latency_ms, units_json, cost_usd, created_at)
```

Prices are stored as integers in pesewas. The order total is always computed by the backend from approved product prices, never by a model.

## 9. Ordering (rebuilt fresh, in two stages)

**Stage A, minimum working version (P0).** The customer picks products on the shop page. The backend computes the total from approved prices and stores a proposal with a price snapshot. A WhatsApp click-to-chat link opens with the order prefilled and a short reference code, and the seller really receives it. The README and demo describe this accurately as click-to-chat, not an automated agent.

**Stage B, full version (P2, only if Cloud API credentials and webhooks work).** Inbound WhatsApp messages reach an agent with tools:
- `list_products` and `get_product` are read-only.
- `propose_order(lines)` computes the total from approved prices, stores the proposal with a hash, a price snapshot and a 15-minute expiry, and writes nothing else.
- `confirm_order(proposal_hash)` is the only write path. It acts only on a stored, unexpired proposal bound to the same customer, whose price snapshot still matches current approved prices, and is idempotent.
- The model never sees or sets prices directly. It receives them from tools.

**Required tests (both stages where they apply):** confirming without a proposal, an expired proposal, another customer's proposal, a duplicate confirmation, ordering a hidden product, and a price changed after the proposal (stale snapshot).

## 10. Repository layout

```
stall-to-store/
  README.md            what it does, built-during-hackathon statement, Meta model usage
  SPEC.md              this file
  .env.example         keys and model ids, no values
  app/
    main.py            FastAPI app
    pipeline/          ingest, transcribe, track, keyframes, identify, associate, cutouts, publish
    models/            Meta and partner adapters (section 6)
    orders/            proposals, click-to-chat links, order safety
    agent/             WhatsApp webhook and agent tools (Stage B only)
    web/               review screen and shop page (Jinja + HTMX, mobile first)
    db.py
  eval/
    labels/            ground truth JSON per video (not the videos)
    run_eval.py        prints the metrics table in section 12
  tests/
  runs/                gitignored
  footage/             gitignored
```

Stack: Python 3.12, FastAPI, SQLite, ffmpeg, Jinja + HTMX, pytest. Hosted on one small VM or container. Stage A needs only a wa.me link; Stage B needs the WhatsApp Cloud API on a test number.

## 11. Priorities

| Priority | Feature | Decision |
|---|---|---|
| P0 | Video ingestion and real segmentation | Must work |
| P0 | Product and price proposals | Must work |
| P0 | Seller review and publish | Must work |
| P0 | Shop page and click-to-chat order (Stage A) with safety tests | Must work |
| P1 | Evaluation and usage logging | Must work |
| P2 | Muse Spark candidate ranking (Step 2) | Strong for the demo, never required to publish |
| P2 | WhatsApp agent (Stage B), with Muse Glimmer for the open-weights bonus | Only if access and time permit |
| P2 | Muse Image background clean-up, with a guard rejecting edits that change the product | Cut first |
| P2 | Twi or mixed narration | Report honestly whatever the result |
| P3 | Public "send a video, get a shop" intake for the X launch | Only if everything above is done; rate-limited, seller-reviewed |

## 12. Evaluation

**Ground truth per video** (`eval/labels/{video}.json`): every product with brand, name and price, and the time span in which the seller says the price.

| Metric | Definition | Target |
|---|---|---|
| Detection recall | Labelled products found by at least one track | 80% or more |
| Duplicate rate | Products represented by more than one track | 10% or less |
| Identification accuracy | Correct brand and name among found products | 80% or more |
| Auto-match precision | Correct price among products the system auto-matched | 90% or more |
| Auto-match coverage | Share of priced products the system auto-matched rather than sent to the seller | Report it, aim for 50% or more |
| Seller corrections | Edits needed before publishing | Report it, lower is better |
| Time to shop | Upload to published page, including review | Report it |
| Cost per video | From `model_calls` | Report it |
| End-to-end completion | Videos that produce a published shop | Report it |
| Order safety failures | Invalid confirmed orders | 0 |

Precision and coverage are always reported together: 95% precision at 10% coverage is a very different system from 85% at 80%.

Split: about one third of the videos for tuning, the rest held out. With 3 to 5 videos, present results as a small initial benchmark, report the number of products tested, and show failures as well as successes.

## 13. Build plan

| Day | Date | Goal | Gate (pass or fail) |
|---|---|---|---|
| 0 | Mon 26 Oct | Hacking is open from kickoff. Claim credits (forms in #resources), clear the payment-authorisation restriction, set the spend notification, read API docs, create the public repo with this spec. Start filming (products and hands, no faces) and labelling. | Credits claimed, account unrestricted, repo created |
| 1 | Tue 27 Oct | **Prove access, then the riskiest assumption.** One real call to each model from Python; save raw responses; record usage units and failure behaviour. Then one video through SAM, transcription and candidate matching with throwaway scripts. | **G1a:** real outputs accessible and usable from code. **G1b:** detection recall 80% or more, and auto-match precision 90% or more at any coverage → full build; coverage decides how much the seller does. Below 90% precision → seller-assisted matching becomes the plan |
| 2 | Wed 28 Oct | Pipeline stages 1 to 4 as proper code, artefacts on disk, `model_calls` logging. Agents 102 session at 4pm. | One video processed end to end into tracks and keyframes |
| 3 | Thu 29 Oct | Stages 5 and 6 (identify, prices, association) and `run_eval.py`. | **G2:** eval runs on 2 videos and prints the table |
| 4 | Fri 30 Oct | Stage 7 cutouts; review screen. Fireworks session at 4pm. | Seller can correct and approve a draft |
| 5 | Sat 31 Oct | Publish, shop page, order Stage A (click-to-chat with stored proposal) and its tests. | **G3:** a real video produces a seller-approved shop, and a real demo order reaches the seller |
| 6 | Sun 1 Nov | Buffer for anything slipped. Then P2: Spark candidate ranking; Stage B agent if Cloud API access works. | P0 and P1 all done |
| 7 | Mon 2 Nov | P2 items in priority order (Glimmer, Muse Image). Record a rough demo. Post the X teaser (tracking overlay). | **G4:** full flow end to end on a real video; rough demo exists |
| 8 | Tue 3 Nov | Run all videos, final eval on held-out set, fix the most visible failures, record demo takes. Last day for organiser troubleshooting. | Results table final |
| 9 | Wed 4 Nov | Edit the 3-minute video, README, repo check. **Freeze 6pm.** Submit, then post on X. | Submission confirmed in the portal |

**Cut order if a day slips:** follow the priority table from the bottom up. Never cut P0 or P1.

## 14. Demo (3 minutes)

| Time | Shows |
|---|---|
| 0:00 to 0:15 | A seller films a real stall and says prices. "This is how most of Accra's shops sell. No website, no catalogue." |
| 0:15 to 0:50 | SAM 3.1 tracking overlays on the footage; products cut out one by one |
| 0:50 to 1:25 | The price matching: transcript lines connected to products, confident matches and flagged ones |
| 1:25 to 1:50 | Seller review: fixes one flagged price, publishes |
| 1:50 to 2:25 | The shop page next to the original video; a customer orders and the seller receives it on WhatsApp (described as click-to-chat unless Stage B is built) |
| 2:25 to 2:50 | Results table from the held-out videos and the model usage summary |
| 2:50 to 3:00 | "Built during the hackathon" and the repo link |

## 15. Budget (estimates to replace with Day 1 measurements)

| Item | Estimate | Confidence |
|---|---|---|
| Muse Spark, about 100 pipeline runs | $25 to $30 | Low: video and audio token costs unconfirmed |
| Muse Image, about 2,000 images | about $20 at $0.01 each | Medium |
| Muse Voice Transcribe | under $1 at $0.18 per audio hour | High |
| SAM 3.1 on video | Unknown | None: measure in the first hour |
| Glimmer on a partner host | about $0 if partner credits arrive | Low |

There is no hard cap: the billing control is an email notification. Make deliberate test calls, log usage per call in `model_calls`, never retry in a loop, and check current billing terms before running a batch. If SAM on video is expensive, sample frames at a lower rate or cut clips shorter.

## 16. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| API access blocked | Day 1 G1a fails; payment restriction persists | Ask in #office-hours and #questions immediately; nothing else proceeds until one real call works |
| Price matching fails | Auto-match precision below 90% | Seller-assisted matching: system proposes candidates, seller confirms each price |
| SAM video too slow or costly | Day 1 measurements | Track on 1 to 2 fps frames, shorter clips |
| Spark cannot take video or audio | Day 1 docs check | Frames as images plus transcript text |
| No word timestamps | Day 1 check of the raw response | Segment-level windows; with no timing, every price goes to the seller |
| WhatsApp Cloud API setup stalls | Credentials or webhook fail | Stay on Stage A click-to-chat |
| Duplicate tracks | Duplicate rate above 10% | Merge tracks with matching names and overlapping appearance |
| Partner credits never arrive | By Day 6 | Agent stays on Muse Spark; skip the open-weights bonus |
| Bad footage | Blur, glare | Refilm on Day 0 or 1; keep a clean reserve video |
| Faces in public footage | Any frame | Film products and hands only; cut any footage showing a face from published material |
