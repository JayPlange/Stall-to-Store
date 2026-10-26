# Label file schema

One file per video: `eval/labels/<video>.json`. Labels hold ground truth only; never the video or seller details.

```json
{
  "video": "stall01",
  "split": "tuning",
  "run_id": "r_20261027T101500_1a2b3c4d",
  "products": [
    {"label_id": "p1", "brand": "Brand", "name": "Evaporated Milk", "size": "160 g",
     "price_pesewas": 1200, "price_span": [3.0, 4.2]}
  ],
  "track_to_label": {"t1": "p1", "t2": null}
}
```

- `split`: `tuning` (thresholds may be tuned on it) or `heldout` (never tuned on).
- `price_pesewas`: integer, or `null` if the seller never says a price for it.
- `price_span`: seconds in which the seller says the price.
- `track_to_label`: filled by a human after the run, from the keyframes on the review screen. Each system
  track maps to a `label_id`, or `null` for a false detection. Unlisted tracks count as false detections.

Identification is scored on the model's original guess (`products.model_brand/model_name`), not the seller's
edits. Auto-match precision uses `products.model_price_pesewas` for products with `match_status = 'auto'`.
