# INFERENCE CONTRACT (v0.1)

A real predictor accepts a decoded RGB image and returns a JSON-compatible dictionary:

```json
{
  "model_id": "effnetv2s-stage8-run001",
  "class_list_sha256": "<sha256 of ordered active classes in the checkpoint>",
  "predictions": [
    {"class_id": 1, "class_key": "potato", "label_zh": "土豆", "score": 0.91},
    {"class_id": 0, "class_key": "corn", "label_zh": "玉米", "score": 0.06}
  ],
  "top1_id": 1,
  "latency_ms": 24.5,
  "status": "ok"
}
```

`score` is a softmax class score **not a calibrated probability of correctness**; values and time above are schema examples, never measurements. The `predictions` list is ordered descending and contains up to Top-3 classes. A `model not ready` condition must be shown explicitly; do not show fake food predictions.

- Input: one JPEG/PNG or PIL RGB image. MVP is **single-main-food image classification**. Mixed-food multi-object detection is out of scope until approved.
- All model adapters must use the SAME `configs/classes.json` mapping. Reject a checkpoint whose embedded **active class list** does not match the same ordered prefix of the frozen class list (8-class checkpoints may coexist with later 12/16-class configs).
- Invalid/unreadable image => clear error, no prediction. Exceeding memory/timeout => clear error, no silent fallback. Model untrained/not loaded => explicit not-ready state.
- Measure inference latency over repeated independent images, with device and preprocessing included/excluded stated clearly. Avoid claiming deployment capability based on desktop performance.
