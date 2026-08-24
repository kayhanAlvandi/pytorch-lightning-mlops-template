# Plan: `TilePredictor.batch_predict()`

## Goal

Add a `batch_predict()` method to `TilePredictor` that processes multiple
images in one call, with a `tile_batch_size` parameter to control GPU
memory. This becomes the inference engine for `compute_reference.py`
(hundreds of validation images) and can later serve a
`POST /predict/batch` API endpoint.

The existing single-image `predict()` becomes a thin wrapper around
`batch_predict()` to avoid code duplication.

---

## Current state (what we're building on)

### `predict()` flow (api/predictor.py:417)
```
chans_reorder → log_image_metadata → preprocess_image → tile_image
→ log_tile_stack → log_tile_stack_member → compute_stats
→ log_tile_channel_stats → predict_tiles → majority_vote
→ log_image_prediction → log_tile_prediction
```

### Key reusable methods (no changes needed)
- `chans_reorder(image_channels, image_metadata)` — canonicalizes channel order, returns `(ndarray (C,H,W), metadata_dict)`. Already a standalone function.
- `preprocess_image(image: np.ndarray (C,H,W)) -> Tensor (C,H,W)` — per-channel percentile normalization.
- `tile_image(tensor) -> list[dict]` — splits into tiles, each dict has `tile` tensor + row/col/x/y/crop_size.
- `predict_tiles(tiles) -> list[dict]` — **already batches tiles into one `model(batch)` call**. This is the GPU hot path.
- `majority_vote(tile_predictions) -> dict` — pure CPU.
- `compute_stats(pixels) -> (mean, std, p1, p5, p95, p99)` — pure CPU, per-channel.

### DBLogger methods (database/dblogger.py)
All except `log_image_prediction` already accept lists and use
`executemany`. They each do their own `commit()`.

| Method | Input | Returns |
|---|---|---|
| `log_image_metadata` | `list[tuple]` | `list[int]` (IDs) |
| `log_tile_stack` | `list[tuple]` | `list[int]` (IDs) |
| `log_tile_stack_member` | `list[tuple]` | `list[int]` (IDs) |
| `log_tile_channel_stats` | `list[tuple]` | `list[int]` (IDs) |
| `log_image_prediction` | **single `tuple`** | `int` (ID) |
| `log_tile_prediction` | `list[tuple]` | `int` (count) |

**Gap:** `log_image_prediction` takes a single tuple, not a list. It
needs a batched variant for `batch_predict()`.

---

## Design

### Signature

```python
def batch_predict(
    self,
    samples: list[tuple[list[np.ndarray], dict]],
    tile_batch_size: int = 64,
) -> list[dict]:
    """Batch prediction over multiple images.

    Args:
        samples: list of (channels_image, image_metadata) tuples,
            one per image. channels_image is a list of (H, W) numpy
            arrays (one per channel). image_metadata is the dict
            built by the API loader or compute_reference (must have
            plate, well, field, root_path, shape, channels,
            channel_files; optionally label and is_reference).
        tile_batch_size: max number of tiles in one GPU forward pass.
            Controls GPU memory. Default 64. Set lower if OOM, higher
            if GPU is underutilized.

    Returns:
        list of result dicts (one per input sample, same order).
    """
```

### Why `tile_batch_size` and not `image_batch_size`

The GPU memory cost is proportional to the number of tiles in a
forward pass, not the number of images. One image can produce
anywhere from 1 to 100+ tiles depending on image size and crop_size.
So the batch size that controls GPU memory is **tiles**, not images.

`tile_batch_size=64` means: concatenate all tiles from all images,
then feed them to the model in chunks of 64 tiles per forward pass.

### Phase structure

```
Phase 1: Per-sample CPU prep (sequential)
  for each sample:
    chans_reorder → preprocess_image → tile_image
    accumulate: tiles, tile metadata, image metadata rows,
                tile_stack rows, tile_stack_member rows,
                channel_stats rows
  → all structures indexed by sample_idx

Phase 2: Batched GPU inference (chunked by tile_batch_size)
  concatenate all tile tensors across all samples
  for each chunk of tile_batch_size tiles:
    normalize + model(chunk) → logits
  concatenate all logits
  split logits back per-sample using tile count offsets

Phase 3: Per-sample majority vote (CPU, sequential)
  for each sample:
    majority_vote(that sample's tile predictions)

Phase 4: Batched DB writes (single thread, sequential calls)
  if db_logger:
    log_image_metadata(all_metadata_rows)          → all_img_ids
    log_tile_stack(all_tile_stack_rows)            → all_tile_stack_ids
    log_tile_stack_member(all_member_rows)         → all_member_ids
    log_tile_channel_stats(all_stats_rows)         → (IDs, unused)
    log_image_predictions(all_image_pred_rows)     → all_img_pred_ids  [NEW batched method]
    log_tile_prediction(all_tile_pred_rows)
```

### Index mapping (the tricky part)

The core challenge is tracking which IDs belong to which sample
after batched DB calls return flat ID lists.

**Per-sample offsets computed in Phase 1:**

```python
sample_offsets = []  # (tile_start, tile_end, member_start, member_end)
tile_cursor = 0
member_cursor = 0
for sample in samples:
    n_tiles = len(sample["tiles"])
    n_channels = len(sample["img_ids"])
    n_members = n_tiles * n_channels
    sample_offsets.append({
        "tile_start": tile_cursor,
        "tile_end": tile_cursor + n_tiles,
        "member_start": member_cursor,
        "member_end": member_cursor + n_members,
    })
    tile_cursor += n_tiles
    member_cursor += n_members
```

**After batched DB calls, split the flat ID lists:**

```python
# all_img_ids is flat: [img_id_ch0_s0, img_id_ch1_s0, ..., img_id_ch0_s1, ...]
# Per sample, img_ids are contiguous. Track per-sample channel count.
img_id_cursor = 0
for i, sample in enumerate(samples):
    n_channels = len(sample["channels"])
    sample["img_ids"] = all_img_ids[img_id_cursor : img_id_cursor + n_channels]
    img_id_cursor += n_channels

# all_tile_stack_ids is flat: one per tile, in sample order.
tile_stack_cursor = 0
for i, sample in enumerate(samples):
    n_tiles = len(sample["tiles"])
    sample["tile_stack_ids"] = all_tile_stack_ids[tile_stack_cursor : tile_stack_cursor + n_tiles]
    tile_stack_cursor += n_tiles

# all_member_ids is flat: tile-major, channel-major, in sample order.
# Per sample: n_tiles * n_channels IDs.
member_cursor = 0
for i, sample in enumerate(samples):
    n_tiles = len(sample["tiles"])
    n_channels = len(sample["img_ids"])
    n_members = n_tiles * n_channels
    sample["member_ids"] = all_member_ids[member_cursor : member_cursor + n_members]
    member_cursor += n_members
```

**Channel stats mapping (same as current predict):**
```python
for i, sample in enumerate(samples):
    n_channels = len(sample["img_ids"])
    for tile_idx, tile_info in enumerate(sample["tiles"]):
        tile_tensor = tile_info["tile"]  # (C, crop, crop)
        for channel_idx in range(n_channels):
            member_id = sample["member_ids"][tile_idx * n_channels + channel_idx]
            pixels = tile_tensor[channel_idx]
            all_stats_rows.append((member_id, *compute_stats(pixels)))
```

### Phase 2: chunked GPU inference

```python
all_tile_tensors = []
for sample in samples:
    for tile in sample["tiles"]:
        all_tile_tensors.append(tile["tile"])

all_tile_predictions = []
for start in range(0, len(all_tile_tensors), tile_batch_size):
    chunk = all_tile_tensors[start : start + tile_batch_size]
    batch = torch.stack(chunk).to(self.device)
    # normalize per tile in chunk
    normalized = torch.stack([self.normalize(batch[i]) for i in range(batch.shape[0])])
    logits = self.model(normalized)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    for j in range(batch.shape[0]):
        all_tile_predictions.append({
            "predicted_idx": preds[j].item(),
            "predicted_class": self.class_names[preds[j].item()],
            "confidence": probs[j, preds[j]].item(),
            "probabilities": {
                name: probs[j, k].item()
                for k, name in enumerate(self.class_names)
            },
        })
```

Then merge tile geometry (row/col/x/y) from Phase 1 tiles with
inference results, split per-sample using tile count offsets, and
run majority_vote per sample.

### `predict()` as wrapper

```python
def predict(self, channels_image, image_metadata):
    return self.batch_predict([(channels_image, image_metadata)], tile_batch_size=...)[0]
```

Default `tile_batch_size` for single-image predict: keep current
behavior (all tiles in one pass). Use a large default like 1024 or
`len(tiles)` so it's effectively one batch for a single image.

---

## Changes required

### 1. `database/dblogger.py` — add batched `log_image_predictions`

Add a new method (keep the existing single-tuple `log_image_prediction`
for backward compatibility):

```python
def log_image_predictions(self, image_predictions: list[tuple]) -> list[int]:
    """Batch insert image prediction rows.

    Args:
        image_predictions: list of tuples, each:
            (plate, well, field, run_id, p_label, t_label,
             total_tiles, vote_fraction, avg_confidence, is_reference)
    Returns:
        list of image_prediction IDs, one per input row, in order.
    """
    # same pattern as log_image_metadata: executemany + RETURNING id
```

### 2. `api/predictor.py` — add `batch_predict()`

- Add `batch_predict()` method with the 4-phase structure above.
- Refactor `predict()` to delegate to `batch_predict()`.
- Add a class-level default `tile_batch_size` (e.g. 64) set in
  `__init__` from a parameter or config.

### 3. `api/predictor.py` — extract `_run_inference_chunk()`

Pull the GPU forward-pass logic out of `predict_tiles()` into a
private helper so both `predict_tiles()` and `batch_predict()` can
use it without duplication:

```python
def _run_inference_chunk(self, tile_tensors: list[torch.Tensor]) -> list[dict]:
    """Run model on a list of tile tensors, return prediction dicts."""
```

`predict_tiles()` becomes:
```python
def predict_tiles(self, tiles):
    if not tiles:
        return []
    predictions = self._run_inference_chunk([t["tile"] for t in tiles])
    # merge geometry from tiles with predictions
    for i, tile_info in enumerate(tiles):
        predictions[i]["row"] = tile_info["row"]
        predictions[i]["col"] = tile_info["col"]
        predictions[i]["y"] = tile_info["y"]
        predictions[i]["x"] = tile_info["x"]
    return predictions
```

### 4. Tests — `tests/api/test_predictor_batch.py` (new file)

Tests with a fake model (no MLflow) to verify:

1. **Basic batch**: 3 synthetic images, verify 3 results returned in
   order, each with correct shape.
2. **Single image via batch_predict**: same result as `predict()`.
3. **`predict()` delegates to `batch_predict()`**: monkeypatch
   `batch_predict` and verify it's called with a 1-element list.
4. **tile_batch_size chunking**: 10 tiles, `tile_batch_size=4` →
   verify model is called 3 times (4+4+2), results are correct and
   in order.
5. **DB logging batched**: with FakeDBLogger, verify each log method
   is called **once** with all rows, not once per image.
6. **Index mapping correctness**: 2 images with different tile
   counts, verify `tile_stack_member_ids` map back to the correct
   tiles/channels.
7. **Channel stats**: verify stats rows are computed for every
   tile/channel pair across all samples.
8. **is_reference propagation**: samples with `is_reference=True`
   in metadata → verify it reaches `log_image_predictions` and
   `log_tile_prediction`.
9. **Empty samples list**: returns `[]`.
10. **GPU OOM fallback** (optional, skip if no GPU): set
    `tile_batch_size=1` and verify it still works (just slower).

### 5. No changes to `api/main.py`

The API still calls `predict()` for single-image requests. A future
`POST /predict/batch` endpoint would call `batch_predict()` directly,
but that's out of scope for this task.

---

## What we explicitly avoid

- **No threading/multiprocessing inside batch_predict.** CPU prep
  (preprocess + tile) is sequential. GPU inference is chunked but
  sequential. DB writes are sequential. This keeps the code simple
  and avoids Psycopg thread-safety issues.
- **No shared transaction across all DB writes.** Each DBLogger
  method commits independently, as it does now. A single-transaction
  wrapper can be added later if atomicity becomes a requirement.
- **No dynamic tile_batch_size tuning.** The caller picks the value.
  Auto-tuning based on GPU memory is a future enhancement.

---

## Implementation order

1. Add `log_image_predictions()` to `DBLogger` (batched version).
2. Extract `_run_inference_chunk()` from `predict_tiles()`.
3. Implement `batch_predict()` with the 4-phase structure.
4. Refactor `predict()` to delegate to `batch_predict()`.
5. Write tests in `tests/api/test_predictor_batch.py`.
6. Run tests, fix issues.
7. Run ruff on changed files.
8. Commit.

---

## Open questions for review

1. **Default `tile_batch_size`**: 64 is a reasonable default for
   224×224 tiles on a typical GPU. Should the default be configurable
   via `Settings` (api/config.py) or hardcoded in `TilePredictor.__init__`?
2. **`predict()` wrapper batch size**: when `predict()` delegates to
   `batch_predict()`, should it use the same `tile_batch_size` or
   pass a large number (e.g. `len(tiles)`) to preserve current
   one-shot behavior?
3. **Error handling**: if one sample fails preprocessing (e.g. bad
   image), should `batch_predict` raise immediately, or skip that
   sample and continue? Recommendation: raise immediately (fail fast).
   Reference computation can catch and log the error per-sample at
   a higher level.
