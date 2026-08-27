"""Score the benchmark dataset with a specific model (per run_id).

Loads the model for the served MLflow run (same loading path as the API's
TilePredictor), reads the registered benchmark samples from the database
(benchmark_dataset + benchmark_dataset_member + image_metadata, populated once
by register_benchmark.py), and runs each sample through the same predict()
pipeline used in production -- with the known benchmark t_label attached and
benchmark_id set -- so benchmark predictions land in the normal
image_prediction/tile_prediction tables, distinguished by benchmark_id.

Those benchmark predictions are a model's supervised-quality baseline: because
every model is scored on the same held-out inputs, their accuracy/F1 are
directly comparable, and run_quality_report.py compares a production window
against this baseline.

Resumable: skips benchmark samples that already have a prediction for this
run_id (via DBLogger.get_benchmark_predictions), so an interrupted run resumes
instead of re-scoring.
"""
from pathlib import Path

import numpy as np

from api.config import Settings
from api.predictor import TilePredictor
from database.dblogger import DBLogger
from monitoring.compute_reference import _load_single_image


def load_benchmark_data(samples: list[dict]) -> tuple[list[list[np.ndarray]], list[dict]]:
    """Load raw pixel data + build predict()-shaped metadata for each sample.

    Args:
        samples: grouped benchmark samples, each a dict with keys benchmark_id,
            plate, well, field, t_label, and channels (list of dicts with
            channel, root_path, file_name in channel_index order).

    Returns:
        (sample_images, images_metadata) matching TilePredictor.predict()'s
        expected shapes, with label = the benchmark t_label, benchmark_id set,
        and is_reference = False.
    """
    images_metadata = []
    sample_images = []
    for sample in samples:
        sample_image = []
        channel_files = []
        channels = []
        for ch in sample["channels"]:
            img = _load_single_image(Path(ch["root_path"]) / ch["file_name"])
            sample_image.append(img)
            channel_files.append(ch["file_name"])
            channels.append(int(ch["channel"]))

        image_metadata = {
            "plate": sample["plate"],
            "well": sample["well"],
            "field": sample["field"],
            "root_path": sample["channels"][0]["root_path"],
            "channels": channels,
            "channel_files": channel_files,
            "shape": sample_image[0].shape,
            "label": sample["t_label"],
            "is_reference": False,
            "benchmark_id": sample["benchmark_id"],
        }
        images_metadata.append(image_metadata)
        sample_images.append(sample_image)

    return sample_images, images_metadata


def _group_members(member_rows: list[dict]) -> list[dict]:
    """Group flat (sample, channel) rows into one dict per benchmark sample."""
    grouped: dict[int, dict] = {}
    for row in member_rows:
        bid = row["benchmark_id"]
        if bid not in grouped:
            grouped[bid] = {
                "benchmark_id": bid,
                "plate": row["plate"],
                "well": row["well"],
                "field": row["field"],
                "t_label": row["t_label"],
                "channels": [],
            }
        grouped[bid]["channels"].append({
            "channel": row["channel"],
            "root_path": row["root_path"],
            "file_name": row["file_name"],
        })
    return list(grouped.values())


def main():
    settings = Settings()

    if not settings.has_model_source:
        print("WARNING: No model source configured.")
        print("Set one of: API_MODEL_NAME or API_RUN_NAME")
        return

    if not settings.has_db_uri:
        print("ERROR: No database URI configured.")
        print("Set API_DB_URI to enable benchmark logging.")
        return

    source = settings.model_name or settings.run_name
    print(f"Loading model: {source}")

    db_logger = DBLogger(db_uri=settings.db_uri)
    print(f"Connecting to database: {settings.db_uri}")
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        predictor = TilePredictor(
            tracking_uri=settings.tracking_uri,
            experiment_name=settings.experiment_name,
            model_name=settings.model_name,
            run_name=settings.run_name,
            crop_size=settings.crop_size,
            stride=settings.effective_stride,
            device=settings.device,
            db_logger=db_logger,
        )

        run_id = predictor.model_info["run_id"]

        member_rows = db_logger.fetch_benchmark_members() or []
        samples = _group_members(member_rows)
        print(f"Found {len(samples)} registered benchmark samples.")
        if not samples:
            print("No benchmark samples registered -- run register_benchmark first. Nothing to do.")
            return

        # Skip samples already scored for this run_id, so an interrupted run resumes.
        already_done = set(db_logger.get_benchmark_predictions(run_id) or [])
        remaining = [s for s in samples if s["benchmark_id"] not in already_done]
        print(f"{len(already_done)} / {len(samples)} benchmark samples already scored for "
              f"run_id={run_id}; processing {len(remaining)} remaining.")
        if not remaining:
            print("Nothing to do.")
            return

        sample_images, images_metadata = load_benchmark_data(remaining)
        print(f"Loaded {len(sample_images)} benchmark sample images")

        n_ok, n_failed = 0, 0
        for i, (sample_image, image_metadata) in enumerate(zip(sample_images, images_metadata), start=1):
            key = (image_metadata["plate"], image_metadata["well"], image_metadata["field"])
            try:
                result = predictor.predict(sample_image, image_metadata)
                n_ok += 1
                print(f"[{i}/{len(remaining)}] {key} -> {result['predicted_class']} "
                      f"(label={image_metadata.get('label')})")
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                print(f"[{i}/{len(remaining)}] {key} FAILED: {e}")

        print(f"Done. {n_ok} succeeded, {n_failed} failed out of {len(remaining)}. "
              f"Results saved in database at {settings.db_uri}")
    finally:
        db_logger.close()


if __name__ == "__main__":
    main()
