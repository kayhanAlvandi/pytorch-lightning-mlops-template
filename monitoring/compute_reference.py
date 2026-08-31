"""Compute reference (validation-set) predictions for drift detection.

Loads the model for a given MLflow run (same loading path as the API's
TilePredictor), downloads its dataset_manifest.json, resolves the
validation samples' image files from disk, and runs each sample through
the same predict() pipeline used in production -- with is_reference=True
and the known validation label attached -- so reference rows land in the
same image_prediction/tile_prediction/tile_channel_stats tables as live
traffic, distinguished only by the is_reference flag.

Resumable: skips validation samples that already have a reference row for
this run_id (via DBLogger.get_reference_samples), so an interrupted run
can be re-launched and only processes what's missing.
"""
import json
from pathlib import Path

import numpy as np

from api.config import Settings
from api.predictor import TilePredictor
from database.dblogger import DBLogger


def _load_single_image(file_path: Path) -> np.ndarray:
    """Load a single raw (unnormalized) image file from disk."""
    import cv2
    import pillow_jxl  # noqa: F401  register JXL support with PIL
    from PIL import Image

    suffix = file_path.suffix.lower()
    if suffix == ".tif":
        image_source = cv2.imread(str(file_path), -1)
    elif suffix == ".jxl":
        image_source = Image.open(file_path)
    else:
        raise ValueError("image path should end with .tif or .jxl")
    if image_source is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return np.array(image_source, dtype=np.float32)


def load_validation_data(val_samples: list[dict]) -> tuple[list[list[np.ndarray]], list[dict]]:
    """Load raw pixel data + build predict()-shaped metadata for each val sample.

    Args:
        val_samples: list of sample dicts from dataset_manifest.json, each with
            keys plate, well, field, label, root_path, channel_files
            ({channel_number(str): filename}).

    Returns:
        (sample_images, images_metadata):
            sample_images: list (one per sample) of list[np.ndarray] (H, W) raw
                pixel arrays, one per channel.
            images_metadata: list of dicts matching TilePredictor.predict()'s
                expected image_metadata shape (plate, well, field, root_path,
                shape, channels, channel_files, label, is_reference).
    """
    images_metadata = []
    sample_images = []
    for sample in val_samples:
        root_path = Path(sample["root_path"])
        sample_image = []
        channel_files = []
        channels = []
        for channel, channel_file in sample["channel_files"].items():
            img = _load_single_image(root_path / channel_file)
            sample_image.append(img)
            channel_files.append(channel_file)
            channels.append(int(channel))

        image_metadata = sample.copy()
        image_metadata["channel_files"] = channel_files
        image_metadata["channels"] = channels
        image_metadata["field"] = int(sample["field"])
        image_metadata["shape"] = sample_image[0].shape
        image_metadata["is_reference"] = True
        images_metadata.append(image_metadata)
        sample_images.append(sample_image)

    return sample_images, images_metadata


def main():
    settings = Settings()
    db_logger: DBLogger | None = None

    if not settings.has_model_source:
        print("WARNING: No model source configured.")
        print("Set one of: API_MODEL_NAME or API_RUN_NAME")
        return

    if not settings.has_db_uri:
        print("ERROR: No database URI configured.")
        print("Set API_DB_URI to enable reference logging.")
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

        try:
            artifact_dir = predictor.model_info["artifact_dir"]
            manifest_path = Path(artifact_dir) / "dataset_manifest.json"
            if not manifest_path.exists():
                print(f"dataset_manifest.json not found in run {run_id} artifacts "
                      f"(looked in {artifact_dir})")
                return
            with open(manifest_path) as f:
                manifest = json.load(f)
        except KeyError:
            print(f"No artifact_dir available for run {run_id} -- could not "
                  "download run artifacts when the model was loaded.")
            return
        except Exception as e:  # noqa: BLE001
            print(f"Failed to read dataset manifest: {e}")
            return

        val_samples = manifest.get("val_samples", [])
        print(f"Found {len(val_samples)} validation samples for model: {source}")
        if not val_samples:
            print("No val_samples found in dataset_manifest.json -- nothing to do.")
            return

        # Skip samples that already have a reference row for this run_id, so an
        # interrupted run resumes instead of re-inserting duplicate rows.
        # field is coerced to int on both sides: image_metadata.field is INTEGER,
        # while older manifests stored the raw zero-padded string ("001"), which
        # would never compare equal and silently defeat the resume.
        already_done = {
            (plate, well, int(field)) for plate, well, field in
            (db_logger.get_reference_samples(run_id) or [])
        }
        remaining = [
            s for s in val_samples
            if (s["plate"], s["well"], int(s["field"])) not in already_done
        ]
        print(f"{len(already_done)} / {len(val_samples)} validation samples already have "
              f"reference rows for run_id={run_id}; processing {len(remaining)} remaining.")
        if not remaining:
            print("Nothing to do.")
            return

        sample_images, images_metadata = load_validation_data(remaining)
        print(f"Loaded {len(sample_images)} sample images")


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
        if db_logger is not None:
            db_logger.close()


if __name__ == "__main__":
    main()
