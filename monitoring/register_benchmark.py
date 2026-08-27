"""Register the benchmark dataset (one-time, model-independent).

The benchmark is a fixed, curated set of samples that were never in any model's
training set, used as the supervised-quality baseline. This script registers
those samples into the database: it inserts their single-channel image files
into ``image_metadata``, one ``benchmark_dataset`` row per ``(plate, well,
field)`` with its known ``t_label``, and the ``benchmark_dataset_member`` rows
linking them. No model and no inference are involved here -- scoring the
benchmark for a specific model is a separate step (``compute_benchmark.py``).

Input is a set of image file paths (e.g. from a glob); ``(plate, well, field,
channel)`` are parsed from each filename with the same parser the API uses, so
parsing stays consistent with production. Ground-truth labels are resolved per
``(plate, well)`` from MongoDB (same ``tools.loading.getCategories`` path as
training/backfill); samples whose well has no label yet are skipped.

Idempotent: ``benchmark_dataset`` is keyed on ``(plate, well, field)`` and
already-registered samples are skipped, so it is safe to rerun as the benchmark
set grows.
"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from database.dblogger import DBLogger
from monitoring.backfill_labels import resolve_labels
from monitoring.config import MonitoringSettings
from utils.filename_parser import extract_info_from_filename


def _read_shape(file_path: Path) -> tuple[int, int]:
    """Read (shape_x, shape_y) = (height, width) without loading full pixels.

    Matches compute_reference's convention (shape_x = rows/H, shape_y = cols/W).
    """
    import pillow_jxl  # noqa: F401  register JXL support with PIL
    from PIL import Image

    with Image.open(file_path) as im:
        width, height = im.size
    return height, width


def group_samples(paths: list[str]) -> dict[tuple[str, str, int], list[dict]]:
    """Group image files into benchmark samples keyed by (plate, well, field).

    Each grouped file dict carries plate, well, field, channel, root_path,
    file_name. Files with unrecognised names (no plate/well parsed) are skipped.
    """
    samples: dict[tuple[str, str, int], list[dict]] = {}
    for p in paths:
        path = Path(p)
        info = extract_info_from_filename(path.name)
        if not info["plate"] or not info["well"]:
            print(f"  skip (unparseable name): {path.name}")
            continue
        key = (info["plate"], info["well"], info["field"])
        samples.setdefault(key, []).append({
            "plate": info["plate"],
            "well": info["well"],
            "field": info["field"],
            "channel": info["channel"],
            "root_path": str(path.parent),
            "file_name": path.name,
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Register benchmark samples into the database.")
    parser.add_argument("globs", nargs="+",
                        help="One or more glob patterns (or explicit paths) of benchmark image "
                             "files, e.g. '/mnt/O/benchmark/*.jxl'.")
    parser.add_argument("--file", help="path to txt file containing list of benchmark image files")
    args = parser.parse_args()

    settings = MonitoringSettings()
    if not settings.has_db_uri:
        print("ERROR: No database URI configured. Set MONITORING_DB_URI.")
        return

    paths: list[str] = []
    if args.file:
        with open(args.file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
    else:
        for pattern in args.globs:
            matched = glob(pattern)
            paths.extend(matched) if matched else paths.append(pattern)
    paths = sorted(set(paths))
    print(f"Found {len(paths)} candidate image files.")

    samples = group_samples(paths)
    print(f"Grouped into {len(samples)} benchmark samples (plate, well, field).")
    if not samples:
        print("Nothing to register.")
        return

    db_logger = DBLogger(db_uri=settings.db_uri)
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        # Resume: skip samples already registered.
        already = set(db_logger.get_benchmark_samples() or [])
        pending = {k: v for k, v in samples.items() if k not in already}
        print(f"{len(already)} already registered; processing {len(pending)} new samples.")
        if not pending:
            print("Nothing to do.")
            return

        # Resolve labels once for all distinct wells.
        wells = sorted({(plate, well) for (plate, well, _field) in pending})
        labels = resolve_labels(wells)
        print(f"MongoDB resolved {len(labels)} / {len(wells)} wells to a treatment.")

        n_ok, n_skipped = 0, 0
        for (plate, well, field), files in pending.items():
            t_label = labels.get((plate, well))
            if t_label is None:
                n_skipped += 1
                print(f"  skip {plate}/{well}/{field}: no label in MongoDB yet")
                continue

            # Channels ascending (matches training/predict input channel order).
            files_sorted = sorted(files, key=lambda f: f["channel"])
            image_rows = []
            for f in files_sorted:
                shape_x, shape_y = _read_shape(Path(f["root_path"]) / f["file_name"])
                image_rows.append((
                    f["plate"], f["well"], f["field"], f["channel"],
                    f["root_path"], f["file_name"], shape_x, shape_y,
                ))
            img_ids = db_logger.log_image_metadata(image_rows)

            benchmark_id = db_logger.log_benchmark_sample((plate, well, field, t_label))
            members = [
                (benchmark_id, img_id, channel_index)
                for channel_index, img_id in enumerate(img_ids)
            ]
            db_logger.log_benchmark_members(members)
            n_ok += 1
            print(f"  registered {plate}/{well}/{field} -> {t_label} "
                  f"({len(img_ids)} channels), benchmark_id={benchmark_id}")

        print(f"Done. {n_ok} samples registered, {n_skipped} skipped (no label).")
    finally:
        db_logger.close()


if __name__ == "__main__":
    main()
