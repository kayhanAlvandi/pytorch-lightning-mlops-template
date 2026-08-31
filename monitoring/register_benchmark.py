"""Register the benchmark dataset (one-time, model-independent).

The benchmark is a fixed, curated set of samples that were never in any model's
training set, used as the supervised-quality baseline. This script registers
those samples into the database: it inserts their single-channel image files
into ``image_metadata``, one ``benchmark_dataset`` row per ``(plate, well,
field)`` with its known ``t_label``, and the ``benchmark_dataset_member`` rows
linking them. No model and no inference are involved here -- scoring the
benchmark for a specific model is a separate step (``compute_benchmark.py``).

Two input modes:

``--manifest`` (preferred) reads a frozen dataset manifest built by
``scripts/build_dataset.py`` -- labels, channel files and image shapes are all
recorded in it, so registration touches neither the image files nor MongoDB and
is fully reproducible from a reviewed, committed artifact.

Positional globs are the fallback: ``(plate, well, field, channel)`` are parsed
from each filename with the same parser the API uses, shapes are read from the
files, and labels are resolved per ``(plate, well)`` from MongoDB. This needs the
image mount and the external ``tools`` package; the manifest mode needs neither.

Idempotent: ``benchmark_dataset`` is keyed on ``(plate, well, field)`` and
already-registered samples are skipped, so it is safe to rerun as the benchmark
set grows.
"""
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

from database.dblogger import DBLogger
from monitoring.config import MonitoringSettings


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
    from utils.filename_parser import extract_info_from_filename

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


def samples_from_manifest(manifest_path: str) -> tuple[dict[tuple[str, str, int], list[dict]],
                                                       dict[tuple[str, str, int], str]]:
    """Read a built dataset manifest into grouped files + per-sample labels.

    Accepts either the flat ``{"samples": [...]}`` form written by
    ``create_manifest_from_samples`` or a training manifest's ``train_samples``/
    ``val_samples`` lists. Entries must carry ``shape`` (built with
    ``shape_mode`` set), since nothing here reopens the image files.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get("samples")
    if entries is None:
        entries = manifest.get("train_samples", []) + manifest.get("val_samples", [])

    samples: dict[tuple[str, str, int], list[dict]] = {}
    labels: dict[tuple[str, str, int], str] = {}
    for entry in entries:
        key = (entry["plate"], entry["well"], int(entry["field"]))
        shape = entry.get("shape")
        if not shape:
            raise ValueError(
                f"Manifest entry {key} has no 'shape'. Rebuild the dataset with "
                "shape_mode=per_sample so image sizes are recorded "
                "(python scripts/build_dataset.py ... shape_mode=per_sample)."
            )
        labels[key] = entry["label"]
        for channel, file_name in entry["channel_files"].items():
            samples.setdefault(key, []).append({
                "plate": entry["plate"],
                "well": entry["well"],
                "field": int(entry["field"]),
                "channel": int(channel),
                "root_path": entry["root_path"],
                "file_name": file_name,
                "shape": (int(shape[0]), int(shape[1])),
            })
    return samples, labels


def collect_paths(globs: list[str], file: str | None) -> list[str]:
    """Expand glob patterns / read a path list file into a sorted path list."""
    paths: list[str] = []
    if file:
        with open(file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
    else:
        for pattern in globs:
            matched = glob(pattern)
            paths.extend(matched) if matched else paths.append(pattern)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description="Register benchmark samples into the database.")
    parser.add_argument("globs", nargs="*",
                        help="Fallback input: glob patterns (or explicit paths) of benchmark "
                             "image files, e.g. '/mnt/O/benchmark/*.jxl'. Needs the image mount "
                             "and MongoDB; prefer --manifest.")
    parser.add_argument("--manifest",
                        help="Path to a dataset manifest built by scripts/build_dataset.py "
                             "(e.g. data/benchmark_v1/dataset_manifest.json). Labels and image "
                             "shapes come from the manifest, so no image or MongoDB access is "
                             "needed.")
    parser.add_argument("--file", help="path to txt file containing list of benchmark image files")
    args = parser.parse_args()

    if not args.manifest and not args.globs and not args.file:
        parser.error("Provide --manifest (preferred), or glob patterns / --file.")

    settings = MonitoringSettings()
    if not settings.has_db_uri:
        print("ERROR: No database URI configured. Set MONITORING_DB_URI.")
        return

    # Manifest mode carries its own labels; glob mode resolves them from MongoDB.
    manifest_labels: dict[tuple[str, str, int], str] | None = None
    if args.manifest:
        print(f"Reading benchmark samples from manifest: {args.manifest}")
        samples, manifest_labels = samples_from_manifest(args.manifest)
    else:
        paths = collect_paths(args.globs, args.file)
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

        if manifest_labels is None:
            # Resolve labels once for all distinct wells.
            from utils.labels import resolve_labels_from_mongodb

            wells = sorted({(plate, well) for (plate, well, _field) in pending})
            well_labels = resolve_labels_from_mongodb(wells)
            print(f"MongoDB resolved {len(well_labels)} / {len(wells)} wells to a treatment.")
            labels = {key: well_labels.get((key[0], key[1])) for key in pending}
        else:
            labels = manifest_labels

        n_ok, n_skipped = 0, 0
        for key, files in pending.items():
            plate, well, field = key
            t_label = labels.get(key)
            if t_label is None:
                n_skipped += 1
                print(f"  skip {plate}/{well}/{field}: no label available yet")
                continue

            # Channels ascending (matches training/predict input channel order).
            files_sorted = sorted(files, key=lambda f: f["channel"])
            image_rows = []
            for f in files_sorted:
                # Manifest entries carry the shape; glob mode reads it off disk.
                shape_x, shape_y = f.get("shape") or _read_shape(
                    Path(f["root_path"]) / f["file_name"]
                )
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
