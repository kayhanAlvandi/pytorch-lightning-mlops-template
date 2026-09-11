"""Backfill production ground-truth labels (t_label) from MongoDB.

Production predictions are logged before their true label is known. This job
finds production ``image_prediction`` rows still missing ``t_label``, looks up
each ``(plate, well)``'s treatment in the shared MongoDB (via the external
``tools.loading.getCategories``, exactly as training does in
``src/dataset.py``), and writes the label back onto every matching production
row -- image-level and its child tile-level rows.

The MongoDB lookup itself lives in ``utils/labels.py``, shared with training's
label providers (``src/dataset.py``) and the dataset builder, so there is one
implementation of "wells -> treatments" rather than one per caller.

Label backfill is global by well: a MongoDB label is a property of the physical
sample ``(plate, well)``, not of which model predicted it, so a resolved label
updates all matching production rows regardless of ``run_id``. Reference and
benchmark rows are left untouched (they already carry known labels), and an
existing non-null ``t_label`` is never overwritten.

Idempotent and safe to rerun: only NULL labels are filled, so a second run with
no newly-available labels is a no-op. Needs the external ``tools`` package
(installed at container start from the mounted ``/app/tools``) and network
access to the shared MongoDB server; the connection string is hardcoded inside
``tools.loading``.
"""
from __future__ import annotations

from database.dblogger import DBLogger
from monitoring.config import MonitoringSettings
from utils.labels import resolve_labels_from_mongodb as resolve_labels


def main():
    settings = MonitoringSettings()

    if not settings.has_db_uri:
        print("ERROR: No database URI configured. Set MONITORING_DB_URI.")
        return

    db_logger = DBLogger(db_uri=settings.db_uri)
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        wells = db_logger.fetch_unlabeled_wells() or []
        print(f"{len(wells)} (plate, well) pairs have unlabeled production predictions.")
        if not wells:
            print("Nothing to backfill.")
            return

        labels = resolve_labels(wells)
        print(f"MongoDB resolved {len(labels)} / {len(wells)} wells to a treatment.")
        if not labels:
            print("No labels available yet -- nothing to write.")
            return

        n_wells, n_image_rows, n_tile_rows = 0, 0, 0
        for (plate, well), treatment in labels.items():
            n_img, n_tile = db_logger.update_t_label(plate, well, treatment)
            if n_img:
                n_wells += 1
                n_image_rows += n_img
                n_tile_rows += n_tile
                print(f"  {plate}/{well} -> {treatment} ({n_img} image rows, {n_tile} tile rows)")

        print(f"Done. {len(wells)} wells checked, {len(labels)} resolved, "
              f"{n_wells} newly labeled: {n_image_rows} image rows / {n_tile_rows} tile rows updated.")
    finally:
        db_logger.close()


if __name__ == "__main__":
    main()
