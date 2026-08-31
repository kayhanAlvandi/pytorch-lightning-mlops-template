"""Ground-truth label resolution for (plate, well) samples.

A label is a property of the physical sample ``(plate, well)`` -- not of a
directory, a model, or a training run -- so resolution takes a well list and
returns a mapping, nothing more. Discovering *which* wells exist is a separate
concern (see ``src/sample_selection.scan_directory``), which keeps this module
free of filesystem access and lets every caller scan the disk only once.

Lives in ``utils/`` because it is shared across images: training/builder code
imports it from ``src/``, and ``monitoring/backfill_labels.py`` imports it in the
lightweight monitoring image. It therefore stays torch-free (stdlib + pandas).

The MongoDB connection string is hardcoded inside the external ``tools.loading``
module (shared team server); ``tools`` is imported lazily so that callers which
only need dummy labels -- or only the ``limit_wells_per_label`` helper -- do not
require the package to be installed.
"""
from __future__ import annotations

Well = tuple[str, str]

DEFAULT_DUMMY_CLASSES: tuple[str, ...] = ("ClassA", "ClassB", "ClassC", "ClassD")


def resolve_labels_from_mongodb(wells: list[Well], collection: str = "tags") -> dict[Well, str]:
    """Look up treatments for (plate, well) pairs in MongoDB.

    Sends a ``Plate``/``Well`` DataFrame through ``tools.loading.getCategories``
    and reads the resulting ``Treatment`` column.

    Only wells that resolved to a usable treatment are returned: a missing
    ``Treatment`` column yields ``{}`` (with a warning), and ``None``/``""``/NaN
    treatments are dropped. Letting those through would put a non-string into
    the label mapping, which later breaks ``sorted()`` in ``LabelEncoder.fit``
    with a confusing ``TypeError`` far from the real cause (bad upstream data).
    """
    import pandas as pd
    from tools.loading import getCategories

    if not wells:
        return {}

    df = pd.DataFrame({"Plate": [w[0] for w in wells], "Well": [w[1] for w in wells]})
    df.drop_duplicates(inplace=True)

    df_labels = getCategories(df, collection=collection)
    if "Treatment" not in df_labels.columns:
        print("WARNING: getCategories returned no 'Treatment' column -- no labels available.")
        return {}

    labels: dict[Well, str] = {}
    for plate, well, treatment in zip(df_labels["Plate"], df_labels["Well"], df_labels["Treatment"]):
        if treatment is None or pd.isna(treatment) or str(treatment) == "":
            continue
        labels[(plate, well)] = str(treatment)
    return labels


def resolve_labels_dummy(
    wells: list[Well],
    class_names: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
) -> dict[Well, str]:
    """Assign deterministic pseudo-random labels, for tests/offline runs.

    Wells are sorted before assignment so the mapping depends only on the well
    set and the seed, never on filesystem iteration order.
    """
    import numpy as np

    class_names = tuple(class_names) if class_names else DEFAULT_DUMMY_CLASSES
    rng = np.random.RandomState(seed)
    return {well: class_names[rng.randint(0, len(class_names))] for well in sorted(wells)}


def resolve_labels_for_wells(
    wells: list[Well] | set[Well],
    source: str = "mongodb",
    class_names: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
    collection: str = "tags",
) -> dict[Well, str]:
    """Resolve ``{(plate, well): treatment}`` from the configured label source.

    Args:
        wells: the wells to look up.
        source: ``"mongodb"`` (real treatments) or ``"dummy"`` (deterministic
            pseudo-random labels for tests/offline use).
        class_names: dummy source only -- the label vocabulary to draw from.
        seed: dummy source only -- makes the assignment reproducible.
        collection: MongoDB source only -- collection passed to ``getCategories``.
    """
    wells = sorted(wells)
    if source == "mongodb":
        return resolve_labels_from_mongodb(wells, collection=collection)
    if source == "dummy":
        return resolve_labels_dummy(wells, class_names=class_names, seed=seed)
    raise ValueError(f"Unknown label source: {source!r} (expected 'mongodb' or 'dummy')")


def limit_wells_per_label(
    labels: dict[Well, str],
    n_per_label: int | None,
    verbose: bool = False,
) -> dict[Well, str]:
    """Keep only the first N wells per label class, for a consistent subset.

    Wells are sorted within each label before truncating, so the selected
    subset is stable across runs and machines. ``None`` means no limit.
    """
    if n_per_label is None:
        return labels

    from collections import defaultdict

    wells_by_label: dict[str, list[Well]] = defaultdict(list)
    for well, label in labels.items():
        wells_by_label[label].append(well)

    limited: dict[Well, str] = {}
    for label, wells in wells_by_label.items():
        for well in sorted(wells)[:n_per_label]:
            limited[well] = label

    print(f"Limited to {n_per_label} wells per label: {len(labels)} -> {len(limited)} wells")
    if verbose:
        for label in sorted(wells_by_label):
            selected = sorted(wells_by_label[label])[:n_per_label]
            print(f"  {label}: {[f'{plate}/{well}' for plate, well in selected]}")

    return limited
