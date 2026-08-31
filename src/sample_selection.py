"""Dataset sample selection: scan a directory once, then filter in memory.

This is the selection layer shared by training and by the standalone dataset
builder (``scripts/build_dataset.py``). It deliberately contains no ``Dataset``
class, no transforms, no tiling and no ``LabelEncoder``: choosing *which*
samples make up a dataset needs none of those, and keeping them out means this
module is torch-free and usable anywhere (labels stay plain strings throughout).

The pipeline is always the same four steps, in this order:

    index   = scan_directory(root_dir)                    # the only disk scan
    wells   = wells_from_index(index, exclude=, include=)
    labels  = resolve_labels_for_wells(wells, ...)         # utils/labels.py
    samples = build_samples(index, labels, channels, ...)

Scanning once matters: the previous split (label provider globbed to discover
wells, then the train dataset globbed again, then the val dataset globbed a
third time) walked the image directory three times per run, which dominates
setup time on a network drive.

``field`` is normalised to ``int`` here -- matching ``utils/filename_parser``,
the ``image_metadata.field INTEGER`` column and ``register_benchmark`` -- so
that sample keys compare equal to keys read back out of the database.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path

from utils.filename_parser import FILENAME_PATTERN

SUPPORTED_EXTENSIONS = (".jxl", ".tif")

Well = tuple[str, str]
SampleKey = tuple[str, str, int]
FileIndex = dict[SampleKey, dict[int, Path]]


def scan_directory(root_dir: str | Path) -> FileIndex:
    """Scan ``root_dir`` once and index every parseable image file.

    Returns ``{(plate, well, field): {channel: path}}``. Files whose names do
    not match the microscopy filename pattern are ignored.
    """
    root_path = Path(root_dir)
    index: FileIndex = defaultdict(dict)

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in root_path.glob(f"*{ext}"):
            match = FILENAME_PATTERN.match(file_path.name)
            if not match:
                continue
            info = match.groupdict()
            key = (info["plate"], info["well"], int(info["field"]))
            index[key][int(info["channel"])] = file_path

    return dict(index)


def wells_from_index(
    index: FileIndex,
    exclude_wells: list[Well] | None = None,
    include_wells: list[Well] | None = None,
) -> set[Well]:
    """Return the (plate, well) pairs present in the index, after filtering.

    ``exclude_wells`` drops wells (e.g. known-corrupted acquisitions);
    ``include_wells`` is a whitelist keeping only those wells (used to pin a
    benchmark set to specific wells). Passing both is rejected rather than
    silently resolved, since their intent conflicts.
    """
    if exclude_wells and include_wells:
        raise ValueError("Pass either exclude_wells or include_wells, not both.")

    wells = {(plate, well) for plate, well, _field in index}

    if include_wells:
        include_set = {tuple(w) for w in include_wells}
        missing = include_set - wells
        if missing:
            print(f"WARNING: {len(missing)} whitelisted wells not found on disk: "
                  f"{sorted(missing)}")
        return wells & include_set

    if exclude_wells:
        exclude_set = {tuple(w) for w in exclude_wells}
        wells -= exclude_set
        print(f"Excluded {len(exclude_set)} wells from dataset")

    return wells


def read_image_shape(file_path: str | Path) -> tuple[int, int]:
    """Read ``(height, width)`` from an image header, without decoding pixels."""
    import pillow_jxl  # noqa: F401  registers JXL support with PIL
    from PIL import Image

    with Image.open(file_path) as im:
        width, height = im.size
    return height, width


def build_samples(
    index: FileIndex,
    labels_dict: dict[Well, str],
    channels: list[int],
    max_samples_per_label: int | None = None,
    seed: int = 42,
    shape_mode: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Turn a file index into labelled samples, one per (plate, well, field).

    Keeps only samples whose well has a label and which have **every** requested
    channel present, then optionally balances to ``max_samples_per_label``.

    Args:
        index: output of ``scan_directory``.
        labels_dict: ``{(plate, well): label}``; samples of unlabelled wells are
            dropped. Restricting this mapping is how the caller restricts the
            dataset (e.g. one call per train/val split).
        channels: required channel numbers; samples missing any are dropped.
        max_samples_per_label: balance to at most N samples per label class.
        seed: seeds the balancing RNG so the selection is reproducible.
        shape_mode: ``None`` (don't read shapes), ``"first"`` (read one image and
            apply its size to every sample -- fast, assumes a uniform
            acquisition) or ``"per_sample"`` (read each sample -- correct for
            heterogeneous sets, needed when the shape is persisted per image).
        verbose: print the selected samples.

    Returns:
        Sample dicts with keys ``plate``, ``well``, ``field``, ``label``,
        ``channel_files`` (``{channel: Path}``) and, when ``shape_mode`` is set,
        ``image_size`` (``(height, width)``).
    """
    channels = sorted(channels)

    samples = []
    for (plate, well, field) in sorted(index):
        if (plate, well) not in labels_dict:
            continue
        channel_files = index[(plate, well, field)]
        if not all(ch in channel_files for ch in channels):
            continue
        samples.append({
            "plate": plate,
            "well": well,
            "field": field,
            "label": labels_dict[(plate, well)],
            "channel_files": {ch: channel_files[ch] for ch in channels},
            "image_size": None,
        })

    if max_samples_per_label is not None:
        samples = _balance_samples(samples, max_samples_per_label, seed=seed, verbose=verbose)

    if shape_mode is not None and samples:
        _attach_shapes(samples, channels[0], shape_mode)

    return samples


def _balance_samples(
    samples: list[dict],
    max_samples_per_label: int,
    seed: int = 42,
    verbose: bool = False,
) -> list[dict]:
    """Randomly keep at most N samples per label, reproducibly for a given seed."""
    rng = random.Random(seed)

    samples_by_label: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        samples_by_label[sample["label"]].append(sample)

    balanced = []
    for label in sorted(samples_by_label):
        label_samples = list(samples_by_label[label])
        rng.shuffle(label_samples)
        selected = label_samples[:max_samples_per_label]
        balanced.extend(selected)
        print(f"  {label}: {len(label_samples)} -> {len(selected)} samples")

    print(f"Balanced to {max_samples_per_label} samples/label: "
          f"{len(samples)} -> {len(balanced)}")

    if verbose:
        print("Selected samples:")
        for sample in balanced:
            print(f"  {sample['plate']}/{sample['well']}/F{sample['field']} -> {sample['label']}")

    return balanced


def _attach_shapes(samples: list[dict], channel: int, shape_mode: str) -> None:
    """Fill each sample's ``image_size`` in place, per ``shape_mode``."""
    if shape_mode == "first":
        shape = read_image_shape(samples[0]["channel_files"][channel])
        for sample in samples:
            sample["image_size"] = shape
    elif shape_mode == "per_sample":
        for sample in samples:
            sample["image_size"] = read_image_shape(sample["channel_files"][channel])
    else:
        raise ValueError(
            f"Unknown shape_mode: {shape_mode!r} (expected 'first' or 'per_sample')"
        )


@dataclass
class DatasetSpec:
    """The short form of a dataset definition: filters, and nothing else.

    Everything here answers "which samples belong to this dataset". Anything
    about *how* they are consumed (batch size, transforms, tiling, splits) is
    deliberately absent, which is what makes a spec reusable outside training.
    """

    root_dir: str
    channels: list[int]
    use_mongodb: bool = True
    exclude_wells: list[Well] | None = None
    include_wells: list[Well] | None = None
    max_wells_per_label: int | None = None
    max_samples_per_label: int | None = None
    seed: int = 42
    verbose: bool = False
    name: str = "dataset"
    dummy_class_names: list[str] | None = dataclass_field(default=None)

    @property
    def label_source(self) -> str:
        return "mongodb" if self.use_mongodb else "dummy"

    def to_dict(self) -> dict:
        """Plain-dict form, used for the config hash in the dataset version."""
        return asdict(self)

    @classmethod
    def from_datamodule_cfg(cls, cfg, name: str = "dataset") -> "DatasetSpec":
        """Build a spec from a training datamodule config (hydra or plain dict).

        Reads only the selection keys, so the same YAML that trained a model can
        rebuild that model's dataset definition outside training.
        """
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(cfg):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass

        dataset_cfg = cfg["dataset"]
        exclude = cfg.get("exclude_wells")
        return cls(
            root_dir=dataset_cfg["root_dir"],
            channels=list(dataset_cfg["channels"]),
            use_mongodb=cfg.get("use_mongodb", True),
            exclude_wells=[tuple(w) for w in exclude] if exclude else None,
            max_wells_per_label=cfg.get("max_wells_per_label"),
            max_samples_per_label=dataset_cfg.get("max_samples_per_label"),
            verbose=dataset_cfg.get("verbose", False),
            name=name,
        )


def build_dataset(spec: DatasetSpec, shape_mode: str | None = "per_sample") -> list[dict]:
    """Run the full selection pipeline for a spec: scan -> wells -> labels -> samples.

    The convenience entry point used by the standalone builder. Training calls
    the individual steps instead, because it has to fit the label encoder and
    split by well in between.
    """
    from utils.labels import limit_wells_per_label, resolve_labels_for_wells

    index = scan_directory(spec.root_dir)
    print(f"Scanned {spec.root_dir}: {len(index)} (plate, well, field) groups found.")

    wells = wells_from_index(index, spec.exclude_wells, spec.include_wells)
    print(f"{len(wells)} wells after filtering.")

    labels = resolve_labels_for_wells(
        wells,
        source=spec.label_source,
        class_names=spec.dummy_class_names,
        seed=spec.seed,
    )
    print(f"{len(labels)} / {len(wells)} wells resolved to a label "
          f"(source={spec.label_source}).")
    if not labels:
        return []

    labels = limit_wells_per_label(labels, spec.max_wells_per_label, verbose=spec.verbose)

    samples = build_samples(
        index,
        labels,
        spec.channels,
        max_samples_per_label=spec.max_samples_per_label,
        seed=spec.seed,
        shape_mode=shape_mode,
        verbose=spec.verbose,
    )
    print(f"{len(samples)} samples selected.")
    return samples
