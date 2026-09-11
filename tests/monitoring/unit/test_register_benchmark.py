"""Unit tests for monitoring/register_benchmark.py.

Covers the manifest-driven path (samples_from_manifest) and the glob fallback
(group_samples, collect_paths). No database, image files, or MongoDB needed.
"""
from __future__ import annotations

import pytest

from monitoring.register_benchmark import (
    collect_paths,
    group_samples,
    samples_from_manifest,
)

# ─────────────────────────────────────────────────────────────────────────────
# samples_from_manifest
# ─────────────────────────────────────────────────────────────────────────────

PLATE = "MIG-Exp03-CP-40X-bin1X1"


class TestSamplesFromManifest:
    def test_flat_samples(self, flat_manifest):
        samples, labels = samples_from_manifest(str(flat_manifest))
        assert len(samples) == 3
        assert len(labels) == 3

        # Each sample has 5 channel files
        for key, files in samples.items():
            assert len(files) == 5
            plate, well, field = key
            assert plate == PLATE
            assert well in ("K07", "K08", "K09")
            assert field == 1

    def test_labels_match_samples(self, flat_manifest):
        samples, labels = samples_from_manifest(str(flat_manifest))
        assert set(samples.keys()) == set(labels.keys())
        assert labels[(PLATE, "K07", 1)] == "ClassA"
        assert labels[(PLATE, "K08", 1)] == "ClassB"
        assert labels[(PLATE, "K09", 1)] == "ClassC"

    def test_channel_files_carry_shape(self, flat_manifest):
        samples, _ = samples_from_manifest(str(flat_manifest))
        for files in samples.values():
            for f in files:
                assert f["shape"] == (2048, 2048)

    def test_train_val_manifest(self, train_val_manifest):
        """train_samples + val_samples are concatenated."""
        samples, labels = samples_from_manifest(str(train_val_manifest))
        assert len(samples) == 3  # 2 train + 1 val
        assert labels[(PLATE, "K07", 1)] == "ClassA"
        assert labels[(PLATE, "K09", 1)] == "ClassC"

    def test_missing_shape_raises(self, manifest_missing_shape):
        with pytest.raises(ValueError, match="no 'shape'"):
            samples_from_manifest(str(manifest_missing_shape))

    def test_field_coerced_to_int(self, flat_manifest):
        samples, _ = samples_from_manifest(str(flat_manifest))
        for (plate, well, field) in samples:
            assert isinstance(field, int)

    def test_channel_coerced_to_int(self, flat_manifest):
        samples, _ = samples_from_manifest(str(flat_manifest))
        for files in samples.values():
            for f in files:
                assert isinstance(f["channel"], int)


# ─────────────────────────────────────────────────────────────────────────────
# group_samples (glob fallback)
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupSamples:
    def test_basic_grouping(self):
        """Files for the same (plate, well, field) are grouped together."""
        names = [
            f"{PLATE}_K07_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(1, 6)
        ]
        paths = [f"/data/images/{n}" for n in names]
        samples = group_samples(paths)
        assert len(samples) == 1
        key = (PLATE, "K07", 1)
        assert key in samples
        assert len(samples[key]) == 5

    def test_multiple_wells_grouped_separately(self):
        names_a = [f"{PLATE}_K07_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(1, 3)]
        names_b = [f"{PLATE}_K08_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(1, 3)]
        paths = [f"/data/{n}" for n in names_a + names_b]
        samples = group_samples(paths)
        assert len(samples) == 2
        assert (PLATE, "K07", 1) in samples
        assert (PLATE, "K08", 1) in samples

    def test_unparseable_skipped(self):
        """Files that don't match the naming pattern are skipped."""
        paths = ["/data/random_file.npy", "/data/no_pattern.jxl"]
        samples = group_samples(paths)
        assert len(samples) == 0

    def test_mixed_parseable_and_unparseable(self):
        good = f"{PLATE}_K07_T0001F001L01A01Z01C01.jxl"
        bad = "random_file.npy"
        samples = group_samples([f"/data/{good}", f"/data/{bad}"])
        assert len(samples) == 1
        assert len(samples[(PLATE, "K07", 1)]) == 1

    def test_file_dict_fields(self):
        paths = [f"/data/images/{PLATE}_K07_T0001F001L01A01Z01C01.jxl"]
        samples = group_samples(paths)
        f = samples[(PLATE, "K07", 1)][0]
        assert f["plate"] == PLATE
        assert f["well"] == "K07"
        assert f["field"] == 1
        assert f["channel"] == 1
        # Path.parent normalizes separators on Windows; compare with Path
        from pathlib import Path
        assert f["root_path"] == str(Path("/data/images"))
        assert f["file_name"] == f"{PLATE}_K07_T0001F001L01A01Z01C01.jxl"


# ─────────────────────────────────────────────────────────────────────────────
# collect_paths
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectPaths:
    def test_explicit_paths_returned_sorted(self, tmp_path):
        f1 = tmp_path / "b.jxl"
        f2 = tmp_path / "a.jxl"
        f1.write_text("")
        f2.write_text("")
        paths = collect_paths([str(f1), str(f2)])
        assert paths == [str(f2), str(f1)]  # sorted

    def test_dedup(self, tmp_path):
        f1 = tmp_path / "a.jxl"
        f1.write_text("")
        paths = collect_paths([str(f1), str(f1)])
        assert len(paths) == 1

    def test_glob_expansion(self, tmp_path):
        f1 = tmp_path / "a.jxl"
        f2 = tmp_path / "b.jxl"
        f1.write_text("")
        f2.write_text("")
        paths = collect_paths([str(tmp_path / "*.jxl")])
        assert len(paths) == 2

    def test_non_matching_glob_kept_as_literal(self):
        """A glob that matches nothing is kept as a literal path (for error messages)."""
        paths = collect_paths(["/nonexistent/*.jxl"])
        assert paths == ["/nonexistent/*.jxl"]
