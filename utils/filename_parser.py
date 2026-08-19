import hashlib
import re

FILENAME_PATTERN = re.compile(
    r"(?P<plate>.+?)_(?P<well>[A-Z]\d+)_T(?P<time>\d+)F(?P<field>\d+)L(?P<layer>\d+)A(?P<action>\d+)Z(?P<z>\d+)C(?P<channel>\d+)\.(?:jxl|tif)$",
    re.IGNORECASE
)


def extract_info_from_filename(filename: str) -> dict:
    """Extract plate, well, field, channel from a microscopy filename.

    Falls back to empty/default values for .npy or unrecognised patterns.
    """
    if filename.endswith(".npy"):
        return {"plate": "", "well": "", "field": 1, "channel": 1, "filename": filename}

    match = FILENAME_PATTERN.match(filename)
    if not match:
        return {"plate": "", "well": "", "field": 1, "channel": 1, "filename": filename}

    info = match.groupdict()
    return {
        "plate": info["plate"],
        "well": info["well"],
        "field": int(info["field"]),
        "channel": int(info["channel"]),
        "filename": filename,
    }


def clean_image_metadata(image_metadata: list[dict]) -> list[tuple]:
    """Convert raw upload metadata into DB-ready tuples.

    Args:
        image_metadata: list of dicts with keys 'filename', 'shape', 'root_path' 
            (as built by the API from UploadFile objects).

    Returns:
        list of tuples: (plate, well, field, channel, root_path, file_name, shape_x, shape_y)
    """
    cleaned = []
    for metadata in image_metadata:
        parsed = extract_info_from_filename(metadata["filename"])
        row = (
            parsed["plate"],
            parsed["well"],
            parsed["field"],
            parsed["channel"],
            metadata["root_path"],
            metadata["filename"],
            metadata["shape"][0],
            metadata["shape"][1],
        )
        cleaned.append(row)
    return cleaned

def clean_tiles_metadata(tiles: list[dict], img_ids: list[int]) -> list[tuple]:
    """Convert tile metadata into DB-ready tuples.

    Args:
        tiles: list of tile dictionaries with keys 'x', 'y', 'col', 'row', 'crop_size'
        img_ids: list of image IDs

    Returns:
        list of tuples: (stack_hash, row_ind, col_ind, x_left, y_top, crop_size)
    """
    cleaned = []
    sorted_ids = sorted(img_ids)
    for tile in tiles:
        hash_input = str((sorted_ids, tile["x"], tile["y"], tile["crop_size"]))
        stack_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        row = (
            stack_hash,
            tile["row"],
            tile["col"],
            tile["x"],
            tile["y"],
            tile["crop_size"],
        )
        cleaned.append(row)

    return cleaned
