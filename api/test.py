"""Simple test client for the prediction API.

Usage:
    # 1. Quick smoke test with a random synthetic image (no real data needed):
    python api/test.py

    # 2. Send a pre-stacked .npy file of shape (C, H, W):
    python api/test.py path/to/image.npy

    # 3. Send one file per channel (stacked in the order given):
    python api/test.py C1.tif C2.tif C3.tif C4.tif C5.tif

Make sure the server is running first:
    uvicorn api.main:app --reload --port 8000
"""
import os
import requests


BASE_URL = "http://localhost:8000"
ROOT_DIR = "O:\\BTSData\\MeasurementData\\MIG\\MIG-Exp03-CP-40X-bin1X1_20230831_143904\\MIG-Exp03-CP-40X-bin1X1"

def print_result(result: dict):
    """Pretty-print the prediction response."""
    img = result["image_prediction"]
    print("\nWHOLE-IMAGE PREDICTION (majority vote)")
    print(f"  predicted_class: {img['predicted_class']}")
    print(f"  confidence:      {img['confidence']:.4f}")
    print(f"  vote_fraction:   {img['vote_fraction']:.4f}")
    print(f"  vote_counts:     {img['vote_counts']}")
    print(f"\n  image_shape:     {result['image_shape']}")
    print(f"  num_tiles:       {result['num_tiles']}")
    print(f"  tile_grid:       {result['tile_grid']}")

    print("\nFIRST FEW TILE PREDICTIONS:")
    for tile in result["tile_predictions"][:3]:
        print(f"  (row={tile['row']}, col={tile['col']}) "
              f"-> {tile['predicted_class']} ({tile['confidence']:.3f})")


def main(): 

    file_names = [
        "MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A04Z01C05.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C02.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C01.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A02Z01C03.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A03Z01C04.jxl"
    ]
    paths = [os.path.join(ROOT_DIR, file_name) for file_name in file_names]
    # Sort by channel number (C01..C05) so channels are stacked in order
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][-2:]))
    print(paths)

    # requests needs (field_name, file_object) tuples; field name must be "files"
    handles = [open(p, "rb") for p in paths]
    try:
        multipart = [("files", h) for h in handles]
        results = requests.post(f"{BASE_URL}/predict", files=multipart)
    finally:
        for h in handles:
            h.close()

    results.raise_for_status()
    print_result(results.json())


if __name__ == "__main__":
    main()
