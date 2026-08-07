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
from contextlib import ExitStack

import requests

BASE_URL = "http://127.0.0.1:8000"
ROOT_DIR = "O:\\BTSData\\MeasurementData\\MIG\\MIG-Exp03-CP-40X-bin1X1_20230831_143904\\MIG-Exp03-CP-40X-bin1X1"

def print_result(result: dict):
    """Pretty-print the prediction response."""
    print("\nWHOLE-IMAGE PREDICTION (majority vote)")
    print(f"  plate: {result['plate']}")
    print(f"  well: {result['well']}")
    print(f"  field: {result['field']}")
    print(f"  run_id: {result['run_id']}")
    print(f"  predicted_class: {result['predicted_class']}")
    print(f"  total_tiles: {result['total_tiles']}")
    print(f"  confidence:      {result['confidence']:.4f}")
    print(f"  vote_fraction:   {result['vote_fraction']:.4f}")


def main(): 

    file_names = [
        "MIG-Exp03-CP-40X-bin1X1_K09_T0001F006L01A04Z01C05.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K09_T0001F006L01A01Z01C02.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K09_T0001F006L01A01Z01C01.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K09_T0001F006L01A02Z01C03.jxl",
        "MIG-Exp03-CP-40X-bin1X1_K09_T0001F006L01A03Z01C04.jxl"
    ]
    # Sort by channel number (C01..C05) so channels are stacked in order
    file_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][-2:]))
    paths = [os.path.join(ROOT_DIR, file_name) for file_name in file_names]
    print(paths)

    # requests needs (field_name, file_object) tuples; field name must be "files"
    with ExitStack() as stack:
        handles = [stack.enter_context(open(p, "rb")) for p in paths]
        multipart = [("files", (file_name, h, f'image/{file_name.split(".")[-1]}')) for file_name, h in zip(file_names, handles)]
        results = requests.post(f"{BASE_URL}/predict", files=multipart, data={"root_path": ROOT_DIR})

    results.raise_for_status()
    print_result(results.json())


if __name__ == "__main__":
    main()
