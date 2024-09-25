import numpy as np
from pathlib import Path
import binascii
import struct

RAW_DATA_PATH = "./raw/"
DATA_PATH = "data"
HEX_GROUP_SIZE = 4
SCALE_X = 41.4738
SCALE_Y = 21.3872
VARIANCE_THRESHOLD = 0.005


def hex_to_int(hex_string: str) -> int:
    return struct.unpack("<h", binascii.unhexlify(hex_string))[0]

def process_hex_data(hex_data: str) -> np.ndarray:
    # Convert hexadecimal data to integers and reshape
    data = np.array(
        [
            hex_to_int(hex_data[i : i + HEX_GROUP_SIZE])
            for i in range(0, len(hex_data), HEX_GROUP_SIZE)
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    data[:, 0] /= SCALE_X
    data[:, 1] /= SCALE_Y
    return data


def should_process_data(data: np.ndarray) -> bool:  # null sampling cancellation
    return data[:, 0].var() >= VARIANCE_THRESHOLD


def save_processed_data(processed_data: np.ndarray, path: Path, counter: int):
    np.save(path / f"{counter}.npy", processed_data)


def dump2npy(raw_data_path: str, data_path: str):
    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    for category in Path(raw_data_path).iterdir():
        category_dir = data_dir / category.name
        category_dir.mkdir(exist_ok=True)

        counter = 0
        for file in category.glob("*.txt"):
            raw_data = file.read_text().replace("\n", "")[8:]
            if len(raw_data) % (HEX_GROUP_SIZE * 2) == 0:
                processed_data = process_hex_data(raw_data)
                if should_process_data(processed_data):
                    counter += 1
                    save_processed_data(processed_data, category_dir, counter)


try:
    dump2npy(RAW_DATA_PATH, DATA_PATH)
except Exception as e:
    print(f"An error occurred: {e}")
