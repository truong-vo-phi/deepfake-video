from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class LandmarkDataset(Dataset):
    def __init__(self, split_csv: str):
        self.split_csv = Path(split_csv)
        self.rows = self._read_csv(self.split_csv)

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = np.load(row["landmark_path"]).astype(np.float32)  # [T, V, C]
        x = np.transpose(x, (2, 0, 1))  # [C, T, V]
        y = int(row["label"])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.long),
            "stem": row["stem"],
            "source_id": int(row["source_id"]),
        }
