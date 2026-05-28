from pathlib import Path
import csv
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GeoPhotoDataset(Dataset):
    def __init__(self, split_csv: str, photo_frames: int = 8, image_size: int = 112):
        self.split_csv = Path(split_csv)
        self.rows = self._read_csv(self.split_csv)
        self.photo_frames = photo_frames
        self.image_size = image_size

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def _load_meta(self, landmark_path: str):
        lp = Path(landmark_path)
        meta_path = lp.with_suffix(".json")
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _center_crop(self, frame):
        h, w = frame.shape[:2]
        s = min(h, w)
        y1 = (h - s) // 2
        x1 = (w - s) // 2
        return frame[y1:y1 + s, x1:x1 + s]

    def _crop_from_bbox(self, frame, bbox_xyxy):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        return frame[y1:y2, x1:x2]

    def _load_photo_sequence(self, meta: dict):
        video_path = meta["video_path"]
        frames_info = meta.get("frames", [])
        if len(frames_info) == 0:
            return np.zeros((self.photo_frames, 3, self.image_size, self.image_size), dtype=np.float32)

        indices = np.linspace(0, len(frames_info) - 1, self.photo_frames).round().astype(int).tolist()
        cap = cv2.VideoCapture(video_path)
        photos = []

        for idx in indices:
            item = frames_info[idx]
            frame_idx = int(item.get("frame_idx", 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                if item.get("detected", False) and "bbox_xyxy" in item:
                    crop = self._crop_from_bbox(frame_bgr, item["bbox_xyxy"])
                else:
                    crop = self._center_crop(frame_bgr)
                if crop.size == 0:
                    crop = self._center_crop(frame_bgr)
                crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            photos.append(np.transpose(rgb, (2, 0, 1)))

        cap.release()
        return np.stack(photos, axis=0).astype(np.float32)  # [Tp, 3, H, W]

    def __getitem__(self, idx):
        row = self.rows[idx]
        landmark_path = row["landmark_path"]
        x_geo = np.load(landmark_path).astype(np.float32)  # [T, V, C]
        x_geo = np.transpose(x_geo, (2, 0, 1))  # [C, T, V]

        meta = self._load_meta(landmark_path)
        if meta is None:
            x_photo = np.zeros((self.photo_frames, 3, self.image_size, self.image_size), dtype=np.float32)
        else:
            x_photo = self._load_photo_sequence(meta)

        y = int(row["label"])
        return {
            "x_geo": torch.from_numpy(x_geo),
            "x_photo": torch.from_numpy(x_photo),
            "y": torch.tensor(y, dtype=torch.long),
            "stem": row["stem"],
            "source_id": int(row["source_id"]),
        }
