import csv
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from runtime_paths import PATHS


NUM_FRAMES = 32
MIN_DETECTED_RATIO = 0.5
SKIP_EXISTING = True
EXPECTED_LANDMARKS = 468
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_indices(frame_count: int, num_frames: int = 32):
    if frame_count <= 0:
        return []
    if num_frames <= 1:
        return [frame_count // 2]
    return np.linspace(0, frame_count - 1, num_frames).round().astype(int).tolist()


def detect_face(face_detection, rgb_frame, margin: float = 0.25):
    h, w = rgb_frame.shape[:2]
    result = face_detection.process(rgb_frame)
    if not result.detections:
        return None
    detection = max(result.detections, key=lambda x: x.score[0] if x.score else 0.0)
    score = float(detection.score[0])
    box = detection.location_data.relative_bounding_box
    x, y, bw, bh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, score


def extract_mesh(face_mesh, rgb_frame, bbox):
    x1, y1, x2, y2, _ = bbox
    crop = rgb_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    result = face_mesh.process(crop)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    crop_h, crop_w = crop.shape[:2]
    points = np.zeros((len(landmarks), 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        points[i, 0] = x1 + lm.x * crop_w
        points[i, 1] = y1 + lm.y * crop_h
        points[i, 2] = lm.z
    return points


def bbox_from_points(points: np.ndarray, width: int, height: int, margin: float = 0.12):
    x1 = float(np.nanmin(points[:, 0]))
    y1 = float(np.nanmin(points[:, 1]))
    x2 = float(np.nanmax(points[:, 0]))
    y2 = float(np.nanmax(points[:, 1]))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * margin
    my = bh * margin
    x1 = max(0, int(round(x1 - mx)))
    y1 = max(0, int(round(y1 - my)))
    x2 = min(width, int(round(x2 + mx)))
    y2 = min(height, int(round(y2 + my)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, 1.0


def normalize(points, bbox):
    x1, y1, x2, y2, score = bbox
    face_w, face_h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    scale = max(face_w, face_h)
    cx, cy = x1 + face_w / 2.0, y1 + face_h / 2.0
    out = np.zeros((points.shape[0], 4), dtype=np.float32)
    out[:, 0] = (points[:, 0] - cx) / scale
    out[:, 1] = (points[:, 1] - cy) / scale
    out[:, 2] = points[:, 2]
    out[:, 3] = score
    return out


def interpolate_missing(data, valid_mask):
    out = data.copy()
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return out
    all_idx = np.arange(data.shape[0])
    for node in range(data.shape[1]):
        for feat in range(3):
            values = out[valid_idx, node, feat]
            out[:, node, feat] = np.interp(all_idx, valid_idx, values)
    out[~valid_mask, :, 3] = 0.0
    return out


def add_motion(x):
    t, v, _ = x.shape
    delta = np.zeros((t, v, 3), dtype=np.float32)
    if t > 1:
        delta[1:] = x[1:, :, :3] - x[:-1, :, :3]
        delta[0] = delta[1]
    speed = np.linalg.norm(delta[:, :, :2], axis=-1, keepdims=True)
    return np.concatenate([x, delta, speed], axis=-1).astype(np.float32)


def ensure_model_asset(filename: str, url: str, label: str) -> Path:
    model_dir = PATHS["out_dir"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / filename
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print(f"Downloading MediaPipe {label} model to: {model_path}")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"MediaPipe Tasks backend requires {filename}, but automatic download failed. "
            f"Download it manually from {url} and save it to {model_path}. "
            f"Original error: {exc!r}"
        ) from exc
    return model_path


def ensure_face_landmarker_model() -> Path:
    return ensure_model_asset("face_landmarker.task", FACE_LANDMARKER_URL, "FaceLandmarker")


def ensure_face_detector_model() -> Path:
    return ensure_model_asset("blaze_face_short_range.tflite", FACE_DETECTOR_URL, "FaceDetector")


class LegacySolutionsExtractor:
    backend_name = "mediapipe_solutions"

    def __enter__(self):
        solutions = mp.solutions
        self.face_detection_ctx = solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh_ctx = solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detection = self.face_detection_ctx.__enter__()
        self.face_mesh = self.face_mesh_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.face_mesh_ctx.__exit__(exc_type, exc, tb)
        self.face_detection_ctx.__exit__(exc_type, exc, tb)

    def extract(self, rgb_frame):
        bbox = detect_face(self.face_detection, rgb_frame)
        if bbox is None:
            return None, None, "face_not_found"
        points = extract_mesh(self.face_mesh, rgb_frame, bbox)
        if points is None:
            return None, None, "mesh_not_found"
        return bbox, points, None


class TasksFaceLandmarkerExtractor:
    backend_name = "mediapipe_tasks_detector_landmarker"

    def __enter__(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        detector_model_path = ensure_face_detector_model()
        landmarker_model_path = ensure_face_landmarker_model()
        detector_options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(detector_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.1,
            min_suppression_threshold=0.3,
        )
        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(landmarker_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector_ctx = vision.FaceDetector.create_from_options(detector_options)
        self.landmarker_ctx = vision.FaceLandmarker.create_from_options(landmarker_options)
        self.detector = self.detector_ctx.__enter__()
        self.landmarker = self.landmarker_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.landmarker_ctx.__exit__(exc_type, exc, tb)
        self.detector_ctx.__exit__(exc_type, exc, tb)

    def _detect_bbox(self, image, width: int, height: int, margin: float = 0.25):
        result = self.detector.detect(image)
        if not result.detections:
            return None
        detection = max(result.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
        box = detection.bounding_box
        score = float(detection.categories[0].score) if detection.categories else 1.0
        x1 = int(box.origin_x)
        y1 = int(box.origin_y)
        x2 = x1 + int(box.width)
        y2 = y1 + int(box.height)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(bw * margin)
        my = int(bh * margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(width, x2 + mx)
        y2 = min(height, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2, score

    def extract(self, rgb_frame):
        h, w = rgb_frame.shape[:2]
        full_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        bbox = self._detect_bbox(full_image, width=w, height=h)
        if bbox is None:
            return None, None, "face_not_found"

        x1, y1, x2, y2, score = bbox
        crop = rgb_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, "empty_crop"

        crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        result = self.landmarker.detect(crop_image)
        if not result.face_landmarks:
            return None, None, "mesh_not_found"

        landmarks = result.face_landmarks[0]
        crop_h, crop_w = crop.shape[:2]
        points = np.zeros((len(landmarks), 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            points[i, 0] = x1 + lm.x * crop_w
            points[i, 1] = y1 + lm.y * crop_h
            points[i, 2] = lm.z

        refined_bbox = bbox_from_points(points, width=w, height=h)
        if refined_bbox is None:
            refined_bbox = bbox
        return (refined_bbox[0], refined_bbox[1], refined_bbox[2], refined_bbox[3], score), points, None


def create_extractor():
    if hasattr(mp, "solutions"):
        return LegacySolutionsExtractor()
    return TasksFaceLandmarkerExtractor()


def fit_landmark_count(feat: np.ndarray, expected: int = EXPECTED_LANDMARKS):
    if feat.shape[0] == expected:
        return feat
    fixed = np.full((expected, feat.shape[1]), np.nan, dtype=np.float32)
    n = min(expected, feat.shape[0])
    fixed[:n] = feat[:n]
    return fixed


def process_video(row, out_dir: Path, extractor):
    video_path = Path(row["video_path"])
    label_name = row["label_name"]
    stem = row["stem"]
    lm_dir = out_dir / "landmarks" / label_name
    lm_dir.mkdir(parents=True, exist_ok=True)
    out_npy = lm_dir / f"{stem}.npy"
    out_json = lm_dir / f"{stem}.json"

    if SKIP_EXISTING and out_npy.exists() and out_json.exists():
        try:
            with out_json.open("r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            cached_ratio = float(cached_meta.get("detected_ratio", 0.0))
            if cached_ratio >= MIN_DETECTED_RATIO:
                return {
                    **row,
                    "status": "ok",
                    "landmark_path": str(out_npy),
                    "meta_path": str(out_json),
                    "detected_frames": cached_meta.get("detected_frames", ""),
                    "failed_frames": cached_meta.get("failed_frames", ""),
                    "detected_ratio": cached_ratio,
                    "backend": cached_meta.get("backend", "cached"),
                }
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {**row, "status": "cannot_open", "landmark_path": "", "meta_path": "", "detected_frames": 0, "failed_frames": NUM_FRAMES, "detected_ratio": 0.0}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(frame_count, NUM_FRAMES)
    seq, valid, infos = [], [], []

    for frame_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            seq.append(np.full((EXPECTED_LANDMARKS, 4), np.nan, dtype=np.float32))
            valid.append(False)
            infos.append({"frame_idx": int(frame_idx), "detected": False, "reason": "read_failed"})
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bbox, points, reason = extractor.extract(rgb)
        if bbox is None or points is None:
            seq.append(np.full((EXPECTED_LANDMARKS, 4), np.nan, dtype=np.float32))
            valid.append(False)
            infos.append({"frame_idx": int(frame_idx), "detected": False, "reason": reason or "landmark_not_found"})
            continue

        feat = fit_landmark_count(normalize(points, bbox))
        seq.append(feat)
        valid.append(True)
        x1, y1, x2, y2, score = bbox
        infos.append({"frame_idx": int(frame_idx), "detected": True, "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)], "face_score": float(score)})

    cap.release()
    seq = np.stack(seq, axis=0).astype(np.float32)
    valid = np.asarray(valid, dtype=bool)
    detected = int(valid.sum())
    failed = int(len(valid) - detected)
    ratio = detected / max(1, len(valid))
    if detected > 0:
        seq = interpolate_missing(seq, valid)
    seq = add_motion(seq)
    np.save(out_npy, seq)

    meta = {
        "video_path": str(video_path),
        "label": int(row["label"]),
        "label_name": label_name,
        "stem": stem,
        "source_id": row.get("source_id"),
        "aug": row.get("aug"),
        "sampled_indices": idxs,
        "landmark_shape": list(seq.shape),
        "features": ["x_norm", "y_norm", "z", "detection_score", "dx", "dy", "dz", "speed"],
        "detected_frames": detected,
        "failed_frames": failed,
        "detected_ratio": ratio,
        "backend": extractor.backend_name,
        "frames": infos,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    status = "ok" if ratio >= MIN_DETECTED_RATIO else "low_detection_ratio"
    return {
        **row,
        "status": status,
        "landmark_path": str(out_npy),
        "meta_path": str(out_json),
        "detected_frames": detected,
        "failed_frames": failed,
        "detected_ratio": ratio,
        "num_landmarks": seq.shape[1],
        "feature_dim": seq.shape[2],
        "backend": extractor.backend_name,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MediaPipe Face landmarks from cleaned video dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of videos to process.")
    args = parser.parse_args()

    rows = read_csv(PATHS["stage02_clean"])
    if args.limit is not None:
        print(f"Limiting processing to the first {args.limit} videos.")
        rows = rows[:args.limit]

    out_dir = PATHS["out_dir"]
    results = []

    with create_extractor() as extractor:
        print(f"Landmark backend: {extractor.backend_name}")
        for row in tqdm(rows, desc=f"Extracting landmarks [{PATHS['dataset_name']}]"):
            try:
                results.append(process_video(row, out_dir, extractor))
            except Exception as exc:
                results.append({**row, "status": "exception", "error": repr(exc)})

    write_csv(PATHS["stage03_manifest"], results)
    failed_rows = [r for r in results if r.get("status") != "ok"]
    write_csv(PATHS["stage03_failed"], failed_rows)
    print(f"Dataset: {PATHS['dataset_name']}")
    print(f"Processed: {len(results)} | OK: {sum(1 for r in results if r.get('status') == 'ok')} | Non-OK: {len(failed_rows)}")
    print(f"stage03_manifest: {PATHS['stage03_manifest']}")
    print(f"stage03_failed: {PATHS['stage03_failed']}")


if __name__ == "__main__":
    main()



