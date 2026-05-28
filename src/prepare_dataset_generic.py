import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


def get_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "opened": False,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else 0.0
    return {
        "opened": True,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample_frame_indices(frame_count: int, num_frames: int = 32) -> list[int]:
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
    best = max(result.detections, key=lambda d: d.score[0] if d.score else 0.0)
    score = float(best.score[0])
    box = best.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, score


def extract_face_mesh(face_mesh, rgb_frame, bbox):
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


def normalize_landmarks(points, bbox):
    x1, y1, x2, y2, score = bbox
    face_w = max(1.0, float(x2 - x1))
    face_h = max(1.0, float(y2 - y1))
    scale = max(face_w, face_h)
    cx = x1 + face_w / 2.0
    cy = y1 + face_h / 2.0
    out = np.zeros((points.shape[0], 4), dtype=np.float32)
    out[:, 0] = (points[:, 0] - cx) / scale
    out[:, 1] = (points[:, 1] - cy) / scale
    out[:, 2] = points[:, 2]
    out[:, 3] = score
    return out


def interpolate_missing_frames(data, valid_mask):
    out = data.copy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return out
    all_indices = np.arange(data.shape[0])
    for node in range(data.shape[1]):
        for feature in range(3):
            values = out[valid_indices, node, feature]
            out[:, node, feature] = np.interp(all_indices, valid_indices, values)
    out[~valid_mask, :, 3] = 0.0
    return out


def add_motion_features(x):
    t, v, _ = x.shape
    delta = np.zeros((t, v, 3), dtype=np.float32)
    if t > 1:
        delta[1:] = x[1:, :, :3] - x[:-1, :, :3]
        delta[0] = delta[1]
    speed = np.linalg.norm(delta[:, :, :2], axis=-1, keepdims=True)
    return np.concatenate([x, delta, speed], axis=-1).astype(np.float32)


def parse_file(filename: str, source_id_regex: re.Pattern, aug_regex: re.Pattern | None):
    m = source_id_regex.search(filename)
    source_id = int(m.group(1)) if m else None
    aug = -1
    is_aug = False
    if aug_regex is not None:
        m_aug = aug_regex.search(filename)
        if m_aug:
            aug = int(m_aug.group(1))
            is_aug = True
    return source_id, aug, is_aug


def build_clean_manifest(records: list[dict]) -> tuple[list[dict], list[dict]]:
    md5_groups = defaultdict(list)
    for r in records:
        md5_groups[r["md5"]].append(r)
    clean = []
    dup = []
    for _, group in md5_groups.items():
        group = sorted(group, key=lambda r: (len(r["filename"]), r["filename"]))
        keep = group[0]
        keep["duplicate_status"] = "kept"
        keep["duplicate_of"] = ""
        clean.append(keep)
        for d in group[1:]:
            d["duplicate_status"] = "skipped"
            d["duplicate_of"] = keep["video_path"]
            dup.append(d)
    clean = sorted(
        clean,
        key=lambda r: (
            r["label"],
            r["source_id"] if r["source_id"] is not None else 10**9,
            r["aug"] if r["aug"] is not None else 10**9,
            r["filename"],
        ),
    )
    return clean, dup


def scan_stage12(
    dataset_root: Path,
    real_subdir: str,
    fake_subdir: str,
    source_id_pattern: str,
    aug_pattern: str,
    out_dir: Path,
):
    real_dir = dataset_root / real_subdir
    fake_dir = dataset_root / fake_subdir
    if not real_dir.exists():
        raise FileNotFoundError(f"Missing real dir: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Missing fake dir: {fake_dir}")

    source_id_regex = re.compile(source_id_pattern, re.IGNORECASE)
    aug_regex = re.compile(aug_pattern, re.IGNORECASE) if aug_pattern else None
    records = []

    for label_name, label, folder in [("real", 0, real_dir), ("fake", 1, fake_dir)]:
        for video_path in tqdm(sorted(folder.glob("*.mp4")), desc=f"Stage1 scan {label_name}"):
            source_id, aug, is_aug = parse_file(video_path.name, source_id_regex, aug_regex)
            info = get_video_info(video_path)
            records.append(
                {
                    "video_path": str(video_path),
                    "filename": video_path.name,
                    "stem": video_path.stem,
                    "label_name": label_name,
                    "label": label,
                    "source_id": source_id,
                    "aug": aug,
                    "is_aug": is_aug,
                    "name_status": "ok" if source_id is not None else "unparsed",
                    "md5": get_md5(video_path),
                    "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 4),
                    **info,
                }
            )

    write_csv(out_dir / "stage01_dataset_index.csv", records)
    clean, dup = build_clean_manifest(records)
    write_csv(out_dir / "stage02_clean_manifest.csv", clean)
    write_csv(out_dir / "stage02_skipped_duplicates.csv", dup)
    print("Stage 1+2 completed.")
    print(f"Scanned: {len(records)} | Clean: {len(clean)} | Duplicates: {len(dup)}")


def process_video(row, out_dir: Path, face_detection, face_mesh, num_frames=32, min_detected_ratio=0.5):
    video_path = Path(row["video_path"])
    label_name = row["label_name"]
    stem = row["stem"]
    landmark_dir = out_dir / "landmarks" / label_name
    landmark_dir.mkdir(parents=True, exist_ok=True)
    out_npy = landmark_dir / f"{stem}.npy"
    out_json = landmark_dir / f"{stem}.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {**row, "status": "cannot_open", "landmark_path": ""}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled = sample_frame_indices(frame_count, num_frames)
    expected = 468
    seq = []
    valid_mask = []
    frames_meta = []

    for frame_idx in sampled:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            seq.append(np.full((expected, 4), np.nan, dtype=np.float32))
            valid_mask.append(False)
            frames_meta.append({"frame_idx": int(frame_idx), "detected": False, "reason": "read_failed"})
            continue

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bbox = detect_face(face_detection, rgb)
        if bbox is None:
            seq.append(np.full((expected, 4), np.nan, dtype=np.float32))
            valid_mask.append(False)
            frames_meta.append({"frame_idx": int(frame_idx), "detected": False, "reason": "face_not_found"})
            continue

        pts = extract_face_mesh(face_mesh, rgb, bbox)
        if pts is None:
            seq.append(np.full((expected, 4), np.nan, dtype=np.float32))
            valid_mask.append(False)
            frames_meta.append({"frame_idx": int(frame_idx), "detected": False, "reason": "mesh_not_found"})
            continue

        feats = normalize_landmarks(pts, bbox)
        if feats.shape[0] != expected:
            fixed = np.full((expected, 4), np.nan, dtype=np.float32)
            n = min(expected, feats.shape[0])
            fixed[:n] = feats[:n]
            feats = fixed
        seq.append(feats)
        valid_mask.append(True)
        x1, y1, x2, y2, score = bbox
        frames_meta.append(
            {
                "frame_idx": int(frame_idx),
                "detected": True,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "face_score": float(score),
            }
        )

    cap.release()
    seq = np.stack(seq, axis=0).astype(np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    detected = int(valid_mask.sum())
    detected_ratio = detected / max(len(valid_mask), 1)
    if detected > 0:
        seq = interpolate_missing_frames(seq, valid_mask)
    seq = add_motion_features(seq)
    np.save(out_npy, seq)

    meta = {
        "video_path": str(video_path),
        "label": int(row["label"]),
        "label_name": label_name,
        "stem": stem,
        "source_id": row.get("source_id"),
        "aug": row.get("aug"),
        "sampled_indices": sampled,
        "landmark_shape": list(seq.shape),
        "features": ["x_norm", "y_norm", "z", "detection_score", "dx", "dy", "dz", "speed"],
        "detected_frames": detected,
        "failed_frames": int(len(valid_mask) - detected),
        "detected_ratio": detected_ratio,
        "frames": frames_meta,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    status = "ok" if detected_ratio >= min_detected_ratio else "low_detection_ratio"
    return {
        **row,
        "status": status,
        "landmark_path": str(out_npy),
        "meta_path": str(out_json),
        "detected_frames": detected,
        "failed_frames": int(len(valid_mask) - detected),
        "detected_ratio": detected_ratio,
        "num_landmarks": seq.shape[1],
        "feature_dim": seq.shape[2],
    }


def run_stage3(out_dir: Path, num_frames: int, max_videos: int | None):
    manifest = out_dir / "stage02_clean_manifest.csv"
    rows = read_csv(manifest)
    if max_videos is not None:
        rows = rows[:max_videos]

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    results = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for row in tqdm(rows, desc="Stage3 extract landmarks"):
            try:
                results.append(process_video(row, out_dir, face_detection, face_mesh, num_frames=num_frames))
            except Exception as e:
                results.append({**row, "status": "exception", "error": repr(e)})

    write_csv(out_dir / "stage03_landmark_manifest.csv", results)
    failed = [r for r in results if r.get("status") != "ok"]
    write_csv(out_dir / "stage03_failed_videos.csv", failed)
    print("Stage 3 completed.")
    print(f"Processed: {len(results)} | Failed/Low: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description="Generic Stage1->3 pipeline for deepfake datasets.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root folder of dataset.")
    parser.add_argument("--real-subdir", type=str, default="real", help="Real videos subdir under dataset root.")
    parser.add_argument("--fake-subdir", type=str, default="fake", help="Fake videos subdir under dataset root.")
    parser.add_argument(
        "--source-id-pattern",
        type=str,
        default=r"(\d+)",
        help="Regex with one capture group for source_id from filename.",
    )
    parser.add_argument(
        "--aug-pattern",
        type=str,
        default=r"aug[_\- ]?(\d+)",
        help="Regex with one capture group for augmentation index from filename. Empty string to disable.",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--num-frames", type=int, default=32, help="Sampled frames per video for stage3.")
    parser.add_argument("--max-videos", type=int, default=None, help="Debug limit for stage3.")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        choices=["all", "12", "3"],
        help="Run stage1+2, stage3, or all.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)

    if args.stages in ("all", "12"):
        scan_stage12(
            dataset_root=dataset_root,
            real_subdir=args.real_subdir,
            fake_subdir=args.fake_subdir,
            source_id_pattern=args.source_id_pattern,
            aug_pattern=args.aug_pattern.strip(),
            out_dir=out_dir,
        )
    if args.stages in ("all", "3"):
        run_stage3(out_dir=out_dir, num_frames=args.num_frames, max_videos=args.max_videos)


if __name__ == "__main__":
    main()
