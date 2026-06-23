import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from runtime_paths import PATHS


REAL_ORIGINAL_RE = re.compile(r"^v(\d+)\.mp4$", re.IGNORECASE)
REAL_AUG_RE = re.compile(r"^real_v(\d+)_aug_(\d+)\.mp4$", re.IGNORECASE)
FAKE_ORIGINAL_RE = re.compile(r"^vs(\d+)\.mp4$", re.IGNORECASE)
FAKE_AUG_RE = re.compile(r"^fake_vs(\d+)_aug_(\d+)(?: - Copy)?(?:-[A-Za-z0-9\-]+)?\.mp4$", re.IGNORECASE)


def stable_source_id(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def get_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"opened": False, "frame_count": 0, "fps": 0.0, "width": 0, "height": 0, "duration_sec": 0.0}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else 0.0
    return {"opened": True, "frame_count": frame_count, "fps": fps, "width": width, "height": height, "duration_sec": duration_sec}


def parse_name_sdfvd(filename: str, label_name: str):
    if label_name == "real":
        m = REAL_ORIGINAL_RE.match(filename)
        if m:
            return int(m.group(1)), -1, False, "ok"
        m = REAL_AUG_RE.match(filename)
        if m:
            return int(m.group(1)), int(m.group(2)), True, "ok"
    if label_name == "fake":
        m = FAKE_ORIGINAL_RE.match(filename)
        if m:
            return int(m.group(1)), -1, False, "ok"
        m = FAKE_AUG_RE.match(filename)
        if m:
            return int(m.group(1)), int(m.group(2)), True, "ok"
    return None, None, None, "unparsed"


def parse_name_biodeepav(filename: str):
    return stable_source_id(Path(filename).stem), -1, False, "ok"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_video_record(video_path: Path, label_name: str, label: int, dataset_name: str) -> dict:
    filename = video_path.name
    stem = video_path.stem
    
    source_id = 0
    aug = -1
    is_aug = False
    name_status = "ok"
    
    if dataset_name == "SDFVD2.0":
        source_id, aug, is_aug, name_status = parse_name_sdfvd(filename, label_name)
    elif dataset_name == "BioDeepAV":
        source_id, aug, is_aug, name_status = parse_name_biodeepav(filename)
    elif dataset_name == "Celeb":
        m = re.match(r"^id(\d+)_", stem, re.IGNORECASE)
        if m:
            source_id = int(m.group(1))
        else:
            source_id = stable_source_id("YouTube_" + stem)
    elif dataset_name == "FaceForensics++_C23":
        first_token = stem.split('_')[0]
        try:
            source_id = int(first_token)
        except ValueError:
            source_id = stable_source_id(first_token)
    elif dataset_name == "FakeAVCeleb":
        speaker_id = video_path.parent.name
        source_id = stable_source_id(speaker_id)
        
    return {
        "source_id": source_id,
        "aug": aug,
        "is_aug": is_aug,
        "name_status": name_status
    }


def get_unique_stem(video_path: Path, dataset_name: str) -> str:
    if dataset_name == "Celeb":
        if "YouTube-real" in video_path.parts or "YouTube-real" in video_path.as_posix():
            return f"youtube_real_{video_path.stem}"
        return video_path.stem
    elif dataset_name == "FaceForensics++_C23":
        return f"{video_path.parent.name}_{video_path.stem}"
    elif dataset_name == "FakeAVCeleb":
        # FakeAVCeleb path layout has: Category/Race/Gender/Speaker/Video
        # parents[3] is Category (e.g. RealVideo-RealAudio), parent is Speaker (e.g. id00076)
        try:
            category = video_path.parents[3].name
            speaker_id = video_path.parent.name
            return f"{category}_{speaker_id}_{video_path.stem}"
        except Exception:
            return f"{video_path.parent.name}_{video_path.stem}"
    else:
        return video_path.stem


def scan_folder(folder: Path, label_name: str, label: int, dataset_name: str, recursive: bool = False) -> list[dict]:
    records = []
    if not folder.exists():
        print(f"Warning: Folder {folder} does not exist. Skipping.")
        return records
        
    if recursive:
        mp4_files = sorted([p for p in folder.rglob("*.mp4") if p.is_file()])
    else:
        mp4_files = sorted([p for p in folder.glob("*.mp4") if p.is_file()])
        
    for video_path in tqdm(mp4_files, desc=f"Scanning {folder.name}"):
        parse_info = parse_video_record(video_path, label_name, label, dataset_name)
        info = get_video_info(video_path)
        unique_stem = get_unique_stem(video_path, dataset_name)
        records.append(
            {
                "video_path": video_path.resolve().as_posix(),
                "filename": video_path.name,
                "stem": unique_stem,
                "label_name": label_name,
                "label": label,
                "md5": get_md5(video_path),
                "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 4),
                "source_id": parse_info["source_id"],
                "source_key": unique_stem,
                "aug": parse_info["aug"],
                "is_aug": parse_info["is_aug"],
                "name_status": parse_info["name_status"],
                **info,
            }
        )
    return records


def scan_dfdc() -> list[dict]:
    train_dir = PATHS["train_dir"]
    metadata_path = PATHS["metadata_json"]
    if train_dir is None or metadata_path is None:
        raise ValueError("DFDC paths are not configured.")
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing DFDC train dir: {train_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing DFDC metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = []
    for filename, item in tqdm(sorted(metadata.items()), desc="Scanning DFDC metadata"):
        video_path = train_dir / filename
        label_name = str(item.get("label", "")).lower()
        if label_name not in {"real", "fake"}:
            name_status = "unsupported_label"
            label = -1
        else:
            name_status = "ok"
            label = 0 if label_name == "real" else 1

        original = item.get("original")
        source_key = original or filename
        info = get_video_info(video_path) if video_path.exists() else {
            "opened": False,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }
        records.append(
            {
                "video_path": video_path.resolve().as_posix() if video_path.exists() else str(video_path),
                "filename": filename,
                "stem": Path(filename).stem,
                "label_name": label_name,
                "label": label,
                "md5": get_md5(video_path) if video_path.exists() else "",
                "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 4) if video_path.exists() else 0.0,
                "source_id": stable_source_id(source_key),
                "source_key": source_key,
                "original": original or "",
                "aug": -1,
                "is_aug": False,
                "name_status": name_status if video_path.exists() else "missing_file",
                "metadata_split": item.get("split", ""),
                **info,
            }
        )
    return records


def build_clean_manifest(records: list[dict]):
    md5_groups = defaultdict(list)
    missing_records = []
    for r in records:
        if not r.get("md5"):
            missing_records.append(r)
        else:
            md5_groups[r["md5"]].append(r)

    clean_records, skipped = [], []
    for _, group in md5_groups.items():
        group = sorted(group, key=lambda r: (" - Copy" in r["filename"], len(r["filename"]), r["filename"]))
        keep = group[0]
        keep["duplicate_status"] = "kept"
        keep["duplicate_of"] = ""
        clean_records.append(keep)
        for dup in group[1:]:
            dup["duplicate_status"] = "skipped"
            dup["duplicate_of"] = keep["video_path"]
            skipped.append(dup)

    for missing in missing_records:
        missing["duplicate_status"] = "missing"
        missing["duplicate_of"] = ""
        skipped.append(missing)

    clean_records = sorted(clean_records, key=lambda r: (r["label"], str(r["source_id"]), str(r["aug"]), r["filename"]))
    return clean_records, skipped


def main():
    dataset_name = PATHS["dataset_name"]
    root = PATHS["dataset_root"]
    records = []

    if dataset_name == "SDFVD2.0":
        real_dir = root / "SDFVD2.0_real"
        fake_dir = root / "SDFVD2.0_fake"
        records.extend(scan_folder(real_dir, "real", 0, dataset_name))
        records.extend(scan_folder(fake_dir, "fake", 1, dataset_name))

    elif dataset_name == "BioDeepAV":
        real_dir = root / "real" / "videos"
        fake_dir = root / "fake" / "videos"
        records.extend(scan_folder(real_dir, "real", 0, dataset_name))
        records.extend(scan_folder(fake_dir, "fake", 1, dataset_name))

    elif dataset_name == "Celeb":
        records.extend(scan_folder(root / "Celeb-real", "real", 0, dataset_name))
        records.extend(scan_folder(root / "YouTube-real", "real", 0, dataset_name))
        records.extend(scan_folder(root / "Celeb-synthesis", "fake", 1, dataset_name))

    elif dataset_name == "FaceForensics++_C23":
        records.extend(scan_folder(root / "original", "real", 0, dataset_name))
        fake_folders = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"]
        for ff in fake_folders:
            records.extend(scan_folder(root / ff, "fake", 1, dataset_name))

    elif dataset_name == "FakeAVCeleb":
        # RealVideo-RealAudio is real (0)
        real_dir = root / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "RealVideo-RealAudio"
        records.extend(scan_folder(real_dir, "real", 0, dataset_name, recursive=True))
        # The other three folders are fake (1)
        fake_folders = ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio"]
        for ff in fake_folders:
            fake_dir = root / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / ff
            records.extend(scan_folder(fake_dir, "fake", 1, dataset_name, recursive=True))

    elif dataset_name == "DFDC":
        records = scan_dfdc()

    else:
        raise ValueError(f"Unsupported DATASET_NAME: {dataset_name}")

    write_csv(PATHS["stage01"], records)
    clean_records, skipped = build_clean_manifest(records)
    write_csv(PATHS["stage02_clean"], clean_records)
    write_csv(PATHS["stage02_skipped"], skipped)
    print(f"Dataset: {PATHS['dataset_name']}")
    print(f"Total: {len(records)} | Clean: {len(clean_records)} | Skipped duplicates/missing: {len(skipped)}")
    print(f"stage01: {PATHS['stage01']}")
    print(f"stage02_clean: {PATHS['stage02_clean']}")
    print(f"stage02_skipped: {PATHS['stage02_skipped']}")


if __name__ == "__main__":
    main()
