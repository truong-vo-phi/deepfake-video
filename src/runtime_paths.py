from pathlib import Path


# Change only this value to switch dataset.
# Supported: "SDFVD2.0", "BioDeepAV", "Celeb", "FaceForensics++_C23", "FakeAVCeleb", "DFDC"
DATASET_NAME = "FaceForensics++_C23"


ROOT = Path("D:/Project/deepfake-video")


def _dataset_paths(name: str):
    if name == "SDFVD2.0":
        dataset_root = ROOT / "SDFVD2.0"
        real_dir = dataset_root / "SDFVD2.0_real"
        fake_dir = dataset_root / "SDFVD2.0_fake"
        train_dir = None
        metadata_json = None
        out_dir = ROOT / "outputs" / "sdfvd2_landmarks"
        splits_dir = ROOT / "splits"
    elif name == "BioDeepAV":
        dataset_root = ROOT / "BioDeepAV"
        real_dir = dataset_root / "real" / "videos"
        fake_dir = dataset_root / "fake" / "videos"
        train_dir = None
        metadata_json = None
        out_dir = ROOT / "outputs" / "biodeepav_landmarks"
        splits_dir = ROOT / "splits_biodeepav"
    elif name == "Celeb":
        dataset_root = ROOT / "Celeb"
        real_dir = dataset_root # Will scan Celeb-real and YouTube-real in clean_dataset.py
        fake_dir = dataset_root / "Celeb-synthesis"
        train_dir = None
        metadata_json = None
        out_dir = ROOT / "outputs" / "celeb_landmarks"
        splits_dir = ROOT / "splits_celeb"
    elif name == "FaceForensics++_C23":
        dataset_root = ROOT / "FaceForensics++_C23"
        real_dir = dataset_root / "original"
        fake_dir = dataset_root # Will scan Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter, DeepFakeDetection in clean_dataset.py
        train_dir = None
        metadata_json = None
        out_dir = ROOT / "outputs" / "faceforensics_landmarks"
        splits_dir = ROOT / "splits_faceforensics"
    elif name == "FakeAVCeleb":
        dataset_root = ROOT / "FakeAVCeleb"
        real_dir = dataset_root / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "RealVideo-RealAudio"
        fake_dir = dataset_root / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" # Will scan FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio in clean_dataset.py
        train_dir = None
        metadata_json = None
        out_dir = ROOT / "outputs" / "fakeavceleb_landmarks"
        splits_dir = ROOT / "splits_fakeavceleb"
    elif name == "DFDC":
        dataset_root = ROOT / "deepfake-detection-challenge"
        real_dir = None
        fake_dir = None
        train_dir = dataset_root / "train_sample_videos"
        metadata_json = train_dir / "metadata.json"
        out_dir = ROOT / "outputs" / "dfdc_landmarks"
        splits_dir = ROOT / "splits_dfdc"
    else:
        raise ValueError(f"Unsupported DATASET_NAME: {name}")

    return {
        "dataset_name": name,
        "dataset_root": dataset_root,
        "real_dir": real_dir,
        "fake_dir": fake_dir,
        "train_dir": train_dir,
        "metadata_json": metadata_json,
        "out_dir": out_dir,
        "splits_dir": splits_dir,
        "stage01": out_dir / "stage01_dataset_index.csv",
        "stage02_clean": out_dir / "stage02_clean_manifest.csv",
        "stage02_skipped": out_dir / "stage02_skipped_duplicates.csv",
        "stage03_manifest": out_dir / "stage03_landmark_manifest.csv",
        "stage03_failed": out_dir / "stage03_failed_videos.csv",
        "stage04_report": out_dir / "stage04_validation_report.csv",
        "train_csv": splits_dir / "train.csv",
        "val_csv": splits_dir / "val.csv",
        "test_csv": splits_dir / "test.csv",
    }


PATHS = _dataset_paths(DATASET_NAME)
