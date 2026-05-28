import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def to_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def validate_npy(path: Path, expected_t: int, expected_v: int, expected_c: int) -> dict:
    if not path.exists():
        return {
            "exists": False,
            "shape_ok": False,
            "has_nan": True,
            "all_zero": True,
            "mean_abs_xyz": 0.0,
            "mean_speed": 0.0,
            "validity": "missing_file",
        }

    try:
        arr = np.load(path)
    except Exception:
        return {
            "exists": True,
            "shape_ok": False,
            "has_nan": True,
            "all_zero": True,
            "mean_abs_xyz": 0.0,
            "mean_speed": 0.0,
            "validity": "cannot_load",
        }

    shape_ok = arr.ndim == 3 and arr.shape == (expected_t, expected_v, expected_c)
    has_nan = bool(np.isnan(arr).any())
    all_zero = bool(np.allclose(arr, 0.0))

    if arr.ndim == 3 and arr.shape[2] >= 8:
        mean_abs_xyz = float(np.mean(np.abs(arr[:, :, :3])))
        mean_speed = float(np.mean(np.abs(arr[:, :, 7])))
    else:
        mean_abs_xyz = 0.0
        mean_speed = 0.0

    validity = "ok"
    if not shape_ok:
        validity = "bad_shape"
    elif has_nan:
        validity = "contains_nan"
    elif all_zero:
        validity = "all_zero"

    return {
        "exists": True,
        "shape_ok": shape_ok,
        "has_nan": has_nan,
        "all_zero": all_zero,
        "mean_abs_xyz": round(mean_abs_xyz, 6),
        "mean_speed": round(mean_speed, 6),
        "validity": validity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="D:/Project/deepfake-video/outputs/sdfvd2_landmarks/stage03_landmark_manifest.csv",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="D:/Project/deepfake-video/outputs/sdfvd2_landmarks/stage04_validation_report.csv",
    )
    parser.add_argument("--expected-t", type=int, default=32)
    parser.add_argument("--expected-v", type=int, default=468)
    parser.add_argument("--expected-c", type=int, default=8)
    parser.add_argument(
        "--min-detected-ratio",
        type=float,
        default=0.5,
        help="Ngưỡng detection ratio để pass quality gate.",
    )
    args = parser.parse_args()

    rows = read_csv(Path(args.manifest))
    report_rows = []

    for row in rows:
        landmark_path = Path(row.get("landmark_path", ""))
        npy_check = validate_npy(
            landmark_path,
            expected_t=args.expected_t,
            expected_v=args.expected_v,
            expected_c=args.expected_c,
        )

        detected_ratio = to_float(row.get("detected_ratio", "0"), default=0.0)
        failed_frames = to_int(row.get("failed_frames", "0"), default=0)
        status_stage03 = row.get("status", "")

        gate_pass = (
            npy_check["validity"] == "ok"
            and detected_ratio >= args.min_detected_ratio
            and status_stage03 == "ok"
        )

        report_rows.append(
            {
                "video_path": row.get("video_path", ""),
                "filename": row.get("filename", ""),
                "stem": row.get("stem", ""),
                "label": row.get("label", ""),
                "label_name": row.get("label_name", ""),
                "source_id": row.get("source_id", ""),
                "aug": row.get("aug", ""),
                "status_stage03": status_stage03,
                "detected_ratio": detected_ratio,
                "failed_frames": failed_frames,
                "landmark_path": str(landmark_path),
                "expected_shape": f"({args.expected_t},{args.expected_v},{args.expected_c})",
                "shape_ok": npy_check["shape_ok"],
                "has_nan": npy_check["has_nan"],
                "all_zero": npy_check["all_zero"],
                "mean_abs_xyz": npy_check["mean_abs_xyz"],
                "mean_speed": npy_check["mean_speed"],
                "validity": npy_check["validity"],
                "quality_gate_pass": gate_pass,
            }
        )

    write_csv(Path(args.out_report), report_rows)

    total = len(report_rows)
    pass_count = sum(1 for r in report_rows if r["quality_gate_pass"])
    fail_count = total - pass_count
    print("Validation completed.")
    print(f"Total: {total}")
    print(f"Pass: {pass_count}")
    print(f"Fail: {fail_count}")
    print(f"Report: {args.out_report}")


if __name__ == "__main__":
    main()
