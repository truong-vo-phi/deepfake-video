import argparse
import csv
import random
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["landmark_path", "label", "label_name", "source_id", "aug", "stem"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def allocate_ids(ids: list[int], train_ratio: float, val_ratio: float):
    n = len(ids)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-report",
        type=str,
        default="D:/Project/deepfake-video/outputs/sdfvd2_landmarks/stage04_validation_report.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="D:/Project/deepfake-video/splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--include-low-quality", action="store_true")
    args = parser.parse_args()

    rows = read_csv(Path(args.input_report))
    if not rows:
        raise ValueError("Input report is empty.")

    if args.include_low_quality:
        usable = rows
    else:
        usable = [r for r in rows if str(r.get("quality_gate_pass", "")).lower() == "true"]

    if not usable:
        raise ValueError("No usable rows after quality filtering.")

    source_ids = sorted({to_int(r.get("source_id", -1), -1) for r in usable if to_int(r.get("source_id", -1), -1) > 0})
    if len(source_ids) < 3:
        raise ValueError("Not enough source_id to split.")

    rng = random.Random(args.seed)
    rng.shuffle(source_ids)

    train_ids, val_ids, test_ids = allocate_ids(source_ids, args.train_ratio, args.val_ratio)

    train_rows = [r for r in usable if to_int(r.get("source_id", -1), -1) in train_ids]
    val_rows = [r for r in usable if to_int(r.get("source_id", -1), -1) in val_ids]
    test_rows = [r for r in usable if to_int(r.get("source_id", -1), -1) in test_ids]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    write_csv(out_dir / "test.csv", test_rows)

    def summarize(name, split_rows):
        real = sum(1 for r in split_rows if str(r.get("label", "")) == "0")
        fake = sum(1 for r in split_rows if str(r.get("label", "")) == "1")
        ids = sorted({to_int(r.get("source_id", -1), -1) for r in split_rows})
        print(f"{name}: rows={len(split_rows)} real={real} fake={fake} source_ids={len(ids)}")

    print("Split completed.")
    print(f"Usable rows: {len(usable)} / Total rows: {len(rows)}")
    summarize("train", train_rows)
    summarize("val", val_rows)
    summarize("test", test_rows)
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
