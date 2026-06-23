import argparse
import sys
import subprocess
import time
from pathlib import Path

# Try to import PATHS and ROOT from src/runtime_paths.py
# First make sure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import PATHS, ROOT


def run_stage(command: list[str], stage_name: str, log_file) -> float:
    print("\n" + "=" * 80)
    print(f" STAGE: {stage_name}")
    print(f" Command: {' '.join(command)}")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f" STAGE: {stage_name}\n")
    log_file.write(f" Command: {' '.join(command)}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    start_time = time.time()
    
    # Run command and capture output in real time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    # Read output line by line as it is written
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

    process.wait()
    duration = time.time() - start_time

    print(f"\n[Stage Finished] Exit Code: {process.returncode} | Duration: {duration:.2f}s\n")
    log_file.write(f"\n[Stage Finished] Exit Code: {process.returncode} | Duration: {duration:.2f}s\n")
    log_file.flush()

    if process.returncode != 0:
        raise RuntimeError(f"Stage '{stage_name}' failed with exit code {process.returncode}")

    return duration


def main():
    parser = argparse.ArgumentParser(description="Automate Deepfake Detection Video Pipeline sequentially.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos processed during landmark extraction (useful for dry runs)."
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip the training and evaluation stages, only run preprocessing and dataset splits."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["all", "baseline", "multistream", "geophoto"],
        help="Which models to train and evaluate: 'all' (runs all 3), 'baseline', 'multistream', or 'geophoto'."
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing stages (clean, extract, validate, split), only run training/evaluation."
    )
    args = parser.parse_args()

    dataset_name = PATHS["dataset_name"]
    out_dir = PATHS["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline_run.log"

    print(f"Initializing pipeline run for active dataset: {dataset_name}")
    print(f"Log file: {log_path}\n")

    # Determine Python executable to run other scripts
    python_exe = sys.executable

    # Prepare stages list
    stages = []

    # Preprocessing
    if not args.skip_preprocessing:
        stages.append(([python_exe, "src/clean_dataset.py"], "Clean Dataset & Remove Duplicates"))

        extract_cmd = [python_exe, "src/extract_landmarks.py"]
        if args.limit is not None:
            extract_cmd.extend(["--limit", str(args.limit)])
        stages.append((extract_cmd, "Extract Face Landmarks via MediaPipe"))

        stages.append(([python_exe, "src/validate_landmarks.py"], "Validate & Filter Landmarks"))
        stages.append(([python_exe, "src/split_by_source_id.py"], "Split Dataset by Source/Speaker ID"))

    # Models
    if not args.skip_models:
        run_baseline = args.models in ["all", "baseline"]
        run_multistream = args.models in ["all", "multistream"]
        run_geophoto = args.models in ["all", "geophoto"]

        if run_baseline:
            stages.append(([python_exe, "src/train.py"], "Train ST-GCN Baseline"))
            stages.append(([python_exe, "src/evaluate.py"], "Evaluate ST-GCN Baseline"))

        if run_multistream:
            stages.append(([python_exe, "src/train_multistream.py"], "Train Multi-stream ST-GCN"))
            stages.append(([python_exe, "src/evaluate_multistream.py"], "Evaluate Multi-stream ST-GCN"))

        if run_geophoto:
            stages.append(([python_exe, "src/train_geophoto.py"], "Train GeoPhoto Cross-Attention ST-GCN"))
            stages.append(([python_exe, "src/evaluate_geophoto.py"], "Evaluate GeoPhoto Cross-Attention ST-GCN"))

    overall_start = time.time()
    durations = {}

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"PIPELINE AUTOMATION RUN FOR DATASET: {dataset_name}\n")
        log_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Python Environment: {python_exe}\n")
        log_file.write(f"Settings: limit={args.limit}, skip_preprocessing={args.skip_preprocessing}, skip_models={args.skip_models}, models={args.models}\n")
        log_file.flush()

        try:
            for cmd, stage_name in stages:
                duration = run_stage(cmd, stage_name, log_file)
                durations[stage_name] = duration
            
            total_duration = time.time() - overall_start
            
            print("\n" + "#" * 80)
            print(" PIPELINE RUN COMPLETED SUCCESSFULLY!")
            print(f" Total Duration: {total_duration:.2f}s")
            print("#" * 80)
            for name, dur in durations.items():
                print(f" - {name}: {dur:.2f}s")
            print("#" * 80 + "\n")

            log_file.write("\n" + "#" * 80 + "\n")
            log_file.write(" PIPELINE RUN COMPLETED SUCCESSFULLY!\n")
            log_file.write(f" Total Duration: {total_duration:.2f}s\n")
            log_file.write("#" * 80 + "\n")
            for name, dur in durations.items():
                log_file.write(f" - {name}: {dur:.2f}s\n")
            log_file.write("#" * 80 + "\n")

        except Exception as e:
            total_duration = time.time() - overall_start
            err_msg = f"\n[ERROR] Pipeline aborted due to failure: {e}\n"
            print(err_msg)
            log_file.write(err_msg)
            log_file.flush()
            sys.exit(1)


if __name__ == "__main__":
    main()
