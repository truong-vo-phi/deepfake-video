import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LandmarkDataset
from metrics_binary import binary_auc, binary_eer, roc_curve_from_scores
from model_multistream_stgcn import MultiStreamSTGCNClassifier
from runtime_paths import PATHS, ROOT
import matplotlib.pyplot as plt


DATASET_TAG = PATHS["dataset_name"].lower().replace(".", "").replace("-", "_")

CONFIG = {
    "test_csv": str(PATHS["test_csv"]),
    "checkpoint_path": str(ROOT / f"checkpoints/{DATASET_TAG}_stgcn_multistream_best.pt"),
    "out_csv": str(ROOT / f"results/{DATASET_TAG}_multistream_test_predictions.csv"),
    "batch_size": 32,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": binary_auc(y_true, y_score),
        "eer": binary_eer(y_true, y_score),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_predictions(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stem", "source_id", "label", "pred", "prob_fake"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Multi-stream ST-GCN on a test set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test CSV.")
    parser.add_argument("--out-csv", type=str, default=None, help="Path to output predictions CSV.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or CONFIG["checkpoint_path"]
    test_csv = args.test_csv or CONFIG["test_csv"]
    out_csv = args.out_csv or CONFIG["out_csv"]

    print(f"\nRunning Multi-stream ST-GCN Evaluation...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Test CSV:   {test_csv}")

    device = torch.device(CONFIG["device"])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiStreamSTGCNClassifier(in_channels=ckpt["in_channels"], num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    a_norm = ckpt["a_norm"].to(device)

    ds = LandmarkDataset(test_csv)
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    y_true, y_pred, pred_rows = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        logits = model(x, a_norm)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()

        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())
        stems = batch["stem"]
        source_ids = batch["source_id"].cpu().numpy().tolist()
        for i in range(len(stems)):
            pred_rows.append(
                {
                    "stem": stems[i],
                    "source_id": int(source_ids[i]),
                    "label": int(y[i]),
                    "pred": int(preds[i]),
                    "prob_fake": float(probs[i]),
                }
            )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_score = np.asarray([r["prob_fake"] for r in pred_rows], dtype=np.float64)
    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    save_predictions(Path(out_csv), pred_rows)

    # Plot and save ROC Curve
    fpr, tpr = roc_curve_from_scores(y_true, y_score)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer_val = metrics["eer"]
    auc_val = metrics["auc"]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random (AUC = 0.5000)")
    plt.scatter(fpr[eer_idx], tpr[eer_idx], color="red", zorder=5, s=50, label=f"EER = {eer_val:.4f}")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Multi-stream ST-GCN")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    out_img = Path(out_csv).with_suffix(".png")
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nMulti-stream test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Predictions saved to: {out_csv}")
    print(f"ROC Curve Plot saved to: {out_img}\n")


if __name__ == "__main__":
    main()

