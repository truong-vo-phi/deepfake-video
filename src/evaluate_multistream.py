import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LandmarkDataset
from model_multistream_stgcn import MultiStreamSTGCNClassifier


CONFIG = {
    "test_csv": "D:/Project/deepfake-video/splits/test.csv",
    "checkpoint_path": "D:/Project/deepfake-video/checkpoints/stgcn_multistream_best.pt",
    "out_csv": "D:/Project/deepfake-video/results/multistream_test_predictions.csv",
    "batch_size": 32,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def save_predictions(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stem", "source_id", "label", "pred", "prob_fake"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    device = torch.device(CONFIG["device"])
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    model = MultiStreamSTGCNClassifier(in_channels=ckpt["in_channels"], num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    a_norm = ckpt["a_norm"].to(device)

    ds = LandmarkDataset(CONFIG["test_csv"])
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

    metrics = compute_binary_metrics(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64))
    save_predictions(Path(CONFIG["out_csv"]), pred_rows)

    print("Multi-stream test metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"Predictions: {CONFIG['out_csv']}")


if __name__ == "__main__":
    main()
