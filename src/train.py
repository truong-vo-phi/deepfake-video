from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

from dataset import LandmarkDataset
from graph import build_adjacency, normalize_adjacency
from model_stgcn import STGCNClassifier


CONFIG = {
    "train_csv": "D:/Project/deepfake-video/splits/train.csv",
    "val_csv": "D:/Project/deepfake-video/splits/val.csv",
    "checkpoint_dir": "D:/Project/deepfake-video/checkpoints",
    "batch_size": 16,
    "num_workers": 0,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seeds": [42, 123, 2026],
    "early_stopping_patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "min_lr": 1e-6,
    "save_prefix": "stgcn_baseline",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, a_norm, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(x, a_norm)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, a_norm, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x, a_norm)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def count_labels(dataset):
    counts = {0: 0, 1: 0}
    for row in dataset.rows:
        counts[int(row["label"])] += 1
    return counts


def build_class_weight(train_ds, device):
    counts = count_labels(train_ds)
    total = counts[0] + counts[1]
    w0 = total / max(counts[0], 1)
    w1 = total / max(counts[1], 1)
    weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    return weights


def train_one_seed(seed: int, train_ds, val_ds, a_norm, c, v, ckpt_dir: Path):
    set_seed(seed)
    device = torch.device(CONFIG["device"])

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    model = STGCNClassifier(in_channels=c, num_classes=2).to(device)
    class_weights = build_class_weight(train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["min_lr"],
    )

    best_path = ckpt_dir / f"{CONFIG['save_prefix']}_seed{seed}_best.pt"
    log_path = ckpt_dir / f"{CONFIG['save_prefix']}_seed{seed}_log.json"

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, a_norm, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, a_norm, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        history.append(row)
        print(
            f"[seed={seed}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.2e}"
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "a_norm": a_norm.detach().cpu(),
                    "config": CONFIG,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "in_channels": c,
                    "num_nodes": v,
                },
                best_path,
            )
        else:
            no_improve += 1

        if no_improve >= CONFIG["early_stopping_patience"]:
            print(f"[seed={seed}] Early stopping at epoch {epoch}.")
            break

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(best_path),
        "log_path": str(log_path),
    }


def main():
    device = torch.device(CONFIG["device"])
    train_ds = LandmarkDataset(CONFIG["train_csv"])
    val_ds = LandmarkDataset(CONFIG["val_csv"])

    sample = train_ds[0]["x"]
    c, _, v = sample.shape
    a = build_adjacency(num_nodes=v)
    a_norm = torch.tensor(normalize_adjacency(a), dtype=torch.float32, device=device)

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for seed in CONFIG["seeds"]:
        result = train_one_seed(seed, train_ds, val_ds, a_norm, c, v, ckpt_dir)
        seed_results.append(result)

    accs = np.array([r["best_val_acc"] for r in seed_results], dtype=np.float32)
    losses = np.array([r["best_val_loss"] for r in seed_results], dtype=np.float32)
    summary = {
        "seeds": CONFIG["seeds"],
        "best_by_seed": seed_results,
        "val_acc_mean": float(accs.mean()),
        "val_acc_std": float(accs.std(ddof=0)),
        "val_loss_mean": float(losses.mean()),
        "val_loss_std": float(losses.std(ddof=0)),
    }

    summary_path = ckpt_dir / f"{CONFIG['save_prefix']}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    best_run = min(seed_results, key=lambda x: x["best_val_loss"])
    canonical_best = ckpt_dir / f"{CONFIG['save_prefix']}_best.pt"
    ckpt_bytes = Path(best_run["checkpoint_path"]).read_bytes()
    canonical_best.write_bytes(ckpt_bytes)

    print("Multi-seed training completed.")
    print(f"Val acc mean/std: {summary['val_acc_mean']:.4f}/{summary['val_acc_std']:.4f}")
    print(f"Val loss mean/std: {summary['val_loss_mean']:.4f}/{summary['val_loss_std']:.4f}")
    print(f"Best run (by val_loss): seed={best_run['seed']} epoch={best_run['best_epoch']}")
    print(f"Canonical checkpoint: {canonical_best}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
