from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

from dataset_geophoto import GeoPhotoDataset
from graph import build_adjacency, normalize_adjacency
from model_geophoto_stgcn import GeoPhotoSTGCN


CONFIG = {
    "train_csv": "D:/Project/deepfake-video/splits/train.csv",
    "val_csv": "D:/Project/deepfake-video/splits/val.csv",
    "checkpoint_dir": "D:/Project/deepfake-video/checkpoints",
    "batch_size": 8,
    "num_workers": 0,
    "lr": 3e-5,
    "weight_decay": 1e-4,
    "epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seeds": [42, 123, 2026],
    "early_stopping_patience": 8,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "min_lr": 1e-6,
    "save_prefix": "geophoto_stgcn",
    "photo_frames": 8,
    "image_size": 112,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    return w / w.mean()


def run_epoch(model, loader, a_norm, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in loader:
        x_geo = batch["x_geo"].to(device)
        x_photo = batch["x_photo"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(x_geo, x_photo, a_norm)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, a_norm, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in loader:
        x_geo = batch["x_geo"].to(device)
        x_photo = batch["x_photo"].to(device)
        y = batch["y"].to(device)
        logits = model(x_geo, x_photo, a_norm)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one_seed(seed, train_ds, val_ds, a_norm, c, v, ckpt_dir: Path):
    set_seed(seed)
    device = torch.device(CONFIG["device"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = GeoPhotoSTGCN(in_channels=c, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=build_class_weight(train_ds, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=CONFIG["scheduler_factor"], patience=CONFIG["scheduler_patience"], min_lr=CONFIG["min_lr"]
    )

    best_path = ckpt_dir / f"{CONFIG['save_prefix']}_seed{seed}_best.pt"
    log_path = ckpt_dir / f"{CONFIG['save_prefix']}_seed{seed}_log.json"
    best_val_loss, best_val_acc, best_epoch = float("inf"), -1.0, -1
    no_improve = 0
    history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, a_norm, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, a_norm, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": lr})
        print(
            f"[seed={seed}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = val_loss, val_acc, epoch
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
    train_ds = GeoPhotoDataset(CONFIG["train_csv"], photo_frames=CONFIG["photo_frames"], image_size=CONFIG["image_size"])
    val_ds = GeoPhotoDataset(CONFIG["val_csv"], photo_frames=CONFIG["photo_frames"], image_size=CONFIG["image_size"])
    sample = train_ds[0]["x_geo"]
    c, _, v = sample.shape
    a_norm = torch.tensor(normalize_adjacency(build_adjacency(num_nodes=v)), dtype=torch.float32, device=device)

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results = [train_one_seed(seed, train_ds, val_ds, a_norm, c, v, ckpt_dir) for seed in CONFIG["seeds"]]

    accs = np.array([r["best_val_acc"] for r in results], dtype=np.float32)
    losses = np.array([r["best_val_loss"] for r in results], dtype=np.float32)
    summary = {
        "seeds": CONFIG["seeds"],
        "best_by_seed": results,
        "val_acc_mean": float(accs.mean()),
        "val_acc_std": float(accs.std(ddof=0)),
        "val_loss_mean": float(losses.mean()),
        "val_loss_std": float(losses.std(ddof=0)),
    }
    summary_path = ckpt_dir / f"{CONFIG['save_prefix']}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    best_run = min(results, key=lambda x: x["best_val_loss"])
    canonical_best = ckpt_dir / f"{CONFIG['save_prefix']}_best.pt"
    canonical_best.write_bytes(Path(best_run["checkpoint_path"]).read_bytes())

    print("GeoPhoto multi-seed training completed.")
    print(f"Val acc mean/std: {summary['val_acc_mean']:.4f}/{summary['val_acc_std']:.4f}")
    print(f"Val loss mean/std: {summary['val_loss_mean']:.4f}/{summary['val_loss_std']:.4f}")
    print(f"Best run (by val_loss): seed={best_run['seed']} epoch={best_run['best_epoch']}")
    print(f"Canonical checkpoint: {canonical_best}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
