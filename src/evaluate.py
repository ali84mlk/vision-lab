"""Evaluate a trained run: overall accuracy, hardest classes, and a confusion matrix image.

    python -m src.evaluate --run-dir checkpoints/run --data-dir <folder>

Training reports one number (accuracy). Evaluation asks WHERE the model fails: which classes get confused
with which. That is how you decide what to improve (more data for a class? different augmentation?).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import get_loaders
from src.model import build_model
from src.utils import default_dirs, default_num_workers, get_device, load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        preds.append(model(images.to(device)).argmax(1).cpu())
        targets.append(labels)
    return torch.cat(preds), torch.cat(targets)


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """cm[true, predicted] = how many examples of class `true` were predicted as `predicted`."""
    return torch.bincount(targets * num_classes + preds, minlength=num_classes**2).reshape(num_classes, num_classes)


def main(argv: list[str] | None = None) -> dict:
    dirs = default_dirs()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=str(Path(dirs["out"]) / "run"))
    p.add_argument("--data-dir", default=dirs["data"])
    p.add_argument("--checkpoint", default="best.pt", help="best.pt or last.pt")
    p.add_argument("--subset", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    device = get_device()
    ckpt_path = Path(args.run_dir) / args.checkpoint
    train_args = torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]  # settings used for training

    _, val_loader, num_classes, class_names = get_loaders(
        train_args["dataset"], args.data_dir, train_args["img_size"], args.batch_size, args.subset,
        default_num_workers() if args.num_workers is None else args.num_workers, train_args["seed"],
    )
    model = build_model(train_args["model"], num_classes, pretrained=False).to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)

    preds, targets = collect_predictions(model, val_loader, device)
    cm = confusion_matrix(preds, targets, num_classes)
    acc = (preds == targets).float().mean().item()
    per_class = cm.diag().float() / cm.sum(1).clamp(min=1).float()

    print(f"{ckpt_path}  ->  accuracy {acc:.2%} on {len(targets)} validation images")
    print("hardest classes:")
    for i in per_class.argsort()[:5].tolist():
        confused_with = cm[i].clone()
        confused_with[i] = 0
        j = int(confused_with.argmax())
        print(f"  {class_names[i]:<28} {per_class[i]:.0%} correct   (most often mistaken for: {class_names[j]})")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm.numpy(), cmap="Blues")
        ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(f"Confusion matrix (accuracy {acc:.1%})")
        fig.colorbar(im)
        out = Path(args.run_dir) / "confusion_matrix.png"
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        print(f"saved {out}")
    return {"accuracy": acc, "per_class": per_class.tolist()}


if __name__ == "__main__":
    main()
