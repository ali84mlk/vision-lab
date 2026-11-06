"""Train an image classifier (transfer learning) with swappable loss / optimizer / LR schedule.

Local smoke test (CPU, ~1 minute):
    python -m src.train --dataset oxford_pet --data-dir <folder> --subset 0.02 --epochs 2 \\
        --model mobilenet_v3_small --img-size 112 --optimizer adamw --scheduler warmup_cosine

Full run on a free Kaggle/Colab GPU: see notebooks/runner.ipynb.

One training STEP, the heart of deep learning (read the loop in `train_one_epoch`):
    1. forward pass:   logits = model(images)
    2. loss:           loss = loss_fn(logits, labels)               (src/losses.py)
    3. backward pass:  loss.backward()  -> gradient of the loss w.r.t. every weight
    4. update:         optimizer.step() -> change the weights using the gradients (src/optim.py)
    5. schedule:       scheduler.step() -> change the learning rate for the next step (src/schedulers.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import DATASETS, get_loaders
from src.losses import LOSS_NAMES, build_loss
from src.model import MODEL_NAMES, build_model, count_parameters, set_backbone_trainable
from src.optim import DEFAULT_LR, DEFAULT_WD, OPTIMIZER_NAMES, build_optimizer
from src.schedulers import SCHEDULER_NAMES, build_scheduler
from src.utils import default_dirs, default_num_workers, get_device, load_checkpoint, save_checkpoint, set_seed


# --------------------------------------------------------------------------- arguments
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    dirs = default_dirs()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--dataset", default="oxford_pet", choices=DATASETS)
    p.add_argument("--data-dir", default=dirs["data"], help="where the dataset lives / is downloaded to")
    p.add_argument("--subset", type=float, default=1.0, help="fraction of the data to use (0.02 = quick debug run)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=None, help="default: auto (0 on Windows, 2 elsewhere)")
    # model
    p.add_argument("--model", default="resnet18", choices=MODEL_NAMES)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-epochs", type=int, default=0, help="train only the new head for this many epochs first")
    # the three things this project is about
    p.add_argument("--loss", default="cross_entropy", choices=LOSS_NAMES)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    p.add_argument("--optimizer", default="adamw", choices=OPTIMIZER_NAMES)
    p.add_argument("--lr", type=float, default=None, help="default depends on the optimizer (src/optim.py)")
    p.add_argument("--weight-decay", type=float, default=None, help="default depends on the optimizer")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--scheduler", default="warmup_cosine", choices=SCHEDULER_NAMES)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    # run control
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="mixed precision (default: on if CUDA)")
    p.add_argument("--out-dir", default=dirs["out"], help="checkpoints/logs go to <out-dir>/<run-name>/")
    p.add_argument("--run-name", default="run")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/<run-name>/last.pt")
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True, help="write last.pt/best.pt every epoch")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    # Resolve the "None = pick a sensible default" arguments once, so they are saved in checkpoints.
    if args.lr is None:
        args.lr = DEFAULT_LR[args.optimizer]
    if args.weight_decay is None:
        args.weight_decay = DEFAULT_WD[args.optimizer]
    if args.num_workers is None:
        args.num_workers = default_num_workers()
    return args


# --------------------------------------------------------------------------- one epoch of training / evaluation
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, use_amp, progress=True):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False, disable=not progress):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)           # forget the previous step's gradients
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)                      # 1. forward pass
            loss = loss_fn(logits, labels)              # 2. loss
        scaler.scale(loss).backward()                   # 3. backward pass (scaled to avoid fp16 underflow)
        scaler.step(optimizer)                          # 4. update the weights
        scaler.update()
        scheduler.step()                                # 5. next learning rate

        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        seen += n
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp=False, progress=False):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="val", leave=False, disable=not progress):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        seen += n
    return total_loss / seen, correct / seen


def initial_loss_check(model, loader, loss_fn, device, num_classes):
    """Before training, an untrained head should give loss ~ ln(K). A big gap means a bug (see learn/01)."""
    model.eval()
    images, labels = next(iter(loader))
    with torch.no_grad():
        loss = loss_fn(model(images.to(device)), labels.to(device)).item()
    print(f"[sanity] initial loss {loss:.3f}  (expected about ln({num_classes}) = {math.log(num_classes):.3f})")


# --------------------------------------------------------------------------- the whole run
def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()
    use_amp = (device.type == "cuda") if args.amp is None else (args.amp and device.type == "cuda")
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} amp={use_amp} run_dir={run_dir}")

    train_loader, val_loader, num_classes, _ = get_loaders(
        args.dataset, args.data_dir, args.img_size, args.batch_size, args.subset, args.num_workers, args.seed
    )
    steps_per_epoch = len(train_loader)
    print(f"{args.dataset}: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val images, {num_classes} classes, {steps_per_epoch} steps/epoch")

    model = build_model(args.model, num_classes, args.pretrained, freeze_backbone=args.freeze_epochs > 0).to(device)

    loss_fn = build_loss(args.loss, args.label_smoothing, args.gamma)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs, steps_per_epoch, args.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_acc = 0, 0.0
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "metrics.csv"
    if args.resume and last_path.exists():
        ckpt = load_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        print(f"resumed from {last_path}: {start_epoch} epoch(s) already done, best acc so far {best_acc:.3f}")
    elif args.resume:
        print(f"--resume given but {last_path} does not exist: starting fresh")
    if start_epoch >= args.epochs:
        print("nothing to do: the checkpoint already finished all epochs (raise --epochs to continue)")

    # requires_grad is not stored in a checkpoint, so restore the frozen/unfrozen state for this epoch.
    if args.freeze_epochs > 0:
        set_backbone_trainable(model, args.model, trainable=start_epoch >= args.freeze_epochs)
    trainable, total = count_parameters(model)
    print(f"{args.model}: {trainable:,} trainable / {total:,} total parameters")
    if start_epoch == 0:
        initial_loss_check(model, val_loader, loss_fn, device, num_classes)

    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
    new_log = not (args.resume and log_path.exists())
    log_file = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    if new_log:
        writer.writeheader()

    tb = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb = SummaryWriter(str(run_dir / "tb"))
        except ImportError:
            print("tensorboard not installed: skipping (pip install tensorboard)")

    history = []
    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
                set_backbone_trainable(model, args.model, True)
                print(f"epoch {epoch + 1}: backbone unfrozen, now fine-tuning everything")

            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, device, use_amp, args.progress)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, use_amp)

            row = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 5), "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 5),
                "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1),
            }
            history.append(row)
            writer.writerow(row)
            log_file.flush()
            if tb:
                for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
                    tb.add_scalar(k, row[k], epoch + 1)
            print(f"epoch {epoch + 1:>3}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} | lr {row['lr']:.2e} | {row['seconds']}s")

            # Checkpoint EVERY epoch so a disconnect (Colab/Kaggle) loses at most one epoch.
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            if args.save_checkpoints:
                save_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_acc=best_acc, args=vars(args))
                if is_best:
                    save_checkpoint(best_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_acc=best_acc, args=vars(args))
    finally:
        log_file.close()
        if tb:
            tb.close()

    print(f"done. best val acc {best_acc:.3f}. checkpoints + metrics.csv in {run_dir}")
    return {"best_acc": best_acc, "history": history, "run_dir": str(run_dir)}


def main(argv: list[str] | None = None) -> dict:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
