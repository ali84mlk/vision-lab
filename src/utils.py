"""Small helpers shared by every script: environment detection, seeding, device, checkpoints.

Why this file exists: the SAME code has to run on your laptop (CPU), on Kaggle and on Colab.
The only things that differ are *where files live* and *whether a GPU exists*, so we isolate
those two questions here and keep the rest of the project environment-agnostic.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- environment
def detect_env() -> str:
    """Return 'kaggle', 'colab' or 'local'."""
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def default_dirs(env: str | None = None) -> dict[str, str]:
    """Where to put datasets and training outputs (checkpoints, logs).

    - Kaggle: only /kaggle/working is kept after a run, so outputs go there.
    - Colab: the VM is wiped when it disconnects, so outputs go to Google Drive when it is mounted.
    - Local: plain folders next to the project.
    """
    env = env or detect_env()
    if env == "kaggle":
        return {"data": "/kaggle/working/data", "out": "/kaggle/working/outputs"}
    if env == "colab":
        drive = "/content/drive/MyDrive/vision-lab"
        out = f"{drive}/outputs" if os.path.exists("/content/drive/MyDrive") else "/content/outputs"
        return {"data": "/content/data", "out": out}
    return {"data": "data", "out": "checkpoints"}


def default_num_workers(env: str | None = None) -> int:
    """Data-loading worker processes. Kaggle/Colab have ~2-4 CPU cores; on Windows, 0 avoids spawn overhead."""
    env = env or detect_env()
    if env == "local":
        return 0 if os.name == "nt" else 2
    return 2


# --------------------------------------------------------------------------- reproducibility / device
def set_seed(seed: int = 0) -> None:
    """Make runs repeatable (same seed -> same init, same shuffling, same augmentation draws)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, *, model, optimizer, scheduler, scaler, epoch: int, best_acc: float, args: dict) -> None:
    """Save EVERYTHING needed to continue training exactly where we stopped.

    Saving only the model weights is not enough to *resume*: Adam keeps running averages (its
    'memory'), the LR scheduler knows which epoch it is on, and the AMP scaler has a loss scale.
    Without them, a resumed run would behave differently from an uninterrupted one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "args": args,
        },
        path,
    )


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict:
    """Load a checkpoint into the given objects (optimizer/scheduler/scaler are optional, e.g. for evaluation)."""
    # weights_only=False because we also store plain Python data (args dict); only load files you trust.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
