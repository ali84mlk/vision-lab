"""Compare optimizers x LR schedules x losses on the real task, with the SAME data, model and seed.

    python -m experiments.compare --subset 0.05 --epochs 3 \\
        --optimizers adamw sgd_momentum --schedulers constant warmup_cosine --losses cross_entropy

Every combination is trained with src.train, and results are appended to <out-dir>/summary.csv after EACH run, so
if Kaggle/Colab disconnects you can re-run the same command and finished runs are skipped.

How to read the results: change ONE thing at a time when you want to learn something. Differences below ~1% are
often noise from a single seed: use `--seeds 0 1 2` to get a mean +/- std before drawing conclusions.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from src import train as train_module
from src.losses import LOSS_NAMES
from src.optim import OPTIMIZER_NAMES
from src.schedulers import SCHEDULER_NAMES
from src.utils import default_dirs


def parse_args(argv=None):
    dirs = default_dirs()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optimizers", nargs="+", default=["sgd_momentum", "adamw"], choices=OPTIMIZER_NAMES)
    p.add_argument("--schedulers", nargs="+", default=["constant", "warmup_cosine"], choices=SCHEDULER_NAMES)
    p.add_argument("--losses", nargs="+", default=["cross_entropy"], choices=LOSS_NAMES)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    # passed through to src.train
    p.add_argument("--dataset", default="oxford_pet")
    p.add_argument("--data-dir", default=dirs["data"])
    p.add_argument("--model", default="resnet18")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--subset", type=float, default=1.0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--keep-checkpoints", action="store_true", help="by default no .pt files are written (saves disk)")
    return p.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    a = parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    done = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    finished = set(done["run_name"]) if len(done) else set()

    grid = list(itertools.product(a.optimizers, a.schedulers, a.losses, a.seeds))
    print(f"{len(grid)} runs: {len(a.optimizers)} optimizers x {len(a.schedulers)} schedulers x {len(a.losses)} losses x {len(a.seeds)} seeds")

    for i, (opt, sched, loss, seed) in enumerate(grid, 1):
        name = f"{opt}-{sched}-{loss}-s{seed}"
        if name in finished:
            print(f"[{i}/{len(grid)}] {name}: already done, skipping")
            continue
        print(f"\n[{i}/{len(grid)}] {name}")
        argv_run = [
            "--dataset", a.dataset, "--data-dir", a.data_dir, "--model", a.model,
            "--epochs", str(a.epochs), "--subset", str(a.subset), "--img-size", str(a.img_size),
            "--batch-size", str(a.batch_size), "--freeze-epochs", str(a.freeze_epochs),
            "--optimizer", opt, "--scheduler", sched, "--loss", loss, "--seed", str(seed),
            "--out-dir", str(out / "runs"), "--run-name", name, "--no-progress",
            "--pretrained" if a.pretrained else "--no-pretrained",
            "--save-checkpoints" if a.keep_checkpoints else "--no-save-checkpoints",
        ]
        result = train_module.main(argv_run)
        last = result["history"][-1]
        row = {
            "run_name": name, "optimizer": opt, "scheduler": sched, "loss": loss, "seed": seed,
            "best_val_acc": result["best_acc"], "final_val_acc": last["val_acc"],
            "final_train_loss": last["train_loss"], "final_val_loss": last["val_loss"],
            "seconds": round(sum(h["seconds"] for h in result["history"]), 1),
        }
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(summary_path, index=False)  # written after every run: safe against disconnects

    report(done, out)
    return done


def report(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    keys = ["optimizer", "scheduler", "loss"]
    agg = df.groupby(keys).agg(
        runs=("seed", "count"),
        best_val_acc=("best_val_acc", "mean"), acc_std=("best_val_acc", "std"),
        final_train_loss=("final_train_loss", "mean"), final_val_loss=("final_val_loss", "mean"),
    ).reset_index().sort_values("best_val_acc", ascending=False)
    print("\n=== results (mean over seeds, best first) ===")
    print(agg.round(4).to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _, r in df.iterrows():
        m = pd.read_csv(out / "runs" / r["run_name"] / "metrics.csv")
        label = f"{r['optimizer']} / {r['scheduler']} / {r['loss']}" + (f" (s{r['seed']})" if df["seed"].nunique() > 1 else "")
        axes[0].plot(m["epoch"], m["train_loss"], label=label)
        axes[1].plot(m["epoch"], m["val_acc"], label=label)
    axes[0].set_title("train loss"); axes[1].set_title("validation accuracy")
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)
    print(f"saved {out / 'summary.csv'} and {out / 'curves.png'}")


if __name__ == "__main__":
    main()
