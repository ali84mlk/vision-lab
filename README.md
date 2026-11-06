# Vision Lab

An image classifier you can build **without a strong PC**, and a hands-on way to learn
**loss functions, optimizers (SGD / momentum / Adam / AdamW) and learning-rate schedules**.

1. **Write and debug locally** on CPU with a tiny slice of the data (`--subset 0.02`).
2. **Train for free** on Kaggle Notebooks (T4 / P100 GPU) or Google Colab, straight from this repo.
3. **Learn by swapping parts.** Every loss, optimizer and schedule is a small readable module, with a
   from-scratch version and a CPU-only study notebook next to it.

The project is built step by step with **one commit per step**: `git log --oneline` reads like a tutorial.

## Learning path

Do these in order (each notebook runs on CPU in about a minute and ends with exercises):

| # | Topic | Read the code | Study notebook | CLI flag |
|---|---|---|---|---|
| 1 | Loss functions: softmax, cross-entropy, label smoothing, focal loss | [`src/losses.py`](src/losses.py) | [`learn/01_loss_functions.ipynb`](learn/01_loss_functions.ipynb) | `--loss` |
| 2 | Optimizers: SGD, momentum, Nesterov, Adam, AdamW | [`src/optim.py`](src/optim.py), [`src/optim_scratch.py`](src/optim_scratch.py) | [`learn/02_optimizers.ipynb`](learn/02_optimizers.ipynb) | `--optimizer` |
| 3 | Learning-rate schedules, warmup, the LR range test | [`src/schedulers.py`](src/schedulers.py) | [`learn/03_lr_schedules.ipynb`](learn/03_lr_schedules.ipynb) | `--scheduler` |
| 4 | Put it together on real images | [`src/train.py`](src/train.py) | [`notebooks/runner.ipynb`](notebooks/runner.ipynb) | |
| 5 | Measure which choices actually help | [`experiments/compare.py`](experiments/compare.py) | (last section of the runner) | |

## Project layout

```
src/
  utils.py          detect Kaggle/Colab/local, data/output dirs, seeding, checkpoints (full training state)
  data.py           oxford_pet / cifar10 / your own image folder; --subset for quick debug runs
  model.py          transfer learning: resnet18 / mobilenet_v3_small / efficientnet_b0, backbone freezing
  losses.py         softmax + cross-entropy from scratch, FocalLoss, build_loss()
  optim.py          build_optimizer(): sgd, sgd_momentum, sgd_nesterov, adam, adamw
  optim_scratch.py  SGD/momentum/Nesterov and Adam/AdamW written by hand (match torch.optim to ~1e-15)
  schedulers.py     constant, step, cosine, warmup_cosine, onecycle (+ simulate_schedule for plots)
  train.py          the training CLI (resumable, CPU or CUDA + mixed precision)
  evaluate.py       accuracy, hardest classes, confusion matrix
experiments/
  compare.py        optimizer x schedule x loss x seed grid -> summary.csv + curves.png
learn/              study notebooks 01-03 (CPU only, executed, with exercises)
notebooks/
  runner.ipynb      one notebook for Kaggle and Colab (clone, train, evaluate, compare)
  kernel-metadata.json   for `kaggle kernels push`
```

## Quick start: locally (CPU)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Smoke test, about a minute. It downloads the Oxford-IIIT Pets dataset (about 800 MB) into `data/` the first time;
pass `--data-dir` to reuse a copy you already have:

```bash
python -m src.train --subset 0.02 --epochs 2 --model mobilenet_v3_small --img-size 112 --batch-size 16 --run-name smoke
python -m src.evaluate --run-dir checkpoints/smoke --subset 0.02
```

At the start you should see `[sanity] initial loss ... (expected about ln(37) = 3.611)`: an untrained classifier should
score about `ln(number of classes)`. If not, something is wrong (see notebook 01).

Swap the three things this project is about:

```bash
python -m src.train --optimizer sgd_momentum --scheduler step --loss cross_entropy ...
python -m src.train --optimizer adamw --scheduler warmup_cosine --loss label_smoothing ...
python -m src.train --freeze-epochs 2 ...      # train only the new head first, then fine-tune everything
python -m src.train --resume ...               # continue from checkpoints/<run-name>/last.pt
```

## Quick start: free GPU on Kaggle

1. Put this repo on GitHub as a **public** repository (the notebook clones it; a private repo would need credentials).
2. On Kaggle: **Create → New Notebook → File → Import Notebook** and upload `notebooks/runner.ipynb`.
3. In the `REPO_URL` cell, put your repo URL.
4. Right panel → *Session options*: **Accelerator: GPU T4 x2** (or P100), **Internet: On**.
5. **Run All**, or **Save Version → Save & Run All** to train in the background and close the tab.
6. To share it: **Share → Public**.

Command-line alternative (needs your own Kaggle API token in `~/.kaggle/kaggle.json`, never committed):
edit the username in `notebooks/kernel-metadata.json`, then `kaggle kernels push -p notebooks`.

## Quick start: Google Colab

Open `notebooks/runner.ipynb` in Colab (*File → Open notebook → GitHub*), choose *Runtime → Change runtime type → T4 GPU*,
set `REPO_URL`, and run all cells. Checkpoints go to your Google Drive (`MyDrive/vision-lab/outputs`), so a disconnect does
not lose the run: re-run the same cell and `--resume` continues.

## Use your own data

Put images in `my_data/<class_name>/*.jpg` (or `my_data/train/...` and `my_data/val/...`), then:

```bash
python -m src.train --dataset folder --data-dir my_data --model resnet18 --epochs 10
```

## Results

Not published yet: the numbers in this repo's test runs are from 2-3% subsets and only prove the code runs. After you
run `experiments/compare.py` on a GPU, paste your `summary.csv` table here (use `--seeds 0 1 2` so the table shows mean and
spread, since small differences from one seed are often just luck).

## Troubleshooting

* **`OSError: [WinError 1114] ... c10.dll` on `import torch` (Windows).** A broken PyTorch install. Create a fresh virtual
  environment and install into it (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`).
* **A Jupyter kernel uses the wrong Python.** Register one for the venv:
  `python -m ipykernel install --user --name vision-lab`.
* **Out of GPU memory.** Lower `--batch-size` or `--img-size`, or switch to `--model mobilenet_v3_small`.
* **Dataloader hangs on Windows.** Use `--num-workers 0` (the default on Windows).
