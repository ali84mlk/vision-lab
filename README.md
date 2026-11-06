# Vision Lab

An image classifier you can build **without a strong PC**, and a hands-on way to learn
**loss functions, optimizers (SGD / momentum / Adam / AdamW) and learning-rate schedules**.

The idea:

1. **Write and debug locally** on CPU with a tiny slice of the data.
2. **Train for free** on Kaggle Notebooks (T4 / P100 GPU) or Google Colab.
3. **Learn by swapping parts**: every loss, optimizer and schedule is a small, readable module,
   with a from-scratch version next to the library version in `learn/`.

> Work in progress: built step by step, one commit per step. See `git log` for the story.

## Status

- [x] Repo setup
- [x] Data + utils
- [x] Model (transfer learning)
- [ ] Losses + study notebook
- [ ] Optimizers + study notebook
- [ ] LR schedules + study notebook
- [ ] Train / evaluate scripts
- [ ] Comparison experiment
- [ ] Kaggle / Colab runner notebook

## Quick start (local, CPU)

```bash
pip install -r requirements.txt
```

More instructions will appear here as the project grows.
