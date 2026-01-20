# Choice-Learn Extension (Lu 2025 in TensorFlow)

This folder contains a **TensorFlow / TensorFlow Probability implementation** of Lu-style sparse demand shock estimation **integrated with the `choice-learn` framework**.

> **Purpose:**
> Translate the econometric ideas in **Lu & Shimizu (2025)** into a modern, differentiable ML framework suitable for neural discrete-choice models.

---

## What This Folder Is (and Is Not)

- A **choice-learn–compatible implementation** of sparse market–product demand shocks
- Uses **TensorFlow + TensorFlow Probability**
- Implements a **MAP-style approximation** to Lu’s Bayesian shrinkage estimator
- Designed to interoperate with neural utility models


---

## Conceptual Role in Part 2

This folder serves as the **bridge between econometrics and machine learning**:

| Folder | Role |
|------|------|
| `replication_lu25/` | Paper-faithful Monte Carlo replication |
| `choice_learn_extension/` | ML-native implementation of Lu-style sparsity |
| `extension_deephalo/` | DeepHalo + Lu-style sparsity combined |


---

## What Is Implemented

- Latent **market fixed effects** (μₜ)
- Sparse **market–product shocks** (dⱼₜ)
- ℓ₁-regularized MAP objective
- Differentiable likelihood compatible with `choice-learn`
- TensorFlow Probability priors and penalties

The implementation prioritizes:
- transparency,
- gradient correctness,
- and compatibility with neural utility backbones.

---

## Directory Structure

```text
choice_learn_extension/
│
├── run_replication_choice_learn.py   # Main executable (entry point) Moved to script
│
└── README.md                         # This file
