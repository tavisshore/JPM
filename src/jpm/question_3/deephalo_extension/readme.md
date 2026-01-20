# Sparse Discrete Choice Modeling with DeepHalo 
# Zhang-Sparse (Lu25-aligned) estimator implemented using DeepHalo (Zhang 2025)


## Overview

This project studies how to efficiently estimate discrete choice models with rich context dependence and structured unobserved heterogeneity. It combines a flexible neural utility backbone (DeepHalo) with a sparse market–product shock decomposition inspired by Lu & Shimizu (2025), and evaluates the approach through controlled Monte Carlo experiments.

The goal is twofold:

1. **Replication**
   Reproduce and validate the Monte Carlo findings of Lu & Shimizu (2025) under sparse and dense unobserved shocks.

2. **Extension**
   Demonstrate that a scalable MAP-based approximation can recover the key benefits of Bayesian shrinkage while dramatically reducing computational cost.

The project emphasizes methodological correctness, diagnostics, and interpretability rather than black-box performance.

---

## Key Ideas

- **Context-dependent utilities**
  DeepHalo provides a structured neural representation of choice utilities that captures interactions across products without relying on hand-crafted features.

- **Structured unobserved heterogeneity**
  Market–product shocks are decomposed into:
  - a market-level component shared across inside goods, and
  - a sparse product-specific deviation.

- **Estimation strategies compared**
  - Classical BLP (with and without valid instruments),
  - Bayesian shrinkage using spike-and-slab priors (MCMC),
  - A fast MAP estimator using ℓ₁ regularization as a convex approximation.

---

## What Was Done

- Replicated Monte Carlo designs from Lu & Shimizu (2025), including:
  - sparse vs. dense unobserved shocks,
  - exogenous vs. endogenous prices.
- Identified and documented reproducibility and non-determinism issues in the original implementation.
- Implemented a MAP-based sparse estimator integrated with DeepHalo.
- Conducted ablation studies isolating:
  - context dependence only,
  - market fixed effects,
  - full sparse market–product shocks.
- Evaluated both parameter recovery and sparsity recovery.

---

## Main Findings

- Classical BLP performs well only when valid instruments are available; without them, bias and variance increase sharply.
- Shrinkage-based estimators substantially improve estimation accuracy in sparse environments.
- The MAP estimator closely matches Bayesian shrinkage in terms of parameter accuracy while being significantly faster.
- Sparsity recovery shows the expected trade-off: MAP favors specificity over sensitivity due to ℓ₁ shrinkage, but still captures the dominant structure.
- Combining a flexible utility backbone with structured regularization yields both interpretability and improved fit.

---

## How to Run

The main experiment can be executed from the project root:

```bash
python extension_deephalo/zhang_sparse_choice_learn.py
```

