# Question 3 – Part 2  
## Sparse Discrete Choice Estimation with DeepHalo

## Purpose of Part 2

Part 2 focuses on **extending and validating modern shrinkage-based discrete choice estimation methods** through simulation and controlled experimentation. Building on the Monte Carlo framework of **Lu & Shimizu (2025)**, this part evaluates whether structured regularization can be combined with a flexible neural utility model to deliver both **statistical accuracy** and **computational scalability**.

The emphasis of Part 2 is not on proposing a new theoretical estimator, but on:
- careful replication,
- diagnostic-driven implementation,
- and principled empirical comparison of competing estimation strategies.

---

## Problem Setting

Discrete choice models often face two simultaneous challenges:
1. **Context dependence** across products that cannot be captured by simple linear utilities.
2. **High-dimensional unobserved heterogeneity**, where only a small subset of market–product shocks are economically meaningful.

Lu & Shimizu (2025) address this setting using Bayesian shrinkage with spike-and-slab priors. While statistically appealing, full posterior inference is computationally expensive and difficult to scale.

Part 2 investigates whether a **MAP-based approximation** using ℓ₁ regularization can recover the same structural insights when paired with a **structured neural utility backbone**.

---

## What Is Implemented

Part 2 implements and evaluates three nested models:

1. **DeepHalo-only**  
   A context-dependent neural utility model without market-level or product-specific shocks.

2. **DeepHalo + market fixed effects (μ)**  
   Adds market-level demand shifts shared across inside goods.

3. **DeepHalo + μ + sparse market–product shocks (d)**  
   Introduces Lu-style sparse deviations estimated via a MAP objective with ℓ₁ regularization.

All models are estimated on synthetic datasets generated to match the Monte Carlo designs of Lu & Shimizu (2025).

---

## Experimental Design

- Monte Carlo simulations follow the paper’s Section 4 design:
  - sparse vs. dense unobserved shocks,
  - exogenous vs. endogenous prices.
- Multiple replications are used to evaluate:
  - parameter bias and variance,
  - likelihood performance,
  - sparsity recovery.
- Ablation studies isolate the contribution of each model component.
- Diagnostics include gradient connectivity checks and objective decomposition.

---

## Key Findings

- Classical BLP performs well only when valid instruments are available; without them, bias and variance increase substantially.
- Shrinkage-based estimators dominate classical approaches in sparse environments.
- The MAP estimator closely matches Bayesian shrinkage in parameter accuracy while being **orders of magnitude faster**.
- ℓ₁ regularization yields conservative sparsity recovery (high specificity, moderate sensitivity), consistent with theory.
- Combining DeepHalo with structured regularization improves fit while preserving interpretability.

---

## How to Run Part 2

From the Part 2 root directory:

```bash
python extension_deephalo/zhang_sparse_choice_learn.py
