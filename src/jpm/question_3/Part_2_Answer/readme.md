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

# Part 2 – Lu (2025) Replication and Extensions

This directory contains all work for **Part 2 of the project**, which focuses on:

1. Replicating the simulation results in **Lu (2025), Section 4**
2. Implementing and evaluating a **shrinkage-based BLP estimator** under sparse market–product demand shocks
3. Integrating discrete-choice models with **choice-learn** using TensorFlow / TensorFlow Probability
4. Extending **DeepHalo (Zhang, 2025)** to settings motivated by Lu-style sparsity
5. Providing a written technical analysis addressing the conceptual questions in the job prompt

All Part 2 components are **self-contained** and do not modify the Part 1 DeepHalo implementation.

---

## Directory overview
part2/
├── README.md #  Part 2 roadmap
├── replication_lu25/ # Lu (2025) Section 4 simulation replication
├── choice_learn_extension/ # choice-learn + TensorFlow implementation
├── extension_deephalo/ # DeepHalo under sparse market–product shocks
├── experiments/ # Additional / robustness experiments 
├── results/ # Saved tables, figures, and diagnostics
└── report/ # Written analysis for Part 2


Each subdirectory contains its own README with more detailed explanations.

---

## Requirements mapping (job post → code)

| Job requirement |          Location        |
|-----------------|--------------------------|
| Replicate Lu (2025) Section 4 simulations | `replication_lu25/` |
| BLP with cost IVs | `replication_lu25/estimators/blp.py` |
| BLP without cost IVs | `replication_lu25/estimators/blp.py` |
| Shrinkage estimator under sparsity | `replication_lu25/estimators/shrinkage.py` |
| Monte Carlo comparison of estimators | `replication_lu25/run_mc_copy.py` |
| TensorFlow / TFP implementation | `choice_learn_extension/` |
| Integration with choice-learn | `choice_learn_extension/` |
| DeepHalo modification motivated by Lu (2025) | `extension_deephalo/` |
| Simulation-based validation of extension | `extension_deephalo/` |
| Discussion of assumptions, IVs, limitations | `report/` |

---

## How to navigate this folder

### 1. Lu (2025) replication
If you want to see **how the paper’s simulation results are reproduced**, start here:

part2/
├── replication_lu25/ 

- Contains the full Monte Carlo setup
- Includes BLP+IV, BLP–IV, and shrinkage estimators
- Focuses on sparse market–product demand shocks
- Main entry point: `run_mc_copy.py``

See `replication_lu25/README.md` for full details.

---------------

### 2. choice-learn implementation
If you want to see **how the models are implemented in a modern ML framework**, go to:

part2/
├── choice_learn_extension/

This directory contains:
- TensorFlow / TensorFlow Probability implementations
- Model abstractions compatible with choice-learn
- Clean separation between econometric structure and optimization

This code is intended to be reusable beyond the replication.

---------------

### 3. DeepHalo extension
If you want to see the **original extension** required by Part 2, go to:

part2/
├── extension_deephalo/

This folder:
- Modifies DeepHalo (from Part 1) to incorporate sparsity motivated by Lu (2025)
- Designs a synthetic experiment to validate the modification
- Compares performance against relevant benchmarks

This is the main creative contribution of Part 2.

---------------

### 4. Results
All saved outputs live in:

part2/
├── results/

Including:
- Replication tables (CSV / LaTeX)
- Raw Monte Carlo estimates
- Diagnostics and runtime summaries

Generated files are not tracked by git.

------------

### 5. Written analysis
The written response to the Part 2 conceptual questions is in:

part2/
├── report/


This includes discussion of:
- Unclear or limiting assumptions in Lu (2025)
- Identification and IV assumptions in BLP
- Instrument design in the credit card offer context
- Applicability of sparsity assumptions
- Interpretation of simulation results

---------------

## Environment and reproducibility

Part 2 is intended to be run in a **TensorFlow-enabled conda environment**.

From the repository root:

```bash
conda activate tf-mac
 1be4ccf1e4620038679e53b55406005cb3f02d42
