# Sparse Discrete Choice Modeling with DeepHalo

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
=======
# Lu (2025) – Section 4 Replication

This folder contains a **standalone Monte Carlo replication** of the simulation results in **Lu (2025), Section 4**, focusing on:

- Standard BLP with valid cost instruments (BLP+IV)
- BLP without cost instruments (BLP–IV)
- A shrinkage-based estimator under sparse market–product demand shocks

The goal is to reproduce the *qualitative and quantitative patterns* emphasized in the paper:
- Endogeneity bias in BLP without valid IVs
- Improvement from valid cost IVs
- Further gains from exploiting sparsity in demand shocks

This code is intentionally **self-contained and explicit**, prioritizing transparency over abstraction.

---

## High-level overview

Each Monte Carlo replication proceeds as follows:

1. **Simulate markets** under a specified Data Generating Process (DGP)
   - Discrete-choice demand with random coefficients
   - Endogenous prices
   - Sparse market–product demand shocks
2. **Estimate demand** using:
   - BLP with cost IVs
   - BLP without cost IVs
   - Shrinkage estimator exploiting sparsity
3. **Aggregate results** across replications
   - Mean, bias, and dispersion of key parameters
4. **Compare estimators** in the spirit of Lu (2025), Section 4

The main entry point is `run_mc.py`.

---

## Directory structure

replication_lu25/
├── run_mc.py # Main Monte Carlo replication script (high computation time)
├── run_single_replication.py # Single-replication debug runner
├── test_blp.py # Lightweight sanity checks
├── sanity_checks.py # Diagnostics for sparsity and simulation
│
├── estimators/
│ ├── blp.py # BLP contraction + IV/GMM estimation
│ └── shrinkage.py # Shrinkage estimator under sparse shocks
│
├── simulation/
│ ├── config.py # Simulation configuration dataclasses
│ ├── dgp.py # Data-generating processes (DGP1–DGP4)
│ ├── market.py # Single-market simulation logic
│ └── simulate.py # Dataset-level simulation wrapper
│
├── results/ # Generated outputs (not tracked by git)



---

## Data Generating Processes (DGPs)

The DGPs are defined in `simulation/dgp.py`.

Key features:

- Markets indexed by `t = 1,...,T`
- Products indexed by `j = 1,...,J`
- Utility:
  \[
  u_{ijt} = \delta_{jt} + \mu_{ijt} + \varepsilon_{ijt}
  \]
- Mean utility:
  \[
  \delta_{jt} = X_{jt}\beta + \xi_{jt}
  \]

### Sparse market–product shocks

The demand shock is decomposed as:
\[
\xi_{jt} = \bar{\xi}_t + \eta_{jt}
\]

- `η_{jt}` is **sparse**:
  - Only a fraction of products have nonzero deviations
  - Remaining products have zero shock
- This matches the motivation in Lu (2025): many products share common shocks, with only a few deviating.

Different DGPs vary:
- Degree of sparsity
- Strength of endogeneity
- Correlation structure

---

## Estimators

### 1. BLP with cost IVs (BLP+IV)

Implemented in `estimators/blp.py`.

- Standard Berry (1994) contraction mapping
- GMM/2SLS objective
- Cost shifters used as valid instruments
- Serves as the main benchmark

### 2. BLP without cost IVs (BLP–IV)

Also in `estimators/blp.py`.

- Cost instruments removed
- Identification relies only on included characteristics
- Included as a deliberately misspecified benchmark to illustrate endogeneity bias

### 3. Shrinkage estimator

Implemented in `estimators/shrinkage.py`.

- Extends the BLP structure by imposing **shrinkage on demand shocks**
- Motivated by sparsity in `η_{jt}`
- Uses a spike-and-slab–style mechanism to regularize latent shocks
- Estimated jointly with demand parameters

This estimator is designed to exploit information that standard BLP ignores.

---

## Main replication script

### `run_mc_copy.py`

This is the **official replication driver**.

What it does:

- Loops over:
  - DGPs (e.g. DGP1–DGP4)
  - Market/product sizes `(T, J)`
  - Monte Carlo replications
- For each replication:
  - Simulates a dataset
  - Estimates all three models
  - Stores parameter estimates
- Aggregates results into summary statistics:
  - Mean
  - Bias
  - Standard deviation

The structure mirrors the tables in Lu (2025), Section 4.

---

## Running the replication

### Environment

This code is intended to be run in a TensorFlow-enabled conda environment.

```bash
# Make environment (Required packages are listed in the repository-level environment.yml)

conda env create -f environment.yml

# Run a single simulation

python run_single_replication.py

# Replicate Table results from section 4:

python run_mc.py
1be4ccf1e4620038679e53b55406005cb3f02d42
