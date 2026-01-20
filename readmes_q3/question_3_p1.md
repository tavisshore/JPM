# Question 3 – Part 1
## Understanding Context Effects in Discrete Choice Models

## Repository Structure (Part 1)
question_3/
│
├── authors/ # Reference and baseline implementations
│
├── choice_learn_ext/ # Lightweight extensions and utilities built on choice-learn
│
├── examples/ # Small runnable examples and demonstrations
│
├── experiments/ # Synthetic experiments (primarily Part 1)
│
├── Part_2_Answer/ # Full implementation and analysis for Part 2
│ ├── extension_deephalo/ # DeepHalo + sparse MAP estimator
│ ├── README.md # Part 2–specific documentation
│ └── ...
│
├── tests/ # Basic sanity and regression tests
│
├── Makefile # Convenience commands for running experiments
├── init.py
└── readme.md # This file (top-level overview)



## Purpose of Part 1

Part 1 focuses on **building intuition and empirical evidence for context effects in discrete choice models**. The objective is to demonstrate why classical utility specifications are insufficient in settings where consumer preferences depend on the composition of the choice set, and to motivate the need for more expressive utility representations.

Rather than proposing a new estimator, Part 1 establishes the **behavioral and modeling foundations** that justify the methods developed in Part 2.

---

## Problem Setting

Standard discrete choice models assume that the utility of an option depends only on its own attributes and an idiosyncratic shock. This assumption rules out well-documented behavioral phenomena such as:

- attraction (decoy) effects,
- similarity effects,
- violations of the independence of irrelevant alternatives (IIA).

These effects arise when the presence or characteristics of other options influence relative preferences, even when those options are not chosen.

Part 1 investigates these issues through controlled synthetic experiments.

---

## What Is Implemented

Part 1 implements a set of simulation-based experiments designed to illustrate context dependence in choice behavior:

- Construction of synthetic choice environments exhibiting attraction and decoy effects.
- Comparison of feature-based and featureless utility representations.
- Demonstration of how classical multinomial logit fails to capture these effects.
- Illustration of how richer representations can encode cross-item interactions.

The emphasis is on **conceptual clarity and diagnostics**, not on large-scale estimation.

---

## Experimental Design

- Choice sets are generated to explicitly violate IIA.
- Utilities are constructed to depend on relative positioning of products rather than only absolute attributes.
- Models are evaluated based on their ability to reproduce observed choice distortions.
- Experiments are deterministic and small-scale to ensure interpretability.

---

## Key Findings

- Classical logit models are fundamentally unable to reproduce attraction and decoy effects.
- Context dependence must be explicitly modeled through interactions across alternatives.
- Even simple synthetic settings reveal large qualitative differences between context-free and context-aware utilities.
- These results motivate the need for more expressive utility backbones, such as DeepHalo.

---

## How to Run Part 1

conda activate q3p2

python experiments/decoy_effect.py

python experiments/attraction_effect_tf.py

python experiments/synthetic_decoy.py
