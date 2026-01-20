# Question 3 — Scripts

This folder contains the **runnable scripts** :
Part 1 — Zhang et al. (2025): Deep context-dependent choice / DeepHalo

Part 2 — Lu & Shimizu (2025): Sparse discrete choice estimation (choice-learn integration)

> Run all commands from the repository root: `JPM/`

---

## Environment setup
In normal usage, no additional environment setup is required, as the repository-level environment already includes all necessary dependencies.

If you encounter environment-related errors (e.g. TensorFlow / Keras compatibility issues), you may use the local fallback setup provided in this folder.

To install the local dependencies:

```bash
pip install -r scripts/question_3/env/requirements-choicelearn.txt
pip install -e .
```
