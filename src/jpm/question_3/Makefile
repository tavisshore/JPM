# -------------------------------------------------------------
# Makefile for JPM Question 3 – DeepHalo reproduction
# -------------------------------------------------------------

# Python and environment
PYTHON      := python
PYTHONPATH  := .

# Experiments (run with python -m)
TABLE1      := experiments.reproduce_table1
DECOY       := experiments.decoy_effect
ATTRACTION  := experiments.attraction_effect_tf
COMPROMISE  := experiments.compromise_effect_tf
TORCH_ATTR  := experiments.attraction_effect_torch

# -------------------------------------------------------------
# 1. Run ALL experiments
# -------------------------------------------------------------
all: table1 decoy attraction compromise

# -------------------------------------------------------------
# 2. Individual experiments
# -------------------------------------------------------------
table1:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(TABLE1)

decoy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(DECOY)

attraction:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(ATTRACTION)

compromise:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(COMPROMISE)

torch-attraction:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(TORCH_ATTR)

# -------------------------------------------------------------
# 3. Run tests
# -------------------------------------------------------------
test:
	PYTHONPATH=$(PYTHONPATH) pytest -v

# -------------------------------------------------------------
# 4. Environment setup (optional)
# -------------------------------------------------------------
env:
	conda env create -f environment.yml || echo "Environment already exists."
	@echo "Activate with: conda activate tf-mac"

# -------------------------------------------------------------
# 5. Clean generated artifacts
# -------------------------------------------------------------
clean:
	rm -f experiments/*.png
	rm -f experiments/*.csv
	rm -rf __pycache__ */__pycache__
	@echo "Clean complete."

.PHONY: all table1 decoy attraction compromise torch-attraction test env clean
