# JPM
MLCOE Summer Associate Internship exercises and reproducible finance models.

## Setup
Create a clean Python 3.10 environment and install the package:
```bash
conda create -n jpm python=3.10
conda activate jpm
python -m pip install .
```

## Environment
EDGAR requires an email for downloads:
```bash
export EDGAR_EMAIL="your_email@jpm.com"
```

## Run the models (Question 1)
- **Plugless** Vélez-Pareja:
  `python scripts/question_1/noplug_vp.py`

- **Consistent** Vélez-Pareja:
  `python scripts/question_1/construct_vp.py`

- Train **LSTM** forecaster:
  `python scripts/question_1/eval_lstm.py --cache_dir /PATH/TO/DESIRED/CACHE`

Downloaded SEC statements are cached under the chosen `--cache_dir`.
