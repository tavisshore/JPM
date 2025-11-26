# JPM
MLCOE Summer Associate Internship Exercises

## Install
If using conda, create a clean 3.10 env
- `conda create -n jpm python=3.10 && conda activate jpm`
- From inside this cloned repo dir:
- `python -m pip install .`

## Question 1
### Environment Variables
- Export an email for SEC access before pulling filings:
- `export EDGAR_EMAIL="your_email@jpm.com"`

### Running Models
- **Plugless** Vélez-Pareja model:
- `python scripts/question_1/noplug_vp.py`
- **Consistent** Vélez-Pareja model:
- `python scripts/question_1/construct_vp.py`
- Train **LSTM** forecaster:
- `python scripts/question_1/eval_lstm.py --cache_dir /PATH/TO/DESIRED/CACHE`
- Financial statements from the SEC will be stored in your cache directory.
