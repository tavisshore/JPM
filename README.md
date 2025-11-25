# JPM
MLCOE Summer Associate Internship Exercises

## Install
- In a virtualenv/conda env: `python -m pip install .`

## Question 1
### Environment Variables
- Export an email for SEC access before pulling filings:
- `export EDGAR_EMAIL="your_email@jpm.com"`

### Running Models
- Plugless Vélez-Pareja model: `python scripts/question_1/noplug_vp.py`
- Train LSTM forecaster:
- `python scripts/question_1/eval_lstm.py --cache_dir /PATH/TO/DESIRED/CACHE`
- Financial statements from the SEC will be stored in your cache directory.
