# JPM
MLCOE Summer Associate Internship Exercises

## Install
- In a virtualenv/conda env: `pip install -e .`

## Question 1
### Environment Variables
- Export an email for SEC access before pulling filings:
- `export EDGAR_EMAIL="your_email@jpm.com"`

### Running Models
- Plugless Vélez-Pareja model: `python -m jpm.question_1.models.plugless`
- Train LSTM forecaster:
- `python -m jpm.question_1.models.lstm --cache_dir /PATH/TO/DESIRED/CACHE`
- Financial statements from the SEC will be stored in your cache directory.
