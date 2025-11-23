# JPM
MLCOE Summer Associate Internship Exercises

## Install
- In a virtualenv/conda env: `pip install -e .`

## Question 1 Setup
- Export an email for SEC access before pulling filings: `export EDGAR_EMAIL="your_email@jpm.com"`

## Question 1: Running Models
- Plugless model: `python -m jpm.question_1.models.plugless`
- Train LSTM forecaster: `python -m jpm.question_1.models.lstm` (accepts CLI args via `train_args`)
