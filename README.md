# JPM - MLCOE Internship Exercises

## Setup
Create a clean Python 3.10 environment and install the package:
```bash
conda create -n jpm python=3.10
conda activate jpm
python -m pip install .
# Optional testing
pytest -v
```

## Environment
EDGAR requires an email for downloads:
```bash
export EDGAR_EMAIL="your_email@address.com"
```
The LLM clients requires API keys (currently just ChatGPT):
```bash
export OPENAI_API_KEY="your_api_key"
```
We use `https://www.exchangerate-api.com/` to retrieve exchange rates for particular dates - this must be set to sucessfully parse non-USD reports.
```bash
export FX_API_KEY='your_fx_api_key'
```

## Question 1 - Financial Statement Forecasting
### Part 1
- **Vélez-Pareja:**
  - **Plugless:** from the paper *Forecasting Financial Statements with No plugs and No Circularity* [1]
    ```bash
    python scripts/question_1/valez/noplug_vp.py
    ```
  - **Consistent:** from the paper *Constructing Consistent Financial Planning Models for Valuation* [2]
    ```bash
    python scripts/question_1/valez/construct_vp.py    # <- pd.series model
    python scripts/question_1/valez/construct_vp_tf.py # <- TF model
    ```

- **Time-series Forecasting:**
  - Train **LSTM** forecaster:
    ```bash
    python scripts/question_1/ml/lstm.py
    ```

### Part 2
- **Ensemble model:**
    The LLM can be used to either adjust the LSTM estimation, or independently predict the future financial statement features before combining the output with the LSTM.
    ```bash
    python scripts/question_1/ml/ensemble.py
    ```

- **Annual Report Parsing:**
    This script uses the same LLM client to parse pdf annual reports, extracting key financial information. Available reports are stored within `assets/`:
    (the argument for parsing is `ticker` although it's the name - to be compatible throughout the config)
    ```bash
    python scripts/question_1/ml/parse_reports.py --ticker ['alibaba', 'exxon', 'evergrande' ...]
    ```
#### Part B: Bonus 1
- **Credit Rating:**
    This script trains an XGBoost model on credit ratings data constructed from our SEC data and `ratingshistory.info`:
    ```bash
    python scripts/question_1/ml/pipeline.py --ticker ['alibaba', 'exxon', 'evergrande' ...]
    ```

## Question 3 - DeepHalo Reproduction
The experiments are implemented as Python modules and can be invoked directly.
- Table 1
  ```
  python -m experiments.reproduce_table1
  ```
- Decoy effect
  ```
  python -m experiments.decoy_effect
  ```
- Attraction effect
  ```
  python -m experiments.attraction_effect_tf
  ```
- Compromise effect
  ```
  python -m experiments.compromise_effect_tf
  ```
- Attraction effect (PyTorch)
  ```
  python -m experiments.attraction_effect_torch
  ```
---

## Citations
```
[1] Velez-Pareja, Ignacio, Forecasting Financial Statements with No Plugs and No Circularity (May 22, 2012). The IUP Journal of Accounting Research & Audit Practices, Vol. X, No. 1, 2011, Available at SSRN: https://ssrn.com/abstract=1031735
[2] Velez-Pareja, Ignacio, Constructing Consistent Financial Planning Models for Valuation (August 15, 2009). IIMS Journal of Management of Science, Vol. 1, January-June 2010, Available at SSRN: https://ssrn.com/abstract=1455304
```
