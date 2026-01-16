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
export EDGAR_EMAIL="your_email@jpm.com"
```
The LLM clients requires API keys (currently just ChatGPT), this is soon to be required for both question parts as company financial statements are now being parsed by LLM to standardise:
```bash
export OPENAI_API_KEY="your_api_key"
```

## Question 1 - Financial Statement Forecasting
### Part 1
- **Vélez-Pareja:**
  - **Plugless:** from the paper *Forecasting Financial Statements with No plugs and No Circularity* [1]
    ```bash
    python scripts/question_1/noplug_vp.py
    ```
  - **Consistent:** from the paper *Constructing Consistent Financial Planning Models for Valuation* [2]
    ```bash
    python scripts/question_1/construct_vp.py

    python scripts/question_1/construct_vp_tf.py <- TF model
    ```

- **Custom models:**
  - Train **LSTM** forecaster:
    ```bash
    python scripts/question_1/eval_lstm.py
    ```

### Part 2
- **Ensemble model:**
    The LLM can be used to either adjust the LSTM estimation, or independently predict the future financial statement features before combining the output with the LSTM.
    ```bash
    python scripts/question_1/eval_ensemble.py
    ```
- **Annual Report Parsing:**
    This script uses the same LLM client to parse pdf annual reports, extracting key financial information. Available files are stored within `assets/`:
    ```bash
    python scripts/question_1/parse_reports.py --company ['alibaba', 'exxon', ...]
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
