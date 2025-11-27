# JPM
MLCOE Summer Associate Internship exercises and reproducible finance models.

## Setup
Create a clean Python 3.10 environment and install the package:
```bash
conda create -n jpm python=3.10
conda activate jpm
python -m pip install .

pytest -v
```

## Environment
EDGAR requires an email for downloads:
```bash
export EDGAR_EMAIL="your_email@jpm.com"
```

## Question 1 - Evaluating Models

- **Vélez-Pareja:**
  - **Plugless:**
    ```bash
    python scripts/question_1/noplug_vp.py
    ```
  - **Consistent:**
    ```bash
    python scripts/question_1/construct_vp.py
    ```

- **Custom models:**
  - Train **LSTM** forecaster:
    ```bash
    python scripts/question_1/eval_lstm.py --cache_dir /PATH/TO/DESIRED/CACHE
    ```

Downloaded SEC statements are cached under the chosen `--cache_dir`.


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

  
  

