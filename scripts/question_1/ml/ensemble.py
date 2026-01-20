"""
Ensemble Model

Combines LSTM forecasting with LLM-based adjustments to create an ensemble
approach for predicting financial statements. The script:

- Trains an LSTM model on historical financial data
- Uses an LLM to refine or average predictions
- Evaluates the ensemble on validation data
- Computes balance sheet identity error and income statement metrics

The ensemble approach leverages both neural network pattern recognition
and LLM reasoning capabilities for improved forecast accuracy.
"""

from jpm.config import Config, DataConfig, LLMConfig, LSTMConfig
from jpm.question_1 import (
    BalanceSheet,
    EdgarData,
    IncomeStatement,
    LSTMForecaster,
    StatementsDataset,
    get_args,
    set_seed,
)

set_seed(42)
args = get_args()

data_cfg = DataConfig.from_args(args)
lstm_cfg = LSTMConfig.from_args(args)
llm_cfg = LLMConfig.from_args(args)
config = Config(data=data_cfg, lstm=lstm_cfg, llm=llm_cfg)

data = EdgarData(config=config)
dataset = StatementsDataset(edgar_data=data)


model = LSTMForecaster(config=config, data=data, dataset=dataset)
model.fit()

validation_results = model.evaluate(stage="val", llm_config=llm_cfg)
model.view_results(stage="val")

# Pass outputs to BS Model
bs = BalanceSheet(config=config, data=data, dataset=dataset, results=validation_results)
bs_pct_error = bs.check_identity()

i_s = IncomeStatement(config=config, dataset=dataset, results=validation_results)
i_s.view()
is_results = i_s.get_results()
