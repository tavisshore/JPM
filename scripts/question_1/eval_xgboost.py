"""Evaluate XGBoost model for financial forecasting."""

from jpm.question_1 import (
    BalanceSheet,
    Config,
    DataConfig,
    EdgarDataLoader,
    IncomeStatement,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    XGBoostForecaster,
    get_args,
    set_seed,
)

set_seed(42)
args = get_args()

data_cfg = DataConfig.from_args(args)
model_cfg = ModelConfig.from_args(args)
train_cfg = TrainingConfig.from_args(args)
loss_cfg = LossConfig.from_args(args)
config = Config(data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg)

data = EdgarDataLoader(config=config)

# Train XGBoost model
model = XGBoostForecaster(config=config, data=data)
model.fit(verbose=1, max_depth=8, n_estimators=200)

# Evaluate on validation set
validation_results = model.evaluate(stage="val")
model.view_results(stage="val")

# Pass outputs to BS Model
bs = BalanceSheet(config=config, data=data, results=validation_results)
bs.check_identity()

# Income Statement to predict Net Income (Loss)
i_s = IncomeStatement(config=config, data=data, results=validation_results)
i_s.view()
