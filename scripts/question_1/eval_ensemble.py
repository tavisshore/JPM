from jpm.question_1 import (
    BalanceSheet,
    Config,
    DataConfig,
    EdgarDataLoader,
    IncomeStatement,
    LLMConfig,
    LossConfig,
    LSTMForecaster,
    ModelConfig,
    TrainingConfig,
    get_args,
    set_seed,
)

set_seed(42)
args = get_args()

data_cfg = DataConfig.from_args(args)
model_cfg = ModelConfig.from_args(args)
train_cfg = TrainingConfig.from_args(args)
loss_cfg = LossConfig.from_args(args)
llm_cfg = LLMConfig.from_args(args)
config = Config(
    data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg, llm=llm_cfg
)

data = EdgarDataLoader(config=config)

model = LSTMForecaster(config=config, data=data)
model.fit()

# model.evaluate(stage="train")
validation_results = model.evaluate(stage="val", llm_config=llm_cfg)
model.view_results(stage="val")

# Pass outputs to BS Model
bs = BalanceSheet(config=config, results=validation_results)
bs.check_identity()

# Income Statement to predict Net Income (Loss)
i_s = IncomeStatement(config=config, results=validation_results)
i_s.view()
