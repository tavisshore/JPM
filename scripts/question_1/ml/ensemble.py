from jpm.question_1 import (
    BalanceSheet,
    Config,
    DataConfig,
    EdgarData,
    IncomeStatement,
    LLMConfig,
    LossConfig,
    LSTMForecaster,
    ModelConfig,
    StatementsDataset,
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

data = EdgarData(config=config)
dataset = StatementsDataset(edgar_data=data)


model = LSTMForecaster(config=config, data=data, dataset=dataset)
model.fit()

# model.evaluate(stage="train")
validation_results = model.evaluate(stage="val", llm_config=llm_cfg)
model.view_results(stage="val")

# Pass outputs to BS Model
bs = BalanceSheet(config=config, data=data, dataset=dataset, results=validation_results)
bs_pct_error = bs.check_identity()

i_s = IncomeStatement(config=config, dataset=dataset, results=validation_results)
i_s.view()
is_results = i_s.get_results()
