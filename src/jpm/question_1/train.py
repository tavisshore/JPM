from src.jpm.question_1.config import (
    Config,
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from src.jpm.question_1.data.ed import EdgarDataLoader
from src.jpm.question_1.misc import train_args
from src.jpm.question_1.models.bs import BalanceSheet
from src.jpm.question_1.models.lstm import LSTMForecaster

args = train_args()

data_cfg = DataConfig.from_args(args)
model_cfg = ModelConfig.from_args(args)
train_cfg = TrainingConfig.from_args(args)
loss_cfg = LossConfig.from_args(args)

config = Config(data=data_cfg, model=model_cfg, training=train_cfg, loss=loss_cfg)

data = EdgarDataLoader(config=config)

model = LSTMForecaster(config=config, data=data)
model.fit()

model.evaluate(stage="train")
validation_results = model.evaluate(stage="val")

model.view_results(stage="train")
model.view_results(stage="val")

# Pass outputs to BS Model
bs_model = BalanceSheet(config=config, model_dict=validation_results)
print(bs_model)
