from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, cast

import tensorflow as tf

from src.jpm.question_1.misc import to_tensor


@dataclass
class IncomeStatement:
    total_revenue: tf.Tensor
    cost_of_revenue: tf.Tensor
    gross_profit: tf.Tensor
    operating_expense: tf.Tensor
    operating_income: tf.Tensor
    interest_expense: tf.Tensor
    other_income_expense: tf.Tensor
    income_before_tax: tf.Tensor
    income_tax_expense: tf.Tensor
    net_income: tf.Tensor

    validity_threshold: tf.Tensor = field(
        default_factory=lambda: cast(tf.Tensor, tf.constant(1e-4, dtype=tf.float32))
    )

    def __post_init__(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, to_tensor(getattr(self, f)))

    def validate(self) -> Dict[str, tf.Tensor]:
        errs: Dict[str, tf.Tensor] = {}
        calc_gross = tf.subtract(self.total_revenue, self.cost_of_revenue)
        errs["gross_profit"] = tf.abs(tf.subtract(self.gross_profit, calc_gross))
        calc_op = tf.subtract(self.gross_profit, self.operating_expense)
        errs["operating_income"] = tf.abs(tf.subtract(self.operating_income, calc_op))
        calc_pre_tax = tf.add(
            tf.subtract(self.operating_income, self.interest_expense),
            self.other_income_expense,
        )
        errs["income_before_tax"] = tf.abs(
            tf.subtract(self.income_before_tax, calc_pre_tax)
        )
        calc_net = tf.subtract(self.income_before_tax, self.income_tax_expense)
        errs["net_income"] = tf.abs(tf.subtract(self.net_income, calc_net))
        return errs

    @property
    def is_valid(self) -> tf.Tensor:
        vals = [
            tf.convert_to_tensor(v, dtype=tf.float32) for v in self.validate().values()
        ]
        return tf.reduce_all(tf.math.less(tf.stack(vals), self.validity_threshold))

    def to_tensor_dict(self) -> Dict[str, tf.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
