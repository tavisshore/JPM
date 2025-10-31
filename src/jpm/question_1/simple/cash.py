from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, cast

import tensorflow as tf

from src.jpm.question_1.misc import to_tensor


@dataclass
class CashFlow:
    cash_from_operations: tf.Tensor
    cash_from_investing: tf.Tensor
    cash_from_financing: tf.Tensor
    net_change_in_cash: tf.Tensor

    validity_threshold: tf.Tensor = field(
        default_factory=lambda: cast(tf.Tensor, tf.constant(1e-4, dtype=tf.float32))
    )

    def __post_init__(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, to_tensor(getattr(self, f)))

    def validate(self) -> Dict[str, tf.Tensor]:
        calc = tf.add(
            tf.add(self.cash_from_operations, self.cash_from_investing),
            self.cash_from_financing,
        )
        return {"net_change_in_cash": tf.abs(self.net_change_in_cash - calc)}

    @property
    def is_valid(self) -> tf.Tensor:
        vals = [
            tf.convert_to_tensor(v, dtype=tf.float32) for v in self.validate().values()
        ]
        return tf.reduce_all(tf.math.less(tf.stack(vals), self.validity_threshold))

    def to_tensor_dict(self) -> Dict[str, tf.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
