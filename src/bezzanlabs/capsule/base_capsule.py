"""
Defines a storing media for all Capsules.
"""
from dataclasses import dataclass
from .types import Inputs, Predictions, Pipeline
from .proto import Classifier, Regressor


@dataclass
class BaseCapsule:
    """
    Stores information required for all Capsules.
    """
    model: Classifier | Regressor | Pipeline
    X_train: Inputs
    y_train: Predictions
    X_test:  Inputs
    y_test: Predictions

    def __setstate__(self, state: dict):
        raise NotImplementedError

    def __getstate__(self) -> dict:
        """

        """

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    def __post_init__(self):
        """
        Post-init the object.
        """
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise RuntimeError("Number of X columns mismatch between train and test.")

        if self.y_train.shape[1] != self.y_test.shape[1]:
            raise RuntimeError("Number of y columns mismatch between train and test.")

        self._n_variables = self.X_train.shape[1]
        self._n_outputs = self.y_train.shape[1]
        self._model_type = _check_last_step(self.model)


def _check_last_step(_object: Classifier | Regressor | Pipeline) -> str:
    """
    Checks type of model. If it is a pipeline, returns type of last step.
    """
    model = _object.steps[-1][1] if isinstance(_object, Pipeline) else _object
    if isinstance(model, Regressor):
        return "regressor"

    return "classifier"
