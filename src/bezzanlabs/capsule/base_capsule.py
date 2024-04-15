"""
Defines a storing media for all Capsules.
"""
import logging
from gzip import compress, decompress
from pickle import dumps, loads

from .proto import Classifier, Regressor
from .types import Inputs, Pipeline, Predictions


class BaseCapsule:
    """
    Stores information and model for all Capsules.
    """
    def __init__(
        self,
        model: Classifier | Regressor | Pipeline,
        logging_level: int = logging.INFO
    ):
        """
        Constructor for BaseCapsule.

        Args:
            model: Model to encapsulate. Can be a Regressor, Classifier or a Pipeline.
            For a pipeline, the last step should be a Regressor or Classifier.
            logging_level: Level of logging to use for debug messages. Default is INFO.
        """
        self._model, self._model_type = model, _check_last_step(model)
        self._logging_level = logging_level

    def __setstate__(self, state: dict):
        self.__dict__.update(loads(decompress(state["data"])))

    def __getstate__(self) -> dict:
        return {"data": compress(dumps(self.__dict__))}

    @property
    def model_type(self) -> str:
        """
        Returns model type.
        """
        return self._model_type

    def set_level(self, logging_: int) -> None:

    def predict(self, X: Inputs) -> Predictions:
        """
        Encapsulates predict method of model, adding some checks and debug logging.
        """

        return self._model.predict(X)

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Encapsulates predict_proba method of model, adding some checks and logging.
        """
        if self.model_type == "regressor":
            raise RuntimeError("Regressors do not have predict_proba method.")

        return self._model.predict_proba(X)


def _check_last_step(_object: Classifier | Regressor | Pipeline) -> str:
    """
    Checks type of model. If it is a pipeline, returns type of last step.

    Raises:
        ValueError: if model type is not Regressor or Classifier.
    """
    model = _object.steps[-1][1] if isinstance(_object, Pipeline) else _object
    if isinstance(model, Regressor):
        return "regressor"
    elif isinstance(model, Classifier):
        return "classifier"

    raise ValueError("Unsupported model type.")
