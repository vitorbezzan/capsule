"""
Base capsule source code.
"""
import copy
import logging
import typing as tp
from datetime import datetime

from .datetime import get_utc_time
from .proto import Classifier, Regressor
from .types import Inputs, Pipeline, Predictions

logger = logging.getLogger(__name__)

Model: tp.TypeAlias = Classifier | Regressor | Pipeline


class CapsuleBase(object):
    """
    Defines base behavior for all capsule types.
    """

    def __init__(self, capsule_name: str, model: Model | None, **kwargs) -> None:
        """
        Constructor for Capsule.

        Args:
            capsule_name: Capsule name to use for internal purposes and registering it
            on a machine.
            model: model object with .predict() and/or .predict_proba() methods.
            kwargs: Any other arguments to pass to Capsule and their internal methods.`
        """
        self.__capsule_name = capsule_name
        self.__model = copy.deepcopy(model)
        self.__kwargs = kwargs
        self.__datetime = get_utc_time(
            error_if_not_available=kwargs.get("error_if_not_available", False),
            ntp_servers=kwargs.get("ntp_servers", None),
        )

    @property
    def capsule_name(self) -> str:
        return self.__capsule_name

    @property
    def datetime(self) -> datetime:
        return self.__datetime

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns values (or classes) for a specific X.
        """
        if isinstance(self.__model, (Classifier, Regressor)):
            logger.debug(f"Generating prediction for model {self.capsule_name}.")
            return self.__model.predict(X)

        raise ValueError("Model in capsule is not a classifier or regressor.")

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Returns probabilities for class in classifier models.
        """
        if isinstance(self.__model, Classifier):
            logger.debug(f"Generating prediction_proba for model {self.capsule_name}.")
            return self.__model.predict_proba(X)

        raise ValueError("Model in capsule is not a classifier.")
