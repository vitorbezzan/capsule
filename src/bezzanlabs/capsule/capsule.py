"""
Base capsule source code.
"""
import logging
import copy
from .proto import Classifier, Regressor
from datetime import datetime

logger = logging.getLogger(__name__)


class Capsule(object):
    """
    Defines the base behavior for a Capsule object.
    """
    __register_datetime: datetime

    def __init__(self, pipeline_or_model: Classifier | Regressor):
        """
        Constructor for Capsule class.

        Args:
            pipeline_or_model: A pipeline, model or model-like object that implements
            predict and/or predict_proba methods.
        """
        self.__pipeline_or_model = copy.deepcopy(pipeline_or_model)

    @property
    def pipeline_or_model(self) -> Classifier | Regressor:
        """
        Returns the pipeline, model or model-like object that implements predict or
        predict_proba methods.
        """
        return self.__pipeline_or_model

    @property
    def register_datetime(self) -> datetime:
        """
        Returns the datetime object that represents the registration datetime.
        """
        if not hasattr(self, "__register_datetime"):
            raise RuntimeError("Capsule not registered.")

        return self.__register_datetime

    def register(self, datetime_register: datetime | None = None):
        """
        Registers the capsule, aka. sets a starting point for the capsule's lifecycle.

        Args:
            datetime_register: A datetime object that represents the registration
            datetime.
        """
        if hasattr(self, "register_datetime"):
            raise RuntimeError("Capsule already registered.")

        if datetime_register is None:
            self.__register_datetime = datetime.now()
        else:
            self.__register_datetime = datetime_register

        logger.debug(f"Capsule registered at {self.__register_datetime}.")
