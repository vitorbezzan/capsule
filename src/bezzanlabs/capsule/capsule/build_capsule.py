"""
Function to create capsules.
"""
from ..proto import Classifier, Regressor, model_types
from ..types import Pipeline
from .capsule import Capsule
from .classifier import ClassifierCapsule
from .regressor import RegressorCapsule


def build_capsule(model_object: model_types, **kwargs) -> Capsule:
    """
    Builds Capsule for a given model.

    Args:
        model_object: Model object containing some scikit-learn signature.
        kwargs: Specific arguments to pass to Capsule constructors.
    """
    model = (
        model_object.steps[-1][1]
        if isinstance(model_object, Pipeline)
        else model_object
    )

    if isinstance(model, Classifier):
        return ClassifierCapsule(model_object, **kwargs)
    elif isinstance(model, Regressor):
        return RegressorCapsule(model_object, **kwargs)

    raise ValueError("Unsupported model type")
