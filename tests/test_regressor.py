"""
Tests regressor capsules.
"""
import pytest
import dill

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from pathlib import Path

from bezzanlabs.capsule.capsule.build_capsule import build_capsule
from bezzanlabs.capsule.services.regressor_metrics import RegressorMetrics
from Crypto.Random import get_random_bytes


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=10000, n_features=20, n_informative=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def regression_model(regression_data):
    X_train, _, y_train, _ = regression_data
    model = ExtraTreesRegressor().fit(X_train, y_train)

    return model


@pytest.fixture(scope="session")
def regression_capsule(regression_data, regression_model):
    X_train, X_test, y_train, y_test = regression_data
    model = regression_model

    return build_capsule(
        model,
        X_test=X_test,
        y_test=y_test,
    )


def test_regression_capsule(regression_capsule):
    regression_capsule.add_service(RegressorMetrics("simple_metrics"))
    assert regression_capsule("simple_metrics")


def test_pickling(regression_capsule):
    with open("regressor.capsule", "wb") as save_file:
        dill.dump(regression_capsule, save_file)

    with open("regressor.capsule", "rb") as open_file:
        new_capsule = dill.load(open_file)

    assert new_capsule("simple_metrics")


def test_pickling_encrypted(regression_capsule):
    regression_capsule.create_key()

    with open("regressor.capsule", "wb") as save_file:
        dill.dump(regression_capsule, save_file)

    with open("regressor.capsule", "rb") as open_file:
        new_capsule = dill.load(open_file)

    assert new_capsule("simple_metrics")


def test_pickling_error(regression_capsule):
    key_path = Path.home() / ".capsule" / f"{regression_capsule.metadata['name']}"
    regression_capsule.create_key()

    with open("regressor.capsule", "wb") as save_file:
        dill.dump(regression_capsule, save_file)

    with open(key_path, "wb") as key_file:
        key_file.write(get_random_bytes(32))

    with pytest.raises(ValueError):
        with open("regressor.capsule", "rb") as open_file:
            dill.load(open_file)

    with open(key_path, "wb") as key_file:
        key_file.write(b"")

    with pytest.raises(RuntimeError):
        with open("regressor.capsule", "rb") as open_file:
            dill.load(open_file)
