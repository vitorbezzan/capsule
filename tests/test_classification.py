"""Tests for ClassificationCapsule."""

import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.capsule import ClassificationCapsule


@pytest.fixture
def trained_multiclass_model_data():
    """Fixture for trained multiclass model data."""
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@pytest.fixture
def trained_binary_model_data():
    """Fixture for trained binary model data."""
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_multiclass(trained_multiclass_model_data):
    """Test multiclass classification."""
    model, X_test, y_test = trained_multiclass_model_data
    capsule = ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)

    assert capsule.n_classes_ == 3
    assert not capsule.get_metrics(X_test).empty


def test_binary(trained_binary_model_data):
    """Test binary classification."""
    model, X_test, y_test = trained_binary_model_data
    capsule = ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)

    assert capsule.n_classes_ == 2
    assert not capsule.get_metrics(X_test).empty
