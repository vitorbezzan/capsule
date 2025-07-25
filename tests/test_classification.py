"""**Tests for Classification capsules.**

This module contains comprehensive tests for the `ClassificationCapsule` class,
including both binary and multiclass classification scenarios. It uses scikit-learn
datasets and models to create realistic test scenarios.

The tests verify the functionality of classification capsules by:
- Creating trained models with both iris (multiclass) and breast cancer (binary) datasets
- Testing the capsule initialization and basic properties
- Validating metrics generation functionality

Examples:
    Run the tests using pytest:

    ```bash
    pytest tests/test_classification.py
    ```
"""

import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.capsule import ClassificationCapsule


@pytest.fixture
def trained_multiclass_model_data():
    """Create a trained multiclass classification model with test data.

    This fixture prepares a RandomForestClassifier trained on the iris dataset
    for multiclass classification testing. The dataset is split into training
    and testing sets with a fixed random state for reproducible results.

    Returns:
        A tuple containing the trained model, test features, and test labels.

    Examples:
        ```python
        def test_example(trained_multiclass_model_data):
            model, X_test, y_test = trained_multiclass_model_data
            # Use the trained model and test data
        ```
    """
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@pytest.fixture
def trained_binary_model_data():
    """Create a trained binary classification model with test data.

    This fixture prepares a RandomForestClassifier trained on the breast cancer
    dataset for binary classification testing. The dataset is split into training
    and testing sets with a fixed random state for reproducible results.

    Returns:
        A tuple containing the trained model, test features, and test labels.

    Examples:
        ```python
        def test_example(trained_binary_model_data):
            model, X_test, y_test = trained_binary_model_data
            # Use the trained model and test data
        ```
    """
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_multiclass(trained_multiclass_model_data):
    """Test ClassificationCapsule functionality with multiclass data.

    This test verifies that the ClassificationCapsule correctly handles
    multiclass classification scenarios using the iris dataset. It checks
    that the capsule properly identifies the number of classes and can
    generate non-empty metrics.

    Args:
        trained_multiclass_model_data: Fixture providing trained model and test data.

    Examples:
        The test validates:

        ```python
        capsule = ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)
        assert capsule.n_classes_ == 3  # Iris has 3 classes
        assert not capsule.get_metrics(X_test).empty  # Metrics are generated
        ```
    """
    model, X_test, y_test = trained_multiclass_model_data
    capsule = ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)

    assert capsule.n_classes_ == 3
    assert not capsule.get_metrics(X_test).empty


def test_binary(trained_binary_model_data):
    """Test ClassificationCapsule functionality with binary data.

    This test verifies that the ClassificationCapsule correctly handles
    binary classification scenarios using the breast cancer dataset. It checks
    that the capsule properly identifies the number of classes and can
    generate non-empty metrics.

    Args:
        trained_binary_model_data: Fixture providing trained model and test data.

    Examples:
        The test validates:

        ```python
        capsule = ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)
        assert capsule.n_classes_ == 2  # Binary classification
        assert not capsule.get_metrics(X_test).empty  # Metrics are generated
        ```
    """
    model, X_test, y_test = trained_binary_model_data
    capsule = ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)

    assert capsule.n_classes_ == 2
    assert not capsule.get_metrics(X_test).empty
