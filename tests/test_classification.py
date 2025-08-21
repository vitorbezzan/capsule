"""Tests for ClassificationCapsule."""

import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from capsule import ClassificationCapsule


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


@pytest.fixture
def binary_classification_capsule(trained_binary_model_data):
    """Fixture to create a ClassificationCapsule with binary classification data."""
    model, X_test, y_test = trained_binary_model_data
    return ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)


@pytest.fixture
def multiclass_classification_capsule(trained_multiclass_model_data):
    """Fixture to create a ClassificationCapsule with multiclass classification data."""
    model, X_test, y_test = trained_multiclass_model_data
    return ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)


def test_roc_curve_binary(binary_classification_capsule):
    """Test the roc_curve method for binary classification."""
    capsule = binary_classification_capsule
    ax = capsule.plots.roc_curve()

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    assert ax.get_title() == "Receiver Operating Characteristic"

    lines = ax.get_lines()
    assert any(line.get_label() == "Random classifier" for line in lines)

    assert any("ROC curve" in line.get_label() for line in lines)

    assert any("AUC" in line.get_label() for line in lines)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim[0] == 0.0
    assert xlim[1] == 1.0
    assert ylim[0] == 0.0

    plt.close(ax.figure)


def test_roc_curve_multiclass(multiclass_classification_capsule):
    """Test the roc_curve method for multiclass classification."""
    capsule = multiclass_classification_capsule
    ax = capsule.plots.roc_curve()

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    assert ax.get_title() == "Receiver Operating Characteristic"

    lines = ax.get_lines()
    assert any(line.get_label() == "Random classifier" for line in lines)

    roc_lines = [line for line in lines if "ROC curve for class" in line.get_label()]
    assert len(roc_lines) == 3

    for line in roc_lines:
        assert "AUC" in line.get_label()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim[0] == 0.0
    assert xlim[1] == 1.0
    assert ylim[0] == 0.0

    plt.close(ax.figure)


def test_pr_curve_binary(binary_classification_capsule):
    """Test the pr_curve method for binary classification."""
    capsule = binary_classification_capsule
    ax = capsule.plots.pr_curve()

    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"
    assert ax.get_title() == "Precision-Recall Curve"

    lines = ax.get_lines()
    assert any(line.get_label() == "No Skill" for line in lines)
    assert any("PR curve" in line.get_label() for line in lines)
    assert any("AP" in line.get_label() for line in lines)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim[0] == 0.0
    assert xlim[1] == 1.0
    assert ylim[0] == 0.0

    plt.close(ax.figure)


def test_pr_curve_multiclass(multiclass_classification_capsule):
    """Test the pr_curve method for multiclass classification."""
    capsule = multiclass_classification_capsule
    ax = capsule.plots.pr_curve()

    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"
    assert ax.get_title() == "Precision-Recall Curve"

    lines = ax.get_lines()
    pr_lines = [line for line in lines if "PR curve for class" in line.get_label()]
    assert len(pr_lines) == 3

    for line in pr_lines:
        assert "AP" in line.get_label()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim[0] == 0.0
    assert xlim[1] == 1.0
    assert ylim[0] == 0.0

    plt.close(ax.figure)
