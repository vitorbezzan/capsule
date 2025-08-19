"""Tests for Regression capsules."""

import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_diabetes, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from capsule import RegressionCapsule


@pytest.fixture
def trained_single_target_model_data():
    """Create a single-target regression model."""
    data = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@pytest.fixture
def trained_multi_target_model_data():
    """Create a multi-target regression model."""
    n_samples = 100
    n_features = 5
    n_targets = 2

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        noise=0.1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = RandomForestRegressor(random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@pytest.fixture
def regression_capsule(trained_single_target_model_data):
    """Fixture to create a RegressionCapsule with trained data."""
    model, X_test, y_test = trained_single_target_model_data
    return RegressionCapsule(model, X_test, y_test)


def test_single_target_diabetes(trained_single_target_model_data):
    """Test RegressionCapsule with single-target diabetes data."""
    model, X_test, y_test = trained_single_target_model_data
    capsule = RegressionCapsule(model=model, X_test=X_test, y_test=y_test)

    assert capsule.n_targets_ == 1
    assert not capsule.get_metrics(X_test).empty


def test_multi_target_diabetes(trained_multi_target_model_data):
    """Test RegressionCapsule with multi-target regression data."""
    model, X_test, y_test = trained_multi_target_model_data

    capsule0 = RegressionCapsule(
        model=model, X_test=X_test, y_test=y_test, target_index=0
    )
    capsule1 = RegressionCapsule(
        model=model, X_test=X_test, y_test=y_test, target_index=1
    )

    assert capsule0.n_targets_ == 2
    assert not capsule0.get_metrics(X_test).empty

    assert capsule1.n_targets_ == 2
    assert not capsule1.get_metrics(X_test).empty


def test_scatter_plot(regression_capsule):
    """Test the scatter method of RegressionPlots."""
    capsule = regression_capsule
    ax = capsule.plots.scatter()

    assert ax.get_xlabel() == "True Values"
    assert ax.get_ylabel() == "Predicted Values"

    lines = ax.get_lines()
    assert any(line.get_label() == "Reference Line" for line in lines)

    plt.close(ax.figure)


def test_residuals_plot(regression_capsule):
    """Test the residuals method of RegressionPlots."""
    capsule = regression_capsule
    ax = capsule.plots.residuals_plot()

    assert ax.get_xlabel() == "Predicted Values"
    assert ax.get_ylabel() == "Residuals (True - Predicted)"

    plt.close(ax.figure)


def test_std_residuals_hist(regression_capsule):
    """Test the standardized residuals histogram of RegressionPlots."""
    capsule = regression_capsule
    ax = capsule.plots.residuals_hist()

    assert ax.get_xlabel() == "Residuals"
    assert ax.get_ylabel() == "Frequency"

    assert len(ax.patches) > 0

    plt.close(ax.figure)
