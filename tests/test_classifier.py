"""
Tests classifier capsules.
"""
import pytest

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from bezzanlabs.capsule.capsule.build_capsule import build_capsule
from bezzanlabs.capsule.services.classifier_metrics import ClassifierMetrics


@pytest.fixture(scope="session")
def classifier_data():
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def classifier_model(classifier_data):
    X_train, _, y_train, _ = classifier_data
    model = ExtraTreesClassifier().fit(X_train, y_train)

    return model


def test_classifier_capsule(classifier_data, classifier_model):
    X_train, X_test, y_train, y_test = classifier_data
    model = classifier_model

    capsule = build_capsule(
        model,
        X_test=X_test,
        y_test=y_test,
    )

    capsule.add_service(ClassifierMetrics("simple_metrics"))
    assert capsule("simple_metrics", X=X_test, y=y_test)
