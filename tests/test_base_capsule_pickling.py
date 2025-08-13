"""Tests for BaseCapsule pickling and unpickling functionality."""

import os
import pickle
import pytest
from unittest.mock import patch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from capsule.base import ClassificationCapsule


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    return X, y


@pytest.fixture
def trained_model_and_data(sample_data):
    """Fixture providing a trained model and test data."""
    X, y = sample_data

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@pytest.fixture
def capsule_instance(trained_model_and_data):
    """Fixture providing a BaseCapsule instance."""
    model, X_test, y_test = trained_model_and_data
    return ClassificationCapsule(model=model, X_test=X_test, y_test=y_test)


class TestBaseCapsulePickling:
    """Test class for BaseCapsule pickling functionality."""

    def test_pickle_without_encryption_key(self, capsule_instance):
        """Test pickling and unpickling without encryption key."""
        with patch.dict(os.environ, {}, clear=True):
            pickled_data = pickle.dumps(capsule_instance)

            unpickled_capsule = pickle.loads(pickled_data)

            # Verify the unpickled capsule has the same attributes
            assert np.array_equal(unpickled_capsule.X_test_, capsule_instance.X_test_)
            assert np.array_equal(unpickled_capsule.y_test_, capsule_instance.y_test_)
            assert unpickled_capsule.n_features_ == capsule_instance.n_features_
            assert unpickled_capsule.n_targets_ == capsule_instance.n_targets_

            # Verify the model still works
            predictions_original = capsule_instance.predict(capsule_instance.X_test_)
            predictions_unpickled = unpickled_capsule.predict(unpickled_capsule.X_test_)
            assert np.array_equal(predictions_original, predictions_unpickled)

    def test_pickle_with_encryption_key(self, capsule_instance):
        """Test pickling and unpickling with encryption key."""
        encryption_key = "my_secret_key_123456789012345678"  # 32 chars for AES-256

        with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
            pickled_data = pickle.dumps(capsule_instance)

            unpickled_capsule = pickle.loads(pickled_data)

            assert np.array_equal(unpickled_capsule.X_test_, capsule_instance.X_test_)
            assert np.array_equal(unpickled_capsule.y_test_, capsule_instance.y_test_)
            assert unpickled_capsule.n_features_ == capsule_instance.n_features_
            assert unpickled_capsule.n_targets_ == capsule_instance.n_targets_

            predictions_original = capsule_instance.predict(capsule_instance.X_test_)
            predictions_unpickled = unpickled_capsule.predict(unpickled_capsule.X_test_)

            assert np.array_equal(predictions_original, predictions_unpickled)

    def test_pickle_with_key_then_unpickle_without_key_fails(self, capsule_instance):
        """Test that unpickling encrypted data without key raises RuntimeError."""
        encryption_key = "my_secret_key_123456789012345678"

        with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
            pickled_data = pickle.dumps(capsule_instance)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                RuntimeError,
                match="CAPSULE_KEY not set. Cannot unpickle encrypted capsule.",
            ):
                pickle.loads(pickled_data)

    def test_pickle_with_key_then_unpickle_with_wrong_key_fails(self, capsule_instance):
        """Test that unpickling with wrong key fails."""
        encryption_key = "my_secret_key_123456789012345678"
        wrong_key = "my_secret_key_123456789012345679"

        with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
            pickled_data = pickle.dumps(capsule_instance)

        with patch.dict(os.environ, {"CAPSULE_KEY": wrong_key}):
            with pytest.raises(Exception):  # Should raise cryptography exception
                pickle.loads(pickled_data)

    def test_pickle_without_key_then_unpickle_with_key_works(self, capsule_instance):
        """Test that data pickled without key can be unpickled with key present."""
        encryption_key = "my_secret_key_123456789012345678"

        with patch.dict(os.environ, {}, clear=True):
            pickled_data = pickle.dumps(capsule_instance)

        with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
            unpickled_capsule = pickle.loads(pickled_data)

            assert np.array_equal(unpickled_capsule.X_test_, capsule_instance.X_test_)
            assert np.array_equal(unpickled_capsule.y_test_, capsule_instance.y_test_)

    def test_getstate_without_key(self, capsule_instance):
        """Test __getstate__ method without encryption key."""
        with patch.dict(os.environ, {}, clear=True):
            state = capsule_instance.__getstate__()

            assert "data" in state
            assert "nonce" not in state

            unpickled_dict = pickle.loads(state["data"])
            assert "X_test_" in unpickled_dict
            assert "y_test_" in unpickled_dict
            assert "model_" in unpickled_dict

    def test_getstate_with_key(self, capsule_instance):
        """Test __getstate__ method with encryption key."""
        encryption_key = "my_secret_key_123456789012345678"

        with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
            state = capsule_instance.__getstate__()

            assert "data" in state
            assert "nonce" in state

            assert len(state["nonce"]) == 12

            with pytest.raises(Exception):
                pickle.loads(state["data"])

    def test_setstate_without_key(self, capsule_instance):
        """Test __setstate__ method without encryption key."""
        with patch.dict(os.environ, {}, clear=True):
            state = capsule_instance.__getstate__()

            new_capsule = ClassificationCapsule.__new__(ClassificationCapsule)
            new_capsule.__setstate__(state)

            assert np.array_equal(new_capsule.X_test_, capsule_instance.X_test_)
            assert np.array_equal(new_capsule.y_test_, capsule_instance.y_test_)
            assert new_capsule.n_features_ == capsule_instance.n_features_

    def test_setstate_with_key(self, capsule_instance):
        """Test __setstate__ method with encryption key."""
        encryption_key = "my_secret_key_123456789012345678"

        with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
            state = capsule_instance.__getstate__()

            new_capsule = ClassificationCapsule.__new__(ClassificationCapsule)
            new_capsule.__setstate__(state)

            assert np.array_equal(new_capsule.X_test_, capsule_instance.X_test_)
            assert np.array_equal(new_capsule.y_test_, capsule_instance.y_test_)
            assert new_capsule.n_features_ == capsule_instance.n_features_

    def test_encryption_key_length_validation(self, capsule_instance):
        """Test that various encryption key lengths work correctly."""
        test_keys = [
            "0123456789101112",  # 16 chars (128-bit)
            "012345678910111213141516",  # 24 chars (192-bit)
            "01234567891011121314151617181920",  # 32 chars (256-bit)
        ]

        for key in test_keys:
            with patch.dict(os.environ, {"CAPSULE_KEY": key}):
                pickled_data = pickle.dumps(capsule_instance)
                unpickled_capsule = pickle.loads(pickled_data)

                assert np.array_equal(
                    unpickled_capsule.X_test_, capsule_instance.X_test_
                )

    def test_multiple_pickle_unpickle_cycles(self, capsule_instance):
        """Test multiple pickle/unpickle cycles maintain data integrity."""
        encryption_key = "my_secret_key_123456789012345678"

        current_capsule = capsule_instance

        for i in range(3):
            with patch.dict(os.environ, {"CAPSULE_KEY": encryption_key}):
                pickled_data = pickle.dumps(current_capsule)
                current_capsule = pickle.loads(pickled_data)

                assert np.array_equal(current_capsule.X_test_, capsule_instance.X_test_)
                assert np.array_equal(current_capsule.y_test_, capsule_instance.y_test_)
                assert current_capsule.n_features_ == capsule_instance.n_features_

                predictions_original = capsule_instance.predict(
                    capsule_instance.X_test_
                )
                predictions_current = current_capsule.predict(current_capsule.X_test_)
                assert np.array_equal(predictions_original, predictions_current)


class TestBaseCapsuleEnvironmentVariables:
    """Test class for environment variable handling."""

    def test_capsule_key_environment_variable_precedence(self, capsule_instance):
        """Test that CAPSULE_KEY environment variable is properly used."""
        key1 = "my_secret_key_123456789012345678"
        key2 = "my_secret_key_123456789012345679"

        # Set first key and pickle
        with patch.dict(os.environ, {"CAPSULE_KEY": key1}):
            state1 = capsule_instance.__getstate__()

        # Set second key and pickle
        with patch.dict(os.environ, {"CAPSULE_KEY": key2}):
            state2 = capsule_instance.__getstate__()

        # States should be different (different encryption)
        assert state1["data"] != state2["data"]
        assert state1["nonce"] != state2["nonce"]

        # But both should decrypt correctly with their respective keys
        new_capsule1 = ClassificationCapsule.__new__(ClassificationCapsule)
        new_capsule2 = ClassificationCapsule.__new__(ClassificationCapsule)

        with patch.dict(os.environ, {"CAPSULE_KEY": key1}):
            new_capsule1.__setstate__(state1)

        with patch.dict(os.environ, {"CAPSULE_KEY": key2}):
            new_capsule2.__setstate__(state2)

        # Both should have the same data as original
        assert np.array_equal(new_capsule1.X_test_, capsule_instance.X_test_)
        assert np.array_equal(new_capsule2.X_test_, capsule_instance.X_test_)
