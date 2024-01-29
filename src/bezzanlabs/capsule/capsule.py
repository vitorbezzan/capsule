"""
Base capsule source code.
"""
import copy
import logging
import typing as tp
from gzip import compress, decompress
from os.path import expanduser
from pickle import dumps, loads

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from .datetime import get_utc_time
from .proto import Classifier, Regressor
from .types import Inputs, Pipeline, Predictions

logger = logging.getLogger(__name__)

Model: tp.TypeAlias = Classifier | Regressor | Pipeline


class Capsule(object):
    """
    Defines a Capsule for model run, encrypting and API exposing.
    """

    def __init__(self, name: str, model: Model | None, **kwargs) -> None:
        """
        Constructor for Capsule.

        Args:
            model: model object with .predict() and/or .predict_proba() methods.
            kwargs: Any other arguments to pass to capsule.
        """
        self._name = name
        self._model = copy.deepcopy(model)

        if isinstance(self._model, Regressor):
            self._model_type = "regression"
        elif isinstance(self._model, Classifier):
            self._model_type = "classifier"
        else:
            self._model_type = "unknown"

        self._kwargs = kwargs
        self._datetime = get_utc_time(
            kwargs.get("error", False),
            kwargs.get("ntp_servers", None),
        )

    def __getstate__(self) -> dict:
        aes_key, state = self.get_bytes()

        state["capsule_key_dir"] = self._kwargs.get("capsule_key_dir", expanduser("~"))
        state["name"] = self._name

        with open(f"{state['capsule_key_dir']}/{state['name']}", "wb") as key_file:
            key_file.write(aes_key)

        return state

    def __setstate__(self, state: dict):
        capsule_dir = state.pop("capsule_key_dir")
        capsule_name = state.pop("name")

        try:
            with open(f"{capsule_dir}/{capsule_name}", "rb") as key_file:
                aes_key = key_file.read()
        except FileNotFoundError:
            logger.warning(
                "capsule_dir is not pointing to the right place. Using default."
            )

            try:
                with open(f"{expanduser('~')}/{capsule_name}", "rb") as key_file:
                    aes_key = key_file.read()
            except FileNotFoundError as error:
                logger.critical(
                    "Failed to read key for capsule.",
                    exc_info=True,
                    stack_info=True,
                )
                raise RuntimeError("Failed to read key for capsule.") from error

        self.__dict__.update(self.read_bytes(state, aes_key))

    def get_bytes(self) -> tuple[bytes, dict]:
        """
        Get bytes of Capsule, and its AES-256 encryption key.

        Returns:
            Tuple containing AES-256 key and bytes representation of encrypted object.
        """
        aes_key = get_random_bytes(32)
        __cipher = AES.new(aes_key, AES.MODE_EAX)

        ciphertext, tag = __cipher.encrypt_and_digest(compress(dumps(self.__dict__)))
        return aes_key, {"object": ciphertext, "tag": tag, "n": __cipher.nonce}

    @staticmethod
    def read_bytes(data: dict, aes_key: bytes) -> dict:
        """
        Returns Capsule contents, given its encrypted bytes representation and a correct
        AES-256 key.

        Returns:
            Dictionary containing object representation.
        """
        __cipher = AES.new(aes_key, AES.MODE_EAX, data["n"])

        return loads(
            decompress(__cipher.decrypt_and_verify(data["object"], data["tag"]))
        )

    def predict(self, X: Inputs) -> Predictions | None:
        if self._model is not None:
            try:
                return self._model.predict(X)
            except AttributeError as error:
                logger.critical(
                    ".predict() is not available. Failing.",
                    exc_info=True,
                    stack_info=True,
                )
                raise RuntimeError("Critical failure in .predict().") from error

        return None

    def predict_proba(self, X: Inputs) -> Predictions | None:
        """
        Returns probabilities for class in classifier models, and returns predictions
        for regressor models.
        """
        if self._model is not None:
            try:
                return self._model.predict_proba(X)  # type: ignore
            except AttributeError:
                logger.warning(
                    ".predict_proba() is not available for this model object.",
                    exc_info=True,
                )
                return self.predict(X)

        return None
