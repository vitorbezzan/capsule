"""
Defines base behavior for Capsules.
"""
import copy
import datetime
import gzip
import typing as tp
import uuid
from pathlib import Path

import dill
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from ..proto import model_types
from ..types import Actuals, Inputs, TServiceReturn


@tp.runtime_checkable
class ServiceProto(tp.Protocol):
    """
    Protocol to specify a Service API.
    """

    name: str

    def start(self, capsule_object: "Capsule") -> "ServiceProto":
        """ """

    def run(self, capsule_object: "Capsule", **kwargs) -> TServiceReturn:
        """ """


class Capsule(BaseEstimator):
    """
    Defines a base behavior for a Capsule.

    Capsules cannot be created using this class. Please view `RegressorCapsule`,
    `ClassifierCapsule` and `build_capsule` definitions for more details.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Capsule:
            raise TypeError()  # pragma: no cover
        return object.__new__(cls)

    def __init__(
        self,
        model: model_types,
        X_test: Inputs,
        y_test: Actuals,
        **kwargs,
    ):
        """
        Constructor for Capsule.

        Args:
            model: Model object to pair with Capsule.
            X_test: Data used for testing the model.
            y_test: Ground truth used for testing the model.
        """
        self.__model = model

        _ = check_array(X_test, force_all_finite="allow-nan")
        self.__data = {
            "X_test": X_test,
            "y_test": y_test,
        }

        _uuid = str(uuid.uuid4())
        self.__metadata = {
            "name": kwargs.get("name", _uuid),
            "id": kwargs.get("id", _uuid),
            "creation_date": str(datetime.datetime.now()),
        }

        self.__services: dict[str, ServiceProto] = {}

    def __getstate__(self) -> dict[str, bytes]:
        """
        Pickles (and optionally encrypts) a Capsule object.
        """
        key_path = Path.home() / ".capsule" / f"{self.__metadata['name']}"

        if key_path.is_file():
            with open(key_path, "rb") as key_file:
                capsule_key = key_file.read()
        else:
            capsule_key = b""

        contents = {
            "id": self.__metadata["name"],
            "data": gzip.compress(
                dill.dumps(self.__dict__),
                compresslevel=9,
            ),
        }

        if not capsule_key:
            return contents

        cipher = AES.new(capsule_key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(dill.dumps(contents))

        return {
            "id": self.__metadata["name"],
            "object": ciphertext,
            "tag": tag,
            "n": cipher.nonce,
        }

    def __setstate__(self, state: dict[str, bytes]) -> None:
        """
        Un-pickles (and optionally decrypts) a Capsule object.
        """
        key_path = Path.home() / ".capsule" / f"{state.pop("id")}"  # type: ignore

        if key_path.is_file():
            with open(key_path, "rb") as key_file:
                capsule_key = key_file.read()
        else:
            capsule_key = b""

        if not capsule_key:
            try:
                self.__dict__.update(dill.loads(gzip.decompress(state["data"])))
            except KeyError as error:
                raise RuntimeError(
                    "Unknown format for capsule. Maybe you forgot the key?"
                ) from error
        else:
            cipher = AES.new(capsule_key, AES.MODE_EAX, state["n"])
            contents = dill.loads(
                cipher.decrypt_and_verify(state["object"], state["tag"])
            )

            self.__dict__.update(dill.loads(gzip.decompress(contents["data"])))

    def __getitem__(self, key: str) -> Inputs | Actuals:
        return self.__data[key]

    @property
    def data(self) -> dict:
        return self.__data

    @property
    def metadata(self) -> dict[str, str]:
        return copy.deepcopy(self.__metadata)  # pragma: no cover

    @property
    def model(self) -> model_types:
        return self.__model

    def fit(self, X: Inputs, **fit_params) -> None:
        raise NotImplementedError()  # pragma: no cover

    def add_service(self, service: ServiceProto) -> None:
        """
        Adds a service to the Capsule.

        Args:
            service: Service to be added in the Capsule object.

        Raises:
            RuntimeError: If service is not compatible with current Capsule object.
        """
        self.__services[service.name] = copy.deepcopy(service.start(self))

    def __call__(self, service_name: str, **kwargs) -> TServiceReturn:
        """
        Runs a service in the Capsule.

        Args:
            service_name: Name of the service to be run.
            kwargs: Keyword params to pass to service runner.

        Raises:
            RuntimeError: If service is not present in the Capsule.
        """
        if service_name in self.__services:
            return self.__services[service_name].run(self, **kwargs)

        raise RuntimeError("Service not available in Capsule.")

    def create_key(self) -> None:
        """
        Creates a key for this Capsule, and saves it on home dir.
        """
        key_path = Path.home() / ".capsule" / f"{self.__metadata['name']}"
        with open(key_path, "wb") as capsule_key:
            capsule_key.write(get_random_bytes(32))
