"""
Base service definition + some basic services
"""
import abc

from ..capsule.capsule import Capsule, ServiceProto
from ..types import TServiceReturn
from .exceptions import ServiceRunError


class Service(abc.ABC, ServiceProto):
    """
    Defines a Service.
    """

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def start(self, capsule_object: Capsule) -> "Service":
        """
        Checks if the Service is compatible with `capsule_object`. Also sets specific
        service configurations on the Capsule if needed. Also runs any pre-work for a
        given Service.
        """

    @abc.abstractmethod
    def _run(self, capsule_object: Capsule, **kwargs) -> TServiceReturn:
        """
        Implementation of Service logic.
        """

    def run(self, capsule_object: Capsule, **kwargs) -> TServiceReturn:
        """
        Runs Service, and returns data for the run.

        Returns (maybe empty) TServiceReturn object if it is possible to run the service
        with 'success', and needs to raise an Exception if the necessary data/parameters
        are not available during runtime or if anything throws an error inside.
        """
        try:
            return self._run(capsule_object, **kwargs)
        except Exception as exception:
            print(exception)
            raise ServiceRunError(
                "Service threw an error during _run()."
            ) from exception
