"""
Base capsule source code.
"""
import logging
from gzip import compress, decompress
from os.path import expanduser
from pickle import dumps, loads

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from .capsule_base import CapsuleBase

logger = logging.getLogger(__name__)


class SecureCapsule(CapsuleBase):
    """
    Defines a Capsule for model run, encrypting and API exposing.
    """

    def __getstate__(self) -> dict:
        aes_key, state = self.get_bytes()
        state["name"] = self.__capsule_name

        try:
            with open(
                f"{expanduser('~')}/.capsules/keys/{state['name']}", "wb"
            ) as key_file:
                key_file.write(aes_key)
        except FileNotFoundError as error:
            logger.critical("Error saving key file.")
            raise RuntimeError("Error saving key file.") from error

        return state

    def __setstate__(self, state: dict):
        capsule_name = state.pop("name")

        try:
            with open(
                f"{expanduser('~')}/.capsules/keys/{capsule_name}", "rb"
            ) as key_file:
                aes_key = key_file.read()
        except FileNotFoundError as error:
            logger.critical("Error saving key file.")
            raise RuntimeError("Error saving key file.") from error

        self.__dict__.update(self.read_bytes(state, aes_key))

    def get_bytes(self) -> tuple[bytes, dict]:
        """
        Get bytes of Capsule, and its AES-256 encryption key.

        Returns:
            Tuple containing AES-256 key and bytes representation of encrypted object.
        """
        aes_key = get_random_bytes(32)
        cipher = AES.new(aes_key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(compress(dumps(self.__dict__)))

        return aes_key, {"object": ciphertext, "tag": tag, "n": cipher.nonce}

    @staticmethod
    def read_bytes(data: dict, aes_key: bytes) -> dict:
        """
        Returns Capsule contents, given its encrypted bytes representation and a correct
        AES-256 key.

        Returns:
            Dictionary containing object representation.
        """
        cipher = AES.new(aes_key, AES.MODE_EAX, data["n"])

        return loads(decompress(cipher.decrypt_and_verify(data["object"], data["tag"])))

