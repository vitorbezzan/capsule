# flake8: noqa
"""
API entrypoint for the package.
"""
import logging
import sys
from os import makedirs
from os.path import exists, expanduser

__all__ = [
    "__package_name__",
    "__version__",
]

__package_name__ = "bezzanlabs.capsule"
__version__ = "0.0.1"


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    capsules_main = f"{expanduser('~')}/.capsules/"
    if exists(capsules_main):
        logger.info(f"{capsules_main} already exists.")
    else:
        makedirs(capsules_main)
        logger.info(f"{capsules_main} initialized and created with success.")

    capsules_dir = f"{capsules_main}/capsules/"
    makedirs(capsules_dir)

    capsules_keys = f"{capsules_main}/keys/"
    makedirs(capsules_keys)
