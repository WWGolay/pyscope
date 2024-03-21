import logging
from pathlib import Path

import pytest

from pyscope import logger
from pyscope.telrun.synctools import sync_manager


def test_sync_manager():
    logger.setLevel(logging.DEBUG)
    logging.getLogger("paramiko").setLevel(logging.DEBUG)
    logging.getLogger("paramiko").addHandler(logging.StreamHandler())
    logger.addHandler(logging.StreamHandler())

    sync_manager(config="./tests/bin/sync.cfg", do_once=True)


if __name__ == "__main__":
    test_sync_manager()
