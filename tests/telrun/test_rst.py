import os

import pytest

from pyscope.telrun import rst


def test_rst(tmp_path):
    os.mkdir(tmp_path / "my_test")

    print("done")


if __name__ == "__main__":
    test_rst("./")
