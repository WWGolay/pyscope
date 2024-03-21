import pytest

from pyscope.telrun import init_telrun_dir


def test_init_telrun_dir(tmp_path):
    init_telrun_dir(tmp_path / "my_test")
    assert (tmp_path / "my_test").is_dir()
