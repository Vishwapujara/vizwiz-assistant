import importlib

import vizwiz

SUBPACKAGES = ["data", "train", "eval", "quantize", "serve", "bench"]


def test_version_is_set():
    assert vizwiz.__version__


def test_subpackages_import():
    for name in SUBPACKAGES:
        importlib.import_module(f"vizwiz.{name}")
