"""Smoke test: the platepu package is installed and importable."""

import platepu


def test_import_platepu() -> None:
    """The package imports and resolves to the installed platepu module."""
    assert platepu.__name__ == "platepu"
