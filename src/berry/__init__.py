from __future__ import annotations

__all__ = ["__version__"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("berry")
except PackageNotFoundError:  # pragma: no cover
    # Fallback for environments where metadata isn't available.
    __version__ = "0.0.0"
