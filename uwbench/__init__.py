"""UWBench: reference evaluation protocol for underwater semantic communications."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uwbench")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"
