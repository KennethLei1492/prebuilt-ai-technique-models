"""pytest conftest: expose the shared fixtures module to tests under any import mode.

The shared constants, fakes and fixtures live in ``td_fixtures.py`` (a name that is unique
across the repository). Adding this directory to ``sys.path`` lets test modules import it
under ``--import-mode=importlib`` as well as the default prepend mode.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from td_fixtures import *  # noqa: E402,F401,F403 - re-export fixtures and helpers
