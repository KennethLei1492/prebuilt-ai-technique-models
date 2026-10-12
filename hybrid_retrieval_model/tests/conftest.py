"""Shared fixtures: a small engineering corpus with identifiers, clauses and code.

Helper classes live in ``hr_fixtures.py`` (a module name unique in the repository);
adding this directory to ``sys.path`` lets test modules import it under
``--import-mode=importlib`` as well as the default prepend mode.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hr_fixtures import *  # noqa: E402,F401,F403 - re-export helpers

CORPUS: list[dict] = [
    {
        "id": "spec-pump",
        "text": (
            "# Pump specification PS-4471 rev B\n\n"
            "## 4.1 Casing\nThe casing shall be ASTM A216 WCB carbon steel with a corrosion allowance of 3 mm. "
            "Design pressure 16 bar at 120 degC.\n\n"
            "## 4.2 Impeller\nThe impeller material shall be duplex stainless steel ASTM A890 Grade 4A (CD3MN). "
            "Impeller balance grade G6.3 per ISO 21940.\n\n"
            "## 4.3 Shaft seal\nMechanical seal per API 682 Category 2, plan 11 flush."
        ),
        "metadata": {"title": "Pump specification PS-4471", "type": "specification", "revision": "B", "severity": 2},
    },
    {
        "id": "incident-7",
        "text": (
            "# Incident INC-4821: cooling water pump cavitation\n\n"
            "Severity 2 incident on the cooling water loop. Pump P-203 showed cavitation noise and a 15% head loss after "
            "the suction strainer clogged. Root cause: NPSH available fell below NPSH required when the strainer differential "
            "exceeded 0.4 bar. Corrective action: differential pressure alarm at 0.25 bar and weekly strainer inspection."
        ),
        "metadata": {"title": "Incident INC-4821", "type": "incident", "severity": 2, "service": "cooling"},
    },
    {
        "id": "runbook-retry",
        "text": (
            "# Client retry configuration\n\n"
            "Retries are controlled by retry.max_attempts, retry.base_delay_ms and retry.jitter. "
            "ERR-1203 is raised when the heartbeat from the field bus is lost for longer than the configured timeout.\n\n"
            "```python\nclient = Client(max_attempts=5, base_delay_ms=200)\nclient.getUserName()\n```"
        ),
        "metadata": {"title": "Retry runbook", "type": "runbook", "severity": 3},
    },
    {
        "id": "datasheet-steel",
        "text": (
            "# Material datasheet: duplex stainless steels\n\n"
            "Duplex grades such as 2205 (UNS S31803 / S32205) combine austenite and ferrite. Yield strength 450 MPa minimum, "
            "tensile 620 MPa, PREN above 34 gives good resistance to chloride pitting and crevice corrosion. "
            "Cast equivalents include ASTM A890 Grade 4A."
        ),
        "metadata": {"title": "Duplex datasheet", "type": "datasheet", "severity": 3},
    },
    {
        "id": "memo-schedule",
        "text": (
            "# Project schedule memo\n\n"
            "The commissioning window moves to the third quarter. Vendor documentation for the compressors is late and the "
            "electrical contractor needs the single line diagram before cable pulling can start."
        ),
        "metadata": {"title": "Schedule memo", "type": "memo", "severity": 3},
    },
]


@pytest.fixture
def corpus() -> list[dict]:
    return [dict(d) for d in CORPUS]
