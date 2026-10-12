"""A small unit parser used by the rule verifier.

It recognises quantities (``12.5 MPa``, ``3 kN``, ``24V``), maps unit tokens to
families with SI conversion factors, and reports unknown units and decimal
numbers that carry no unit. It is deliberately conservative: the goal is to
catch the mistakes that matter in engineering answers (missing or mixed units,
unknown unit tokens) with zero false positives on counts and identifiers.

Unit tokens are matched **case-sensitively first** so that SI prefixes keep
their meaning (``mW`` is a milliwatt, ``MW`` a megawatt, ``mPa`` a millipascal,
``MPa`` a megapascal). A case-insensitive lookup is used only for tokens whose
spelling is unambiguous once lower-cased (``bar``, ``PSI``, ``rpm``, ``kn``, ...);
tokens that would collide (``mn`` could be ``mN`` or ``MN``) resolve to the
spelling engineers overwhelmingly mean when they drop the case (``mpa`` ->
``MPa``, ``mm`` -> ``mm``) or stay unknown.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

#: canonical unit token -> (family, factor to the family's base unit). Keys are case-sensitive.
UNIT_TABLE: dict[str, tuple[str, float]] = {
    # length (m)
    "nm": ("length", 1e-9), "um": ("length", 1e-6), "µm": ("length", 1e-6), "mm": ("length", 1e-3), "cm": ("length", 1e-2),
    "m": ("length", 1.0), "km": ("length", 1e3), "Mm": ("length", 1e6), "in": ("length", 0.0254), "ft": ("length", 0.3048),
    # area (m2)
    "mm2": ("area", 1e-6), "cm2": ("area", 1e-4), "m2": ("area", 1.0), "km2": ("area", 1e6), "in2": ("area", 0.0254**2), "ft2": ("area", 0.3048**2),
    # volume (m3)
    "mm3": ("volume", 1e-9), "cm3": ("volume", 1e-6), "m3": ("volume", 1.0), "L": ("volume", 1e-3), "mL": ("volume", 1e-6), "dm3": ("volume", 1e-3),
    # mass (kg)
    "mg": ("mass", 1e-6), "g": ("mass", 1e-3), "kg": ("mass", 1.0), "t": ("mass", 1e3), "lb": ("mass", 0.45359237),
    # force (N)
    "N": ("force", 1.0), "mN": ("force", 1e-3), "kN": ("force", 1e3), "MN": ("force", 1e6), "lbf": ("force", 4.4482216),
    # pressure / stress (Pa)
    "Pa": ("pressure", 1.0), "mPa": ("pressure", 1e-3), "kPa": ("pressure", 1e3), "MPa": ("pressure", 1e6), "GPa": ("pressure", 1e9),
    "bar": ("pressure", 1e5), "mbar": ("pressure", 1e2), "psi": ("pressure", 6894.757), "ksi": ("pressure", 6.894757e6),
    # torque / energy (J; N·m treated as its own family)
    "nm_torque": ("torque", 1.0), "J": ("energy", 1.0), "mJ": ("energy", 1e-3), "kJ": ("energy", 1e3), "MJ": ("energy", 1e6),
    "Wh": ("energy", 3600.0), "kWh": ("energy", 3.6e6), "MWh": ("energy", 3.6e9),
    # power (W)
    "W": ("power", 1.0), "mW": ("power", 1e-3), "kW": ("power", 1e3), "MW": ("power", 1e6), "GW": ("power", 1e9), "hp": ("power", 745.7),
    # electrical
    "V": ("voltage", 1.0), "mV": ("voltage", 1e-3), "kV": ("voltage", 1e3), "MV": ("voltage", 1e6),
    "A": ("current", 1.0), "mA": ("current", 1e-3), "uA": ("current", 1e-6), "µA": ("current", 1e-6), "kA": ("current", 1e3),
    "ohm": ("resistance", 1.0), "mohm": ("resistance", 1e-3), "kohm": ("resistance", 1e3), "Mohm": ("resistance", 1e6), "Ω": ("resistance", 1.0),
    "F": ("capacitance", 1.0), "mF": ("capacitance", 1e-3), "uF": ("capacitance", 1e-6), "µF": ("capacitance", 1e-6), "nF": ("capacitance", 1e-9), "pF": ("capacitance", 1e-12),
    "H": ("inductance", 1.0), "mH": ("inductance", 1e-3), "uH": ("inductance", 1e-6), "µH": ("inductance", 1e-6),
    "Hz": ("frequency", 1.0), "kHz": ("frequency", 1e3), "MHz": ("frequency", 1e6), "GHz": ("frequency", 1e9),
    "dB": ("level", 1.0),
    # temperature (separate families; conversion handled specially)
    "K": ("temperature_k", 1.0), "degC": ("temperature_c", 1.0), "°C": ("temperature_c", 1.0), "C": ("temperature_c", 1.0),
    # time (s)
    "ms": ("time", 1e-3), "us": ("time", 1e-6), "µs": ("time", 1e-6), "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0), "hr": ("time", 3600.0),
    # velocity (m/s) / acceleration (m/s2)
    "m/s": ("velocity", 1.0), "mm/s": ("velocity", 1e-3), "cm/s": ("velocity", 1e-2), "km/h": ("velocity", 1 / 3.6), "ft/s": ("velocity", 0.3048),
    "m/s2": ("acceleration", 1.0), "mm/s2": ("acceleration", 1e-3), "g0": ("acceleration", 9.80665),
    # flow / rate
    "rpm": ("angular_speed", 1.0), "rad/s": ("angular_speed", 60 / (2 * 3.141592653589793)),
    "kg/h": ("mass_flow", 1 / 3600), "kg/s": ("mass_flow", 1.0), "m3/h": ("volume_flow", 1 / 3600), "m3/s": ("volume_flow", 1.0), "L/min": ("volume_flow", 1e-3 / 60), "L/s": ("volume_flow", 1e-3),
    # amount / concentration
    "mol": ("amount", 1.0), "mmol": ("amount", 1e-3), "kmol": ("amount", 1e3), "ppm": ("fraction", 1e-6), "%": ("fraction", 1e-2),
    # density / distributed load
    "kg/m3": ("density", 1.0), "g/cm3": ("density", 1e3), "kN/m": ("line_load", 1e3), "N/m": ("line_load", 1.0), "kN/m2": ("pressure", 1e3), "N/mm2": ("pressure", 1e6),
}

#: lower-cased spellings whose meaning is unambiguous (or conventional) -> canonical key
_LOWER_INDEX: dict[str, str] = {}
for _key in UNIT_TABLE:
    _low = _key.lower()
    if _low in _LOWER_INDEX and _LOWER_INDEX[_low] != _key:
        _LOWER_INDEX[_low] = ""  # ambiguous once lower-cased (mN/MN, mW/MW, mV/MV, mJ/MJ, h/H ...)
    else:
        _LOWER_INDEX.setdefault(_low, _key)
#: conventional reading of an ambiguous lower-cased token (engineers who drop the case mean these)
_PREFERRED_ON_CONFLICT: dict[str, str] = {"mpa": "MPa", "mm": "mm", "ms": "ms", "mg": "mg", "mhz": "MHz", "h": "h", "mf": "mF", "mh": "mH"}
_LOWER_INDEX.update(_PREFERRED_ON_CONFLICT)
_LOWER_INDEX = {k: v for k, v in _LOWER_INDEX.items() if v}

# N·m / Nm torque is ambiguous with nanometre; engineering text almost always means torque when capitalised.
_TORQUE_TOKENS = {"Nm", "N·m", "N-m", "N.m", "kNm", "kN·m", "kN-m", "kN.m"}
_TORQUE_FACTORS = {"Nm": 1.0, "N·m": 1.0, "N-m": 1.0, "N.m": 1.0, "kNm": 1e3, "kN·m": 1e3, "kN-m": 1e3, "kN.m": 1e3}

_QUANTITY_RE = re.compile(
    r"(?<![\w.])(?P<value>[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?)\s?(?P<unit>°C|degC|µm|µF|µA|µH|µs|Ω|%|[A-Za-z]+[0-9²³]?(?:[/·.\-][A-Za-z0-9²³]+)?)?(?![\w])",
)
#: Ordinary English words that follow counts ("7 bolts", "4 days") and must not be read as units.
_WORDY_UNITS = {
    "and", "or", "to", "of", "for", "with", "at", "in", "the", "is", "per", "by", "x", "times", "items", "bolts", "bars",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes", "second", "seconds",
    "pcs", "pieces", "piece", "units", "unit", "nos", "off", "each", "holes", "hole", "steps", "step", "cycles", "cycle",
    "samples", "sample", "people", "persons", "points", "point", "places", "pages", "page", "parts", "part", "runs", "run",
    "passes", "pass", "tests", "test", "cases", "case", "layers", "layer", "coats", "coat", "sets", "set", "pairs", "pair",
    "rows", "row", "columns", "column", "bits", "bit", "bytes", "byte", "kb", "mb", "gb", "tb",
}


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: Optional[str] = Field(default=None, description="Canonical UNIT_TABLE key (e.g. 'MPa'), or the raw token when unknown")
    family: Optional[str] = None
    raw: str
    start: int
    end: int


class UnitReport(BaseModel):
    """Result of :func:`check_units`."""

    model_config = ConfigDict(extra="forbid")

    quantities: list[Quantity] = Field(default_factory=list)
    unknown_units: list[str] = Field(default_factory=list)
    bare_decimals: list[str] = Field(default_factory=list, description="Decimal numbers with no unit")
    families: dict[str, int] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def normalize_unit(token: str) -> Optional[str]:
    """Map a raw unit token to a key of :data:`UNIT_TABLE` (None if unknown).

    Exact (case-sensitive) spelling wins, so ``mW``/``MW`` and ``mPa``/``MPa`` stay
    distinct; a lower-cased lookup is tried only for unambiguous spellings.
    """
    if token in _TORQUE_TOKENS:
        return "nm_torque"
    t = token.replace("·", "/").replace("^", "").replace("³", "3").replace("²", "2")
    if t in UNIT_TABLE:
        return t
    low = t.lower()
    if low in _LOWER_INDEX:
        return _LOWER_INDEX[low]
    return None


def unit_family(token: str) -> Optional[str]:
    """Family of a unit token (``'pressure'`` for ``MPa``), or None when unknown."""
    key = normalize_unit(token)
    return UNIT_TABLE[key][0] if key else None


def parse_quantities(text: str) -> list[Quantity]:
    """Find number(+unit) occurrences. Units that are ordinary words are treated as absent."""
    out: list[Quantity] = []
    for m in _QUANTITY_RE.finditer(text):
        raw_val = m.group("value").replace(",", "")
        try:
            value = float(raw_val)
        except ValueError:
            continue
        unit_tok = m.group("unit")
        unit = family = None
        raw = m.group(0).strip()
        if unit_tok and unit_tok.lower() not in _WORDY_UNITS:
            key = normalize_unit(unit_tok)
            if key is not None:
                unit, family = key, UNIT_TABLE[key][0]
            else:
                unit = unit_tok  # unknown unit token kept raw
        elif unit_tok:
            raw = m.group("value")  # an ordinary word followed the number; it is not a unit
        out.append(Quantity(value=value, unit=unit, family=family, raw=raw, start=m.start(), end=m.end()))
    return out


def convert(value: float, unit_from: str, unit_to: str) -> float:
    """Convert between units of the same family (temperature C/K handled)."""
    a, b = normalize_unit(unit_from), normalize_unit(unit_to)
    if a is None or b is None:
        raise ValueError(f"unknown unit: {unit_from if a is None else unit_to}")
    fa, ka = UNIT_TABLE[a]
    fb, kb = UNIT_TABLE[b]
    if a == "nm_torque":
        ka = _TORQUE_FACTORS.get(unit_from, 1.0)
    if b == "nm_torque":
        kb = _TORQUE_FACTORS.get(unit_to, 1.0)
    if {fa, fb} == {"temperature_c", "temperature_k"}:
        return value + 273.15 if fa == "temperature_c" else value - 273.15
    if fa != fb:
        raise ValueError(f"incompatible families: {fa} vs {fb}")
    return value * ka / kb


def check_units(text: str, *, expected_units: list[str] | None = None, require_units: bool = False) -> UnitReport:
    """Check unit hygiene of ``text``.

    * unknown unit tokens attached to numbers are reported;
    * when ``require_units`` is true, decimal numbers without a unit are reported;
    * when ``expected_units`` is given, units outside the domain's list and outside
      the known table are reported (a domain list extends, not replaces, the table).
    """
    qs = parse_quantities(text)
    report = UnitReport(quantities=qs)
    expected = {u.lower() for u in (expected_units or [])}
    for q in qs:
        if q.unit is not None and q.family is None:
            if q.unit.lower() in expected or q.unit.lower().isalpha() and len(q.unit) > 4:
                continue  # domain-specific or just a word glued to a number
            report.unknown_units.append(q.unit)
        elif q.unit is None and require_units and "." in q.raw:
            report.bare_decimals.append(q.raw)
        if q.family:
            report.families[q.family] = report.families.get(q.family, 0) + 1
    if report.unknown_units:
        report.issues.append("unknown unit token(s): " + ", ".join(sorted(set(report.unknown_units))))
    if report.bare_decimals:
        report.issues.append("quantities without units: " + ", ".join(report.bare_decimals[:5]))
    return report


__all__ = ["UNIT_TABLE", "Quantity", "UnitReport", "check_units", "convert", "normalize_unit", "parse_quantities", "unit_family"]
