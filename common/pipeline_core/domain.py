"""Domain profiles: the mechanism that adapts every model to an engineering field.

A :class:`DomainProfile` bundles vocabulary, units, document types,
verification rules and retrieval/decomposition hints. Components read the
profile named by ``TaskRequest.domain`` and use it to shape prompts, choose
retrieval weights (identifiers favour lexical search), pick verifiers (unit
checks, standards references) and set risk posture (safety-critical domains
demand higher confidence before auto-accepting an answer).

Add a new field by calling :func:`register_profile` - no component code changes.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .prompts import bullet_list, xml

log = logging.getLogger(__name__)

RiskLevel = Literal["low", "medium", "high", "safety_critical"]


class DomainProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    glossary: dict[str, str] = Field(default_factory=dict, description="Term -> definition, injected into prompts")
    units: list[str] = Field(default_factory=list, description="Expected units; verification checks consistency")
    document_types: list[str] = Field(default_factory=list)
    verification_rules: list[str] = Field(default_factory=list, description="Natural-language checks a verifier must apply")
    style_guidance: str = ""
    code_languages: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "medium"
    min_confidence_to_accept: float = Field(default=0.6, ge=0.0, le=1.0)
    retrieval_hints: dict[str, float | str] = Field(
        default_factory=dict,
        description="e.g. {'sparse_weight': 0.6} - identifiers, part numbers and standards favour lexical matching",
    )
    decomposition_hints: list[str] = Field(default_factory=list)
    standards: list[str] = Field(default_factory=list, description="Standards bodies / codes commonly referenced")

    def system_text(self) -> str:
        """Stable, cache-friendly rendering for inclusion in system prompts."""
        parts = [f"Domain: {self.name}. {self.description}"]
        if self.glossary:
            parts.append(xml("glossary", bullet_list(f"{k}: {v}" for k, v in sorted(self.glossary.items()))))
        if self.units:
            parts.append("Common units: " + ", ".join(self.units))
        if self.standards:
            parts.append("Relevant standards: " + ", ".join(self.standards))
        if self.verification_rules:
            parts.append(xml("verification_rules", bullet_list(self.verification_rules)))
        if self.style_guidance:
            parts.append(self.style_guidance)
        parts.append(f"Risk level: {self.risk_level}.")
        return "\n".join(parts)


_PROFILES: dict[str, DomainProfile] = {}


def register_profile(profile: DomainProfile, *, overwrite: bool = False) -> DomainProfile:
    if profile.name in _PROFILES and not overwrite:
        raise ValueError(f"domain profile '{profile.name}' already registered")
    _PROFILES[profile.name] = profile
    return profile


def get_profile(name: Optional[str]) -> DomainProfile:
    if not name:
        return _PROFILES["general"]
    prof = _PROFILES.get(name)
    if prof is None:
        log.warning("unknown domain profile '%s'; falling back to 'general'", name)
        return _PROFILES["general"]
    return prof


def list_profiles() -> list[str]:
    return sorted(_PROFILES)


# Built-in profiles ---------------------------------------------------------

register_profile(
    DomainProfile(
        name="general",
        description="General knowledge work with no domain-specific conventions.",
        risk_level="low",
        min_confidence_to_accept=0.5,
    )
)

register_profile(
    DomainProfile(
        name="software_engineering",
        description="Software design, implementation, review and operations.",
        glossary={
            "SLO": "Service level objective - a target for a reliability metric",
            "ADR": "Architecture decision record",
            "CI/CD": "Continuous integration and continuous delivery pipeline",
        },
        document_types=["source code", "API specification", "design document", "runbook", "incident report", "test log", "ADR"],
        verification_rules=[
            "Code must compile or parse in the stated language.",
            "Claims about library behaviour must match the referenced version's documentation.",
            "Security-relevant changes must identify the trust boundary they affect.",
        ],
        style_guidance="Prefer concrete code and command examples; name files and functions exactly.",
        code_languages=["python", "typescript", "go", "java", "rust", "sql", "bash"],
        risk_level="medium",
        min_confidence_to_accept=0.65,
        retrieval_hints={"sparse_weight": 0.55, "reason": "identifiers, error strings and API names are exact-match tokens"},
        decomposition_hints=["Separate interface design from implementation", "Include a test or verification sub-task for each change"],
        standards=["OWASP ASVS", "SemVer", "OpenAPI", "RFC 2119"],
    )
)

register_profile(
    DomainProfile(
        name="mechanical_engineering",
        description="Mechanical design, materials, thermal and structural analysis, manufacturing.",
        glossary={
            "FoS": "Factor of safety - ratio of allowable to applied load or stress",
            "GD&T": "Geometric dimensioning and tolerancing",
            "BOM": "Bill of materials",
        },
        units=["N", "kN", "MPa", "GPa", "mm", "m", "kg", "K", "degC", "W", "J", "rpm", "Nm"],
        document_types=["drawing", "specification", "BOM", "test report", "FMEA", "material datasheet", "calculation note"],
        verification_rules=[
            "Every quantity must carry a unit and units must be dimensionally consistent.",
            "Stress and load results must be compared against an allowable with an explicit factor of safety.",
            "Material properties must cite a datasheet or standard and the temperature at which they apply.",
        ],
        style_guidance="Show governing equations, inputs with units, and the result with its margin.",
        risk_level="high",
        min_confidence_to_accept=0.75,
        retrieval_hints={"sparse_weight": 0.6, "reason": "part numbers, material grades and standard designations are exact tokens"},
        decomposition_hints=["Identify loads and boundary conditions before analysis", "Separate sizing from verification"],
        standards=["ASME Y14.5", "ASME BPVC", "ISO 2768", "ASTM material specifications", "ISO 286"],
    )
)

register_profile(
    DomainProfile(
        name="electrical_engineering",
        description="Circuit design, power systems, signal integrity, embedded hardware.",
        glossary={
            "SOA": "Safe operating area of a semiconductor device",
            "PSRR": "Power supply rejection ratio",
            "EMC": "Electromagnetic compatibility",
        },
        units=["V", "A", "mA", "ohm", "kohm", "W", "Hz", "kHz", "MHz", "F", "uF", "nF", "H", "dB", "degC"],
        document_types=["schematic", "datasheet", "netlist", "test procedure", "compliance report", "firmware source"],
        verification_rules=[
            "Component ratings must exceed worst-case operating conditions with stated derating.",
            "Power and thermal budgets must sum consistently.",
            "Signal-level claims must state the reference (ground, rail) and measurement condition.",
        ],
        style_guidance="State operating conditions, tolerances and worst-case corners explicitly.",
        code_languages=["c", "cpp", "verilog", "vhdl", "python"],
        risk_level="high",
        min_confidence_to_accept=0.75,
        retrieval_hints={"sparse_weight": 0.6, "reason": "part numbers and pin names are exact tokens"},
        standards=["IEC 61000", "IPC-2221", "IEEE 1149.1", "UL 60950/62368"],
    )
)

register_profile(
    DomainProfile(
        name="civil_structural_engineering",
        description="Structural design, geotechnics, construction documents and codes.",
        units=["kN", "kN/m", "kPa", "MPa", "m", "mm", "kg/m3"],
        document_types=["drawing", "specification", "geotechnical report", "load calculation", "code clause"],
        verification_rules=[
            "Load combinations must follow the governing code and be stated.",
            "Capacity checks must report demand/capacity ratio and the code clause used.",
            "Units must be consistent and SI unless the project specifies otherwise.",
        ],
        risk_level="safety_critical",
        min_confidence_to_accept=0.85,
        retrieval_hints={"sparse_weight": 0.6, "reason": "code clause numbers and section designations are exact tokens"},
        standards=["Eurocode EN 1990-1999", "ASCE 7", "ACI 318", "AISC 360", "IBC"],
    )
)

register_profile(
    DomainProfile(
        name="systems_engineering",
        description="Requirements, architecture, interfaces, verification and validation across disciplines.",
        glossary={
            "V&V": "Verification and validation",
            "ICD": "Interface control document",
            "RTM": "Requirements traceability matrix",
        },
        document_types=["requirements specification", "ICD", "architecture description", "test plan", "trade study", "RTM"],
        verification_rules=[
            "Each requirement must be unambiguous, verifiable and traced to a parent or stakeholder need.",
            "Interfaces must specify both sides and the governing ICD.",
            "Trade studies must state criteria, weights and sensitivity.",
        ],
        style_guidance="Use 'shall' for requirements; keep identifiers (REQ-xxx) stable.",
        risk_level="high",
        min_confidence_to_accept=0.75,
        retrieval_hints={"sparse_weight": 0.6, "reason": "requirement identifiers are exact tokens"},
        decomposition_hints=["Decompose along the system hierarchy", "Attach a verification method to each requirement"],
        standards=["ISO/IEC/IEEE 15288", "INCOSE SE Handbook", "ISO 26262", "DO-178C", "ARP4754A"],
    )
)

register_profile(
    DomainProfile(
        name="data_engineering",
        description="Data pipelines, schemas, quality, warehousing and analytics engineering.",
        document_types=["schema", "SQL", "DAG definition", "data contract", "quality report", "lineage graph"],
        verification_rules=[
            "SQL must be syntactically valid for the named dialect.",
            "Schema changes must state backward compatibility impact.",
            "Aggregations must declare grain and time zone handling.",
        ],
        code_languages=["sql", "python", "scala"],
        risk_level="medium",
        min_confidence_to_accept=0.65,
        retrieval_hints={"sparse_weight": 0.55, "reason": "table and column names are exact tokens"},
        standards=["ISO 8601", "SQL:2016", "Apache Parquet", "OpenLineage"],
    )
)

register_profile(
    DomainProfile(
        name="chemical_process_engineering",
        description="Process design, mass and energy balances, unit operations and safety.",
        units=["kg/h", "mol", "kPa", "bar", "K", "degC", "kW", "m3/h", "ppm"],
        document_types=["PFD", "P&ID", "HAZOP report", "material safety data sheet", "heat and material balance"],
        verification_rules=[
            "Mass and energy balances must close within stated tolerance.",
            "Operating conditions must be compared with material compatibility and relief settings.",
            "Hazards must reference the HAZOP node or safeguard.",
        ],
        risk_level="safety_critical",
        min_confidence_to_accept=0.85,
        retrieval_hints={"sparse_weight": 0.55},
        standards=["API 520/521", "IEC 61511", "ASME B31.3", "OSHA PSM"],
    )
)
