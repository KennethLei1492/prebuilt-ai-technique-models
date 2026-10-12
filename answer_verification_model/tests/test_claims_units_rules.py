"""Pure logic: claim splitting/support, unit parsing, rule checks, sandbox."""
from __future__ import annotations

import sys

import pytest

from pipeline_core import Evidence

from answer_verification import check_units, citation_metrics, convert, grounding_metrics, lexical_support, parse_quantities, run_rule_checks, split_claims
from answer_verification.sandbox import extract_code, parse_test_summary, run_python

EV = [
    Evidence(id="spec-1", content="Section 4.2: the pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces."),
    Evidence(id="spec-2", content="Table 2: discharge piping design pressure 16 bar at 80 degC."),
]


def test_split_claims_types_and_hedges():
    text = "The casing allowance is 3 mm. First, lift the pump.\n```python\nreturn x\n```\nIn summary, it works. See [spec-1]."
    claims = split_claims(text, min_words=2)
    types = {c.text[:12]: c.type for c in claims}
    assert any(t == "numerical" for t in types.values())
    assert any(t == "procedural" for t in types.values())
    assert not any(c.text.startswith("In summary") for c in claims)


def test_split_claims_drops_fragments_below_min_words():
    assert split_claims("9.6 mm") == []  # the GroundingVerifier treats such an answer as one claim (see test_verifiers)
    assert len(split_claims("The required thickness is 9.6 mm.")) == 1


def test_lexical_support_labels():
    sup = lexical_support("The pump casing requires a corrosion allowance of 3 mm on wetted surfaces.", EV)
    assert sup.label == "supported" and sup.evidence_ids == ["spec-1"]
    con = lexical_support("The pump casing requires a corrosion allowance of 6 mm on wetted surfaces.", EV)
    assert con.label == "contradicted" and con.evidence_ids == ["spec-1"]
    partial = lexical_support("The discharge piping design pressure is 25 bar at 80 degC.", EV)
    assert partial.label == "contradicted", "one matching number must not rescue a wrong value"
    uns = lexical_support("The impeller is made of bronze alloy C95800.", EV)
    assert uns.label == "unsupported"
    grade_ev = [Evidence(id="c5", content="Clause 5.1: the impeller shall be cast in ASTM A743 CA6NM martensitic stainless steel.")]
    assert lexical_support("The impeller shall be ASTM A743 CA6NM martensitic stainless steel.", grade_ev).label == "supported"
    assert lexical_support("The impeller shall be ASTM A216 WCB carbon steel.", grade_ev).label == "contradicted", "identifier mismatch is a contradiction"
    spanning = lexical_support("P = 1.5 MPa, R = 600 mm and S = 138 MPa at 150 degC.", [Evidence(id="a", content="Design case P = 1.5 MPa, R = 600 mm, S = 138 MPa."), Evidence(id="b", content="Table 1A: S = 138 MPa at 150 degC.")])
    assert spanning.label == "supported" and spanning.evidence_ids == ["a", "b"], "keys may be covered by several passages"


def test_grounding_and_citation_metrics():
    verdicts = [lexical_support("A corrosion allowance of 3 mm on wetted surfaces of the casing [spec-1].", EV), lexical_support("The impeller is bronze.", EV)]
    gm = grounding_metrics(verdicts)
    assert gm["claim_count"] == 2 and gm["supported"] == 1 and gm["supported_fraction"] == 0.5
    cm = citation_metrics("A corrosion allowance of 3 mm [spec-1]. The impeller is bronze [spec-9].", verdicts, ["spec-1", "spec-2"])
    assert cm["citation_precision"] == 0.5 and cm["citation_recall"] == 0.5 and cm["citations"] == 2


def test_parse_quantities_and_convert():
    qs = parse_quantities("Stress 12.5 MPa, load 3 kN, supply 24V, torque 40 Nm, 2 bolts")
    units = {q.unit for q in qs}
    assert {"MPa", "kN", "V", "nm_torque"} <= units, "units are reported as canonical UNIT_TABLE keys (case-sensitive)"
    assert convert(3, "kN", "N") == 3000
    assert convert(25, "degC", "K") == pytest.approx(298.15)
    with pytest.raises(ValueError):
        convert(1, "kN", "MPa")


def test_si_prefix_case_is_preserved():
    """Regression: lower-casing before lookup collapsed milli and mega (mW -> MW, mPa -> MPa)."""
    assert convert(1, "mW", "W") == pytest.approx(1e-3)
    assert convert(1, "MW", "W") == pytest.approx(1e6)
    assert convert(1, "mN", "N") == pytest.approx(1e-3) and convert(1, "MN", "N") == pytest.approx(1e6)
    assert convert(1, "mPa", "Pa") == pytest.approx(1e-3) and convert(1, "MPa", "Pa") == pytest.approx(1e6)
    assert convert(1, "Mm", "m") == pytest.approx(1e6) and convert(1, "mm", "m") == pytest.approx(1e-3)
    assert convert(1, "BAR", "Pa") == pytest.approx(1e5), "case-insensitive fallback for unambiguous spellings"
    assert convert(1, "mpa", "Pa") == pytest.approx(1e6), "dropped-case 'mpa' means MPa by convention"
    assert {q.unit for q in parse_quantities("3 mW and 5 MW")} == {"mW", "MW"}


def test_check_units_knows_engineering_families_and_count_words():
    """Regression: velocity / acceleration / area / volume were unknown and '4 days' was read as a unit."""
    rep = check_units("3.0 mm/s rms, 2 m/s, 4 m2, 3 L, 9.81 m/s2, 12 L/min over 4 days and 2 bars of soap", expected_units=["MPa", "mm"])
    assert rep.unknown_units == []
    assert {"velocity", "acceleration", "area", "volume", "volume_flow"} <= set(rep.families)


def test_check_units_reports_unknown_and_missing():
    rep = check_units("The stress is 12.5 and the load is 3 kN plus 7 blp", expected_units=["MPa"], require_units=True)
    assert rep.bare_decimals == ["12.5"]
    assert "blp" in rep.unknown_units
    assert check_units("7 bolts and 3 kN").unknown_units == [], "ordinary words glued to counts are not units"
    assert not rep.ok
    assert check_units("The load is 3 kN at 20 degC").ok


def test_rule_checks_cover_each_rule_type():
    answer = "Summary\n```json\n{\"a\": 1}\n```\nRisks\nThe value is 42 units."
    rules = [
        {"type": "regex", "pattern": r"value is \d+", "description": "states value"},
        {"type": "forbidden", "pattern": r"TODO"},
        {"type": "range", "min": 0, "max": 10, "description": "range"},
        {"type": "required_sections", "sections": ["Summary", "Risks", "Assumptions"]},
        {"type": "max_words", "value": 3},
        {"type": "json_schema", "required": ["a", "b"]},
        {"type": "mystery"},
    ]
    checks = {c.rule.split(":")[0]: c for c in run_rule_checks(answer, rules=rules)}
    assert checks["regex"].passed and checks["forbidden"].passed
    assert not checks["range"].passed
    assert not checks["required_sections"].passed and "Assumptions" in checks["required_sections"].detail
    assert not checks["max_words"].passed and checks["max_words"].severity == "soft"
    assert not checks["json"].passed and "b" in checks["json"].detail
    assert checks["unknown"].passed and checks["unknown"].severity == "soft"


def test_rule_checks_reference_units_and_code():
    ok = run_rule_checks("The preload is 12.6 kN", answer_type="numerical", reference="12.5 kN", tolerance=0.02, expected_units=["kN"])
    assert any(c.rule == "reference_match" and c.passed for c in ok)
    assert any(c.rule == "units.present" and c.passed for c in ok)
    bad = run_rule_checks("The preload is 14.0", answer_type="numerical", reference="12.5 kN", expected_units=["kN"])
    assert any(c.rule == "reference_match" and not c.passed for c in bad)
    assert any(c.rule == "units.present" and not c.passed for c in bad)
    broken = run_rule_checks("```python\ndef f(:\n    pass\n```", answer_type="code", code_languages=["python"])
    assert any(c.rule == "code.parses" and not c.passed for c in broken)


def test_rule_checks_compare_the_answer_number_not_the_last_number():
    """Regression: reference and range checks used the last number in the text ('80' in '16 bar at 80 degC')."""
    ref = run_rule_checks("The design pressure is 16 bar at 80 degC.", answer_type="numerical", reference="16 bar", expected_units=["bar"])
    match = next(c for c in ref if c.rule == "reference_match")
    assert match.passed, match.detail
    rng = run_rule_checks("Use a 75 kW motor at 1480 rpm.", answer_type="numerical", rules=[{"type": "range", "min": 50, "max": 100, "unit": "kW"}])
    assert next(c for c in rng if c.rule.startswith("range")).passed
    rng_no_unit = run_rule_checks("Use a 75 kW motor at 1480 rpm.", answer_type="numerical", rules=[{"type": "range", "min": 50, "max": 100}])
    assert next(c for c in rng_no_unit if c.rule.startswith("range")).passed, "without a unit the first stand-alone number is the answer"
    bolt = run_rule_checks("The preload is 12.5 kN per M16 bolt", answer_type="numerical", reference="12.5 kN")
    assert next(c for c in bolt if c.rule == "reference_match").passed, "digits glued to letters (M16) are identifiers, not values"
    wrong = run_rule_checks("The design pressure is 25 bar at 80 degC.", answer_type="numerical", reference="16 bar")
    assert not next(c for c in wrong if c.rule == "reference_match").passed


def test_rule_checks_return_nothing_when_no_rule_applies_and_skip_prose_compile():
    """Regression: a domain without units produced zero checks, which RuleVerifier reported as a conclusive pass."""
    assert run_rule_checks("2+2=5", answer_type="numerical") == []
    assert run_rule_checks("The pump will return to service after 3 hours of cooling.", answer_type="code", code_languages=["python"]) == [], "prose mislabelled 'code' is not compiled"
    assert any(c.rule == "code.parses" for c in run_rule_checks("```python\nx = (\n```", answer_type="code", code_languages=["python"]))


def test_sandbox_extract_and_summary():
    code, lang = extract_code("Here:\n```python\nx = 1\n```\nand\n```js\nlet y = 2;\n```", language_hint="js")
    assert lang == "js" and "let y" in code
    assert parse_test_summary("3 passed, 1 failed in 0.1s") == (3, 1)


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="needs a subprocess")
def test_sandbox_runs_tests_pass_fail_and_timeout():
    good = run_python("def add(a, b):\n    return a + b\n", "assert add(1, 2) == 3\nprint('1 passed')", timeout_s=30)
    assert good.passed and good.exit_code == 0 and good.tests_passed == 1
    bad = run_python("def add(a, b):\n    return a - b\n", "assert add(1, 2) == 3", timeout_s=30)
    assert not bad.passed and "AssertionError" in bad.stderr
    slow = run_python("import time\nwhile True:\n    time.sleep(0.05)\n", "pass", timeout_s=1.0)
    assert slow.timed_out and not slow.passed
