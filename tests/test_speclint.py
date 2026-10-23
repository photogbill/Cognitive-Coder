# SPDX-License-Identifier: Apache-2.0
"""speclint — what a spec says that a build will faithfully get wrong."""

from cognitive_coder import speclint

SPEC = """# Begin 4

* `hull_integrity: float = 100.0`
* `warp_heat: float = 0.0`
* `Reserve_energy: float = 100.0`
* `shields: list[float]` — six sectors
* `SUBTICKS_PER_TURN = 10`, with `dt = 0.1`
* class `Ship` holds the above; `WeaponBank` has `range_limit`
* `parse_command_line(ship_id: str, line: str)`
* every turn runs `SUBTICKS_PER_TURN = 12` sub-ticks
* heat above the rating raises `warpHeat` each sub-tick
"""


def test_the_capitalised_field_is_named_with_its_line():
    f = [x for x in speclint.lint(SPEC) if x.kind == "case"
         and x.text == "Reserve_energy"]
    assert [(x.line, x.text, x.suggestion) for x in f] == [
        (5, "Reserve_energy", "reserve_energy")]
    assert "snake_case" in f[0].sentence


def test_a_constant_with_two_values_is_a_finding():
    f = [x for x in speclint.lint(SPEC) if x.kind == "constant"]
    assert len(f) == 1 and f[0].text == "SUBTICKS_PER_TURN"
    assert "10 on line 7" in f[0].sentence and "12 on line 10" in f[0].sentence


def test_two_spellings_of_one_name():
    f = [x for x in speclint.lint(SPEC) if x.kind == "spelling"]
    assert len(f) == 1
    assert (f[0].text, f[0].suggestion) == ("warpHeat", "warp_heat")
    assert "line 11" in f[0].sentence and "line 4" in f[0].sentence


def test_camel_case_in_a_snake_case_spec_is_a_case_finding():
    f = [x for x in speclint.lint(SPEC) if x.kind == "case"
         and x.text == "warpHeat"]
    assert f and f[0].suggestion == "warp_heat"


def test_classes_and_constants_are_not_case_findings():
    assert not [x for x in speclint.lint(
        "class `Ship`\n`SUBTICKS_PER_TURN = 10`\n`WeaponBank`\n")
        if x.kind == "case"]


def test_a_consistent_spec_is_silent():
    assert speclint.lint("`warp_heat: float`\n`hull: float`\n") == []
    assert speclint.report([]) == ""


def test_the_report_reads_as_sentences():
    text = speclint.report(speclint.lint(SPEC))
    assert text.startswith("The specification has 4 thing(s)")
    assert "line 5:" in text and "nothing is guessed" in text
