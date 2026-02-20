from pathlib import Path

from backend.agent import _extract_math_expression, _extract_run_command, _system_prompt


def test_system_prompt_strict_on() -> None:
    prompt = _system_prompt(Path('/tmp'), strict_facts=True)
    assert 'Strict facts mode is enabled' in prompt


def test_system_prompt_strict_off() -> None:
    prompt = _system_prompt(Path('/tmp'), strict_facts=False)
    assert 'Strict facts mode is enabled' not in prompt


def test_extract_run_command_strips_trailing_cwd() -> None:
    cmd = _extract_run_command("run command python -m pytest -q in /tmp/project")
    assert cmd == "python -m pytest -q"


def test_extract_math_expression_basic() -> None:
    expr = _extract_math_expression("calculate (27 * 14) + 2")
    assert expr == "(27 * 14) + 2"
