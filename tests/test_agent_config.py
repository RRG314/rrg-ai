from pathlib import Path

from backend.agent import _system_prompt


def test_system_prompt_strict_on() -> None:
    prompt = _system_prompt(Path('/tmp'), strict_facts=True)
    assert 'Strict facts mode is enabled' in prompt


def test_system_prompt_strict_off() -> None:
    prompt = _system_prompt(Path('/tmp'), strict_facts=False)
    assert 'Strict facts mode is enabled' not in prompt
