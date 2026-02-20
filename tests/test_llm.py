from backend.llm import OllamaClient


def test_choose_model_exact_match() -> None:
    client = OllamaClient(model='llama3.2:3b')
    assert client._choose_model(['llama3.2:3b', 'qwen2.5:7b']) == 'llama3.2:3b'


def test_choose_model_same_base() -> None:
    client = OllamaClient(model='llama3.2:3b')
    assert client._choose_model(['llama3.2:latest']) == 'llama3.2:latest'


def test_choose_model_fallback_first() -> None:
    client = OllamaClient(model='missing:model')
    assert client._choose_model(['qwen2.5:7b', 'llama3.2:3b']) == 'qwen2.5:7b'
