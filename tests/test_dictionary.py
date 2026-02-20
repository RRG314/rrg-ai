from backend.agent import _extract_dictionary_term
from backend.tools.web import _normalize_word, _parse_dictionary_payload


def test_extract_dictionary_term_patterns() -> None:
    assert _extract_dictionary_term("define entropy") == "entropy"
    assert _extract_dictionary_term("what does logic mean?") == "logic"
    assert _extract_dictionary_term("give me the definition of theorem") == "theorem"
    assert _extract_dictionary_term("search web for ai") == ""


def test_parse_dictionary_payload() -> None:
    payload = [
        {
            "word": "entropy",
            "phonetic": "/ˈɛntrəpi/",
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definitions": [
                        {
                            "definition": "A measure of disorder in a system.",
                            "example": "Entropy increases in isolated systems.",
                        }
                    ],
                }
            ],
        }
    ]

    out = _parse_dictionary_payload("entropy", payload, max_definitions=4)
    assert out["word"] == "entropy"
    assert out["source"] == "dictionaryapi.dev"
    assert out["definitions"]
    assert out["definitions"][0]["part_of_speech"] == "noun"


def test_normalize_word() -> None:
    assert _normalize_word("Logic!") == "logic"
    assert _normalize_word("mother-in-law") == "mother-in-law"
