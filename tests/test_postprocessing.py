"""
Unit tests for Context-Aware Homoglyph Disambiguation.
"""

from src.postprocessing import HomoglyphDisambiguator


def test_homoglyph_disambiguation_numeric_sequence():
    disambiguator = HomoglyphDisambiguator()
    # Sequence '20O4' -> predominantly digits, 'O' should become '0'
    raw_chars = ['2', '0', 'O', '4']
    res, actions = disambiguator.disambiguate(raw_chars, mode="auto")
    assert res == "2004"
    assert len(actions) == 1
    assert actions[0]["original"] == 'O'
    assert actions[0]["disambiguated"] == '0'


def test_homoglyph_disambiguation_alphabetic_sequence():
    disambiguator = HomoglyphDisambiguator()
    # Sequence 'B0X' -> predominantly letters, '0' should become 'O'
    raw_chars = ['B', '0', 'X']
    res, actions = disambiguator.disambiguate(raw_chars, mode="auto")
    assert res == "BOX"
    assert len(actions) == 1
    assert actions[0]["original"] == '0'
    assert actions[0]["disambiguated"] == 'O'


def test_forced_digits_mode():
    disambiguator = HomoglyphDisambiguator()
    raw_chars = ['S', 'I', 'B']  # Ambiguous letters
    res, actions = disambiguator.disambiguate(raw_chars, mode="digits_only")
    assert res == "518"


def test_forced_letters_mode():
    disambiguator = HomoglyphDisambiguator()
    raw_chars = ['5', '1', '8']  # Digits mapped to letters
    res, actions = disambiguator.disambiguate(raw_chars, mode="letters_only")
    assert res == "SIB"


def test_mixed_sandwiched_disambiguation():
    disambiguator = HomoglyphDisambiguator()
    # Balanced length where target_mode would be mixed
    # 'A' '0' 'B' -> '0' sandwiched between 'A' and 'B' -> should become 'O'
    raw_chars = ['A', '0', 'B', '1', '2']
    res, actions = disambiguator.disambiguate(raw_chars, mode="mixed")
    assert res[1] == 'O'


def test_homoglyph_disambiguation_with_top_k_lists():
    disambiguator = HomoglyphDisambiguator()
    # Sequence ['2', 'Z', '7'] with top-k candidate lists
    raw_chars = ['2', 'Z', '7']
    top_k_lists = [
        [{"character": "2", "probability": 0.95}, {"character": "Z", "probability": 0.04}],
        [{"character": "Z", "probability": 0.70}, {"character": "2", "probability": 0.28}],
        [{"character": "7", "probability": 0.98}, {"character": "1", "probability": 0.01}]
    ]
    res, actions = disambiguator.disambiguate(raw_chars, top_k_lists=top_k_lists, mode="auto")
    assert res == "227"
    assert len(actions) == 1
    assert actions[0]["original"] == 'Z'
    assert actions[0]["disambiguated"] == '2'


def test_homoglyph_disambiguation_letters_mode_with_top_k():
    disambiguator = HomoglyphDisambiguator()
    raw_chars = ['A', '0', 'C']
    top_k_lists = [
        [{"character": "A", "probability": 0.95}],
        [{"character": "0", "probability": 0.60}, {"character": "O", "probability": 0.38}],
        [{"character": "C", "probability": 0.92}]
    ]
    res, actions = disambiguator.disambiguate(raw_chars, top_k_lists=top_k_lists, mode="letters_only")
    assert res == "AOC"
    assert len(actions) == 1
    assert actions[0]["original"] == '0'
    assert actions[0]["disambiguated"] == 'O'

