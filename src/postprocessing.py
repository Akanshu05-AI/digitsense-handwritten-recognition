"""
DigitSense: Post-Processing & Homoglyph Disambiguation Module
Resolves topological ambiguities between visually indistinguishable handwritten characters
(e.g., '0' vs 'O', '1' vs 'I'/'l', '5' vs 'S', '8' vs 'B', '2' vs 'Z') using sequence context,
character class distributions, and top-k candidate re-ranking.
"""

from typing import List, Dict, Any, Tuple, Optional
import copy

# Mapping ambiguous letters to their most common digit counterparts
LETTER_TO_DIGIT = {
    'O': '0',
    'o': '0',
    'D': '0',
    'I': '1',
    'l': '1',
    'S': '5',
    's': '5',
    'B': '8',
    'Z': '2',
    'z': '2',
    'b': '6',
    'q': '9',
    'g': '9'
}

# Mapping ambiguous digits to their most common uppercase letter counterparts
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '5': 'S',
    '8': 'B',
    '2': 'Z',
    '6': 'G',
    '9': 'q'
}


class HomoglyphDisambiguator:
    """
    Context-aware post-processing engine to disambiguate handwritten homoglyphs.
    """

    def __init__(self, confidence_margin: float = 0.35):
        """
        Args:
            confidence_margin: Maximum confidence difference to allow re-ranking to a top-k alternative.
        """
        self.confidence_margin = confidence_margin

    def disambiguate(
        self,
        predicted_chars: List[str],
        top_k_lists: Optional[List[List[Dict[str, Any]]]] = None,
        mode: str = "auto"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Disambiguate a sequence of predicted characters.

        Args:
            predicted_chars: List of recognized character strings (e.g., ['2', '0', 'O', '4']).
            top_k_lists: Optional list of top-k dictionaries for each position.
            mode: 'auto' (infer from sequence), 'digits_only' (force numeric), 'letters_only' (force alpha), or 'none'.

        Returns:
            resolved_text (str): Final disambiguated string.
            actions (List[Dict[str, Any]]): Log of adjustments made with rationales.
        """
        if not predicted_chars:
            return "", []

        if mode == "none":
            return "".join(predicted_chars), []

        n = len(predicted_chars)
        output_chars = list(predicted_chars)
        actions: List[Dict[str, Any]] = []

        # Count digits vs alphabetic characters in the raw prediction
        num_digits = sum(1 for c in predicted_chars if c.isdigit())
        num_alpha = sum(1 for c in predicted_chars if c.isalpha())
        total_valid = num_digits + num_alpha

        # Determine target modality if in auto mode
        target_mode = mode
        if mode == "auto":
            if total_valid > 0:
                digit_ratio = num_digits / total_valid
                if digit_ratio >= 0.55:
                    target_mode = "digits_only"
                elif digit_ratio <= 0.40:
                    target_mode = "letters_only"
                else:
                    target_mode = "mixed"
            else:
                target_mode = "mixed"

        for idx, char in enumerate(predicted_chars):
            new_char = char
            rationale = None

            # Mode 1: Global numeric context
            if target_mode == "digits_only" and char in LETTER_TO_DIGIT:
                # Check top-k candidates if available to verify if the digit was plausible
                candidate_digit = LETTER_TO_DIGIT[char]
                can_replace = True

                if top_k_lists and idx < len(top_k_lists):
                    top_classes = [
                        (c.get("character") or c.get("class_name") or str(c)) if isinstance(c, dict) else str(c)
                        for c in top_k_lists[idx]
                    ]
                    # Allow replace if candidate is in top-5 or if mode is strictly forced
                    if candidate_digit in top_classes:
                        rationale = f"Re-ranked homoglyph '{char}' to '{candidate_digit}' (candidate in Top-K; sequence is numeric)."
                    else:
                        rationale = f"Mapped ambiguous letter '{char}' to '{candidate_digit}' under numeric sequence constraint."
                else:
                    rationale = f"Mapped ambiguous letter '{char}' to '{candidate_digit}' under numeric sequence constraint."

                if can_replace:
                    new_char = candidate_digit

            # Mode 2: Global alphabetic context
            elif target_mode == "letters_only" and char in DIGIT_TO_LETTER:
                candidate_letter = DIGIT_TO_LETTER[char]
                can_replace = True

                if top_k_lists and idx < len(top_k_lists):
                    top_classes = [
                        (c.get("character") or c.get("class_name") or str(c)) if isinstance(c, dict) else str(c)
                        for c in top_k_lists[idx]
                    ]
                    if candidate_letter in top_classes:
                        rationale = f"Re-ranked homoglyph '{char}' to '{candidate_letter}' (candidate in Top-K; sequence is alphabetic)."
                    else:
                        rationale = f"Mapped ambiguous digit '{char}' to '{candidate_letter}' under alphabetic sequence constraint."
                else:
                    rationale = f"Mapped ambiguous digit '{char}' to '{candidate_letter}' under alphabetic sequence constraint."

                if can_replace:
                    new_char = candidate_letter

            # Mode 3: Local n-gram / neighbor context for mixed strings
            elif target_mode == "mixed":
                prev_char = output_chars[idx - 1] if idx > 0 else None
                next_char = predicted_chars[idx + 1] if idx < n - 1 else None

                # Sandwiched between digits: D-X-D
                if (prev_char and prev_char.isdigit()) and (next_char and next_char.isdigit()):
                    if char in LETTER_TO_DIGIT:
                        new_char = LETTER_TO_DIGIT[char]
                        rationale = f"Sandwiched between digits '{prev_char}' and '{next_char}': resolved '{char}' to '{new_char}'."

                # Sandwiched between letters: L-X-L
                elif (prev_char and prev_char.isalpha()) and (next_char and next_char.isalpha()):
                    if char in DIGIT_TO_LETTER:
                        new_char = DIGIT_TO_LETTER[char]
                        rationale = f"Sandwiched between letters '{prev_char}' and '{next_char}': resolved '{char}' to '{new_char}'."

            if new_char != char:
                output_chars[idx] = new_char
                actions.append({
                    "position": idx + 1,
                    "original": char,
                    "disambiguated": new_char,
                    "rationale": rationale or "Homoglyph disambiguation"
                })

        resolved_text = "".join(output_chars)
        return resolved_text, actions
