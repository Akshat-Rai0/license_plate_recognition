"""Enhanced post-processing with context-aware corrections."""

import re

CHAR_CONFUSIONS = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6',
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G',
}


def filter_valid_characters(plate_text: str) -> str:
    """Filter out invalid characters."""
    filtered = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    return filtered


def correct_by_position(plate_text: str, format_name='generic') -> str:
    """Correct characters based on expected position."""
    corrected = list(plate_text)
    
    for i, char in enumerate(corrected):
        # For Nigerian format: XX##XX####
        if format_name == 'nigerian':
            if i < 2 and char.isdigit():  # Should be letter
                corrected[i] = CHAR_CONFUSIONS.get(char, char)
            elif 2 <= i < 4 and char.isalpha():  # Should be digit
                corrected[i] = CHAR_CONFUSIONS.get(char, char)
            elif 4 <= i < 6 and char.isdigit():  # Should be letter
                corrected[i] = CHAR_CONFUSIONS.get(char, char)
            elif i >= 6 and char.isalpha():  # Should be digit
                corrected[i] = CHAR_CONFUSIONS.get(char, char)
    
    return ''.join(corrected)


def validate_and_correct_plate(plate_text: str, 
                               format_name: str = 'generic',
                               confidence_scores: list = None) -> str:
    """Comprehensive validation and correction."""
    if not plate_text:
        return plate_text
    
    cleaned = filter_valid_characters(plate_text)
    corrected = correct_by_position(cleaned, format_name)
    
    return corrected


def get_alternative_readings(plate_text: str, num_alternatives: int = 3) -> list:
    """Generate alternative readings."""
    alternatives = set()
    alternatives.add(plate_text)
    
    for i in range(len(plate_text)):
        char = plate_text[i]
        if char in CHAR_CONFUSIONS:
            alt_char = CHAR_CONFUSIONS[char]
            alt_text = plate_text[:i] + alt_char + plate_text[i+1:]
            alternatives.add(alt_text)
    
    scored = [(alt, 1.0) for alt in alternatives]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored[:num_alternatives]
