"""
Post-processing utilities for license plate recognition.
Includes character validation and correction based on common patterns.
"""

def validate_and_correct_plate(plate_text: str) -> str:
    """
    Validate and correct common character confusions in license plate text.
    Based on common OCR errors and Nigerian plate format patterns.
    """
    if not plate_text:
        return plate_text
    
    # Common character confusions based on shape similarity
    corrections = {
        'V': 'L',  # V often confused with L
        'G': 'C',  # G often confused with C  
        '9': '2',  # 9 often confused with 2
        'S': '5',  # S often confused with 5
        'O': '0',  # O often confused with 0 (in number positions)
        'I': '1',  # I often confused with 1
        'Z': '2',  # Z often confused with 2
        'B': '8',  # B sometimes confused with 8
        'D': '0',  # D sometimes confused with 0
    }
    
    # Nigerian plate format: XX##XX#### (2 letters, 2 numbers, 2 letters, 4 numbers)
    # But we'll be flexible and just correct obvious errors
    corrected = list(plate_text)
    
    for i, char in enumerate(corrected):
        if char in corrections:
            # Apply correction
            corrected[i] = corrections[char]
    
    return ''.join(corrected)


def filter_valid_characters(plate_text: str) -> str:
    """
    Filter out invalid characters, keeping only alphanumeric.
    """
    import re
    # Keep only alphanumeric characters
    filtered = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    return filtered


def apply_plate_format_rules(plate_text: str) -> str:
    """
    Apply format rules for Nigerian license plates if applicable.
    Format: XX##XX#### (but we'll be flexible)
    """
    # Remove spaces and special characters
    cleaned = filter_valid_characters(plate_text)
    
    # If too short or too long, return as is (might be partial detection)
    if len(cleaned) < 4 or len(cleaned) > 12:
        return cleaned
    
    return cleaned
