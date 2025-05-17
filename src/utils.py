import unicodedata

def normalize_text(text):
    """Normalize Unicode characters for Sinhala/Tamil text."""
    return unicodedata.normalize('NFC', text)