"""
Tokenizer — Simple word-level tokenizer for Thai + English mixed text.
"""
import re

def tokenize_thai(text: str):
    """
    Simple word-level tokenizer for Thai + English mixed text.
    Filters out short tokens and lowercases English text.
    """
    if not text:
        return []
        
    tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]
