def clean(text: str) -> str:
    """
    Normalises user text exactly the same way in training & inference.
    """
    return text.lower().strip()