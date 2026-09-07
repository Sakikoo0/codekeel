def parse(text):
    if not text.strip():
        raise ValueError("empty input")
    return text.strip()
