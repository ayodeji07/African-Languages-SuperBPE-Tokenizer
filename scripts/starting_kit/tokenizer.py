class Tokenizer:
    """Minimal lossless baseline; replace it with your SuperBPE tokenizer."""

    def encode(self, texts: list[str]) -> list[list[int]]:
        return [
            [ord(character) + 1 for character in text] + [0]
            for text in texts
        ]

    def decode(self, encoded_texts: list[list[int]]) -> list[str]:
        return [
            "".join(chr(token - 1) for token in tokens if token != 0)
            for tokens in encoded_texts
        ]
