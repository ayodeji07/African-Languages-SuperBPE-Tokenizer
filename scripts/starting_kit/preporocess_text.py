import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_ASCII_PUNCTUATION_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "–": "-",
    "—": "-",
    "―": "-",
    "…": "...",
    "፣": ",",
    "።": ".",
    "፤": ";",
    "፥": ":",
    "፦": ":",
    "፧": "?",
    "፨": ".",
})


def _ascii_character_description(character: str, prefix: str = "") -> str:
    """Describe a single non-ASCII character using only ASCII."""
    name = unicodedata.name(character, "UNNAMED CHARACTER")
    return f"{prefix}[U+{ord(character):04X} {name}]"


def _force_ascii_without_loss(text: str) -> str:
    """Keep ASCII text and describe any remaining non-ASCII characters."""
    ascii_parts = []
    for character in text:
        if character.isascii():
            ascii_parts.append(character if character.isprintable() else " ")
        else:
            ascii_parts.append(_ascii_character_description(character))
    return "".join(ascii_parts)


def _get_unidecode():
    """Return an installed transliterator, if one is available."""
    try:
        from unidecode import unidecode

        return unidecode
    except ImportError:
        try:
            from text_unidecode import unidecode

            return unidecode
        except ImportError:
            return None


def _transliterate_to_ascii(text: str) -> str:
    """Return a readable ASCII approximation of multilingual text."""
    unidecode = _get_unidecode()
    ascii_parts = []

    for character in text:
        if character.isascii():
            ascii_parts.append(character if character.isprintable() else " ")
            continue

        transliterated = unidecode(character) if unidecode else ""
        transliterated = _force_ascii_without_loss(transliterated).strip()
        if transliterated:
            ascii_parts.append(_ascii_character_description(character, transliterated))
        else:
            decomposed = unicodedata.normalize("NFKD", character)
            ascii_equivalent = decomposed.encode("ascii", errors="ignore").decode(
                "ascii"
            )
            ascii_parts.append(
                _ascii_character_description(character, ascii_equivalent)
            )

    return "".join(ascii_parts)


def preprocess_text(text: str | bytes) -> str:
    """Normalize multilingual text to readable ASCII with single spaces.

    Accented Latin text is folded to plain ASCII, and non-Latin scripts are
    transliterated when Unidecode or text-unidecode is installed. Every
    non-ASCII source character also keeps an explicit Unicode code point/name
    marker so preprocessing does not silently drop information.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    else:
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ASCII_PUNCTUATION_TRANSLATION)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _transliterate_to_ascii(text)
    text = _force_ascii_without_loss(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
