import re

# Comprehensive emoji pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"  # Enclosed characters
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U0001f191-\U0001f251"  # Enclosed Ideographic Supplement
    "\U00002600-\U000026ff"  # Miscellaneous Symbols
    "\U00002700-\U000027bf"  # Dingbats
    "\U0000fe00-\U0000fe0f"  # Variation Selectors
    "\U0001f018-\U0001f270"  # Various asian characters
    "\U00002300-\U000023ff"  # Miscellaneous Technical
    "\U0000200d"  # Zero Width Joiner
    "\U0001f3fb-\U0001f3ff"  # Skin tone modifiers
    "\U00002640-\U00002642"  # Gender symbols
    "\U0000fe0f"  # Variation selector
    "\U000020e3"  # Combining Enclosing Keycap
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(text: str) -> str:
    """
    Remove emojis from text by replacing them with a space,
    then normalizing whitespace while preserving newlines and leading whitespace.

    Not perfect, but will hopefully work ok...
    """
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        # Preserve leading whitespace
        leading_match = re.match(r"^([ \t]*)", line)
        leading_ws = leading_match.group(1) if leading_match else ""

        # Work with the content after leading whitespace
        content = line[len(leading_ws) :]

        # Replace emojis with a single space
        content = EMOJI_PATTERN.sub(" ", content)

        # Normalize whitespace: collapse multiple spaces/tabs into single space
        content = re.sub(r"[ \t]+", " ", content)

        # Remove trailing whitespace
        content = content.rstrip()

        # Remove leading space if content starts with it (but preserve leading_ws)
        content = content.lstrip()

        # Reconstruct line
        result_lines.append(leading_ws + content)

    return str("\n".join(result_lines))
