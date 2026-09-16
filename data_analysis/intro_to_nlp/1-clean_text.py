#!/usr/bin/env python3
"""
This module cleans and normalises SMS messages.
"""
import re
import emoji


_DATASET_PLACEHOLDER_MAP = {
    '<#>':       '<NUM>',
    '<decimal>': '<NUM>',
    '<time>':    '<TIME>',
    '<url>':     '<URL>',
    '<email>':   '<EMAIL>',
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses, etc. with ASCII equivalents."""
    replacements = {
        r"[''‚‛]":    "'",
        r"[""„‟]":    '"',
        r"[‐‑‒–—―−]": "-",
        r"…":          "...",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    """
    Clean and normalize an SMS message.
    """
    # Return "" for None or not str input
    if text is None or not isinstance(text, str)::
        return ""

    # Convert to lowercase and strip leading/trailing whitespace
    text = text.lower().strip()

    # Remap dataset-native placeholders using _DATASET_PLACEHOLDER_MAP
    for old, new in _DATASET_PLACEHOLDER_MAP.items():
        text = text.replace(old, new)

    # Convert Unicode punctuation to ASCII with normalize_unicode_punct()
    text = normalize_unicode_punct(text)

    # If replace_url=True, replace URLs with<URL>using r'https?://\S+|www\.\S+'
    if replace_url:
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)

    # If replace_num=True, replace numbers in two passes:
    # 1-phone-like strings: r'\+?\d[\d\s\-]{6,}\d'
    if replace_num:
        text = re.sub(r'\+?\d[\d\s\-]{6,}\d', '<NUM>', text)
        # 2-integers, decimals, currency-prefixed amounts
        text = re.sub(r'(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b',
                      '<NUM>', text)

    # Handle emoji using emoji.replace_emoji():
    if emoji_action == "replace":
        text = emoji.replace_emoji(text, replace='<EMO>')
    elif emoji_action == "remove":
        text = emoji.replace_emoji(text, replace=' ')
    elif emoji_action == "keep":
        pass

    # Collapse runs of repeated ! or ? to a single character
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    # Collapse any whitespace sequences to a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text
