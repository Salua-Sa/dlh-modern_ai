#!/usr/bin/env python3
""" This module creates a high-level interface for performing
language translation using a pre-trained large language model.
"""
import transformers


def translate_text(model_name, src_lang, tgt_lang):
    """
    Create a high-level interface for performing language translation
    using a pre-trained large language model.

    Args:
        model_name: Name of the pre-trained model to use.
        src_lang: Source language code (e.g., "en" for English).
        tgt_lang: Target language code (e.g., "fr" for French).

    Returns:
        translator: A Hugging Face pipeline object.
    """
    translator = transformers.pipeline("translation",
                                       model=model_name,
                                       src_lang=src_lang,
                                       tgt_lang=tgt_lang)

    return translator
