#!/usr/bin/env python3
"""This module pulls quotes from embedded JSON‑LD on a page.
"""
import json
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """
    Pulls quotes from embedded JSON‑LD on a page.

    Args:
        url: The Quotes List endpoint.

    Return:
         A list of quote dicts.
    """
    list_quote_dicts = []

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    scripts = soup.find_all(
        "script",
        type="application/ld+json")

    for script in scripts:
        dic_text = json.loads(script.string)

        if dic_text.get("@type") == "Quote":
            text = dic_text.get("text")
            author = dic_text.get("author", {}).get("name")
            tags = dic_text.get("keywords", "")

        list_quote_dicts.append({
            "text": text,
            "author": author,
            "tags": tags
            })

    return list_quote_dicts
