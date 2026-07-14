#!/usr/bin/env python3
"""This module scraps quotes through the site's API.
"""
import json
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """
    Fetches quote data from all the quotes' API pages.

    Args:
        base_url: The root URL of the site.

    Return:
         A list of quote dicts.
    """
    list_quote_dicts = []
    page = 1

    while True:
        api_url = f"{base_url}/api/quotes?page={page}"
        json_text = fetch_html(api_url)

        dic_text = json.loads(json_text)

        for quote in dic_text["quotes"]:
            list_quote_dicts.append({
                "text": quote["text"],
                "author": quote["author"]["name"],
                "tags": quote["tags"]
                })
        if dic_text["has_next"]:
            page += 1
        else:
            break

    return list_quote_dicts
