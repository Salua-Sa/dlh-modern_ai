#!/usr/bin/env python3
"""This module fetches a web page and returns its HTML as text.
"""
import requests


def fetch_html(url, headers=None, timeout=10):
    """
    Fetches a web page.

    Args:
        url: The page to retrieve.
        headers: An optional dict of HTTP headers.
        timeout: The number of seconds to wait before abording.

    Return:
        The HTML content.
    """
    response = requests.get(
        url,
        headers=headers,
        timeout=timeout
        )
    response.raise_for_status()
    return response.text
