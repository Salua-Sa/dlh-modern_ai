#!/usr/bin/env python3
"""This module follows scraps quotes from multiple paginated web pages.
"""
from bs4 import BeautifulSoup
import time
from urllib import parse
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """
    Scrape quotes from all pages by following each Next link.

    Args:
        base_url: The first page URL.

    Return:
         A list of dicts.
    """
    list_dic = []
    current_url = base_url
    while current_url:
        current_url_quote = scrape_basic(current_url)
        list_dic.extend(current_url_quote)
        html = fetch_html(current_url)
        soup = BeautifulSoup(html, "html.parser")
        next_button = soup.find("li", class_="next")

        if next_button is None:
            break

        next_href = next_button.find("a")["href"]
        next_url = parse.urljoin(current_url, next_href)

        time.sleep(1)
        current_url = next_url

    return list_dic
