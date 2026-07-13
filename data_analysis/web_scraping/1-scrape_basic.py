#!/usr/bin/env python3
"""This module crapes the first page of quotes from a static web page.
"""
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """
    Scrapes the first page of quotes from quotes.toscrape.com.

    Args:
        url: The Quotes List endpoin.

    Return:
         A list of dicts.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    list_dic = []

    quote_block = soup.find_all("div", class_="quote")

    for quote in quote_block:
        text = quote.find("span", class_="text").get_text(strip=True)
        author = quote.find("small", class_="author").get_text(strip=True)
        tags = []
        tag = quote.find_all("a", class_="tag")
        for t in tag:
            tag_text = t.get_text(strip=True)
            tags.append(tag_text)

        list_dic.append({
            "text": text,
            "author": author,
            "tags": tags
            })

    return list_dic
