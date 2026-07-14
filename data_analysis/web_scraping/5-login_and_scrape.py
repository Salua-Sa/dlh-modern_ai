#!/usr/bin/env python3
"""This module logs in and scrapes quotes visible only after authentication.
"""
import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """
    Logs in and scrapes quotes visible after authentication.

    Args:
        login_url: The login page.
        user: The user name.
        pwd: The password.

    Return:
         A list of quote dicts.
    """
    list_quote_dicts = []

    session = requests.Session()

    login_response = session.get(login_url)
    login_response.raise_for_status()

    soup = BeautifulSoup(
        login_response.text, "html.parser")

    csrf_token = soup.find(
        "input",
        attrs={"name": "csrf_token"}
        )["value"]

    payload = {
        "username": user,
        "password": pwd,
        "csrf_token": csrf_token}

    login_result = session.post(
        login_url,
        data=payload)
    login_result.raise_for_status()

    quotes_response = session.get(
        "https://quotes.toscrape.com/")
    quotes_response.raise_for_status()

    soup = BeautifulSoup(
        quotes_response.text,
        "html.parser")

    quotes_blocks = soup.find_all(
        "div",
        class_="quote")

    for quote in quotes_blocks:
        text = quote.find(
            "span",
            class_="text"
            ).get_text(strip=True)
        author = quote.find(
            "small",
            class_="author"
            ).get_text(strip=True)
        tags = []
        tag = quote.find_all("a", class_="tag")
        for t in tag:
            tag_text = t.get_text(strip=True)
            tags.append(tag_text)

        list_quote_dicts.append({
            "text": text,
            "author": author,
            "tags": tags
            })

    return list_quote_dicts
