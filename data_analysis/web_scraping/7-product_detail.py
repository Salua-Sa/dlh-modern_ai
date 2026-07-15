#!/usr/bin/env python3
"""This module opens a detail page for one product.
"""
import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """
    Opens a detail page for one product.

    Args:
        url: The login page.
        delay: waiting time.

    Return:
         A dictionary.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    time.sleep(delay)

    title = driver.find_elements(
        "css selector",
        ".caption h4")[1].text
    price = driver.find_element(
        "css selector",
        "h4.price").text
    description = driver.find_element(
        "css selector",
        "p.description").text
    rating = len(driver.find_elements(
        "css selector",
        ".ratings .ws-icon.ws-icon-star"))

    return ({
        "title": title,
        "price": price,
        "description": description,
        "rating": rating})

    driver.quit()
