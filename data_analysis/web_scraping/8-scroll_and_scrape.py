#!/usr/bin/env python3
"""This module scrolls and extracts all products
from a JS‐rendered infinite‐scroll page.
"""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """
    Opens a detail page for one product.

    Args:
        url: The login page.
        scroll_pause: waiting time.

    Return:
         A dictionary.
    """

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)

    driver.get(url)

    last_height = driver.execute_script(
        "return document.body.scrollHeight")

    while True:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        start_time = time.time()
        while time.time() - start_time < scroll_pause:
            new_height = driver.execute_script(
                "return document.body.scrollHeight")
            if new_height > last_height:
                break
            time.sleep(0.1)

        new_height = driver.execute_script(
            "return document.body.scrollHeight"
            )
        if new_height == last_height:
            break
        last_height = new_height

    products = driver.find_elements(
        "css selector",
        "div.thumbnail"
        )

    unique_products = []
    seen_key = set()

    for product in products:
        title = product.find_element(
            "css selector",
            "a.title").get_attribute("title")
        price = product.find_element(
            "css selector",
            "h4.price").text
        description = product.find_element(
            "css selector",
            "p.description").text
        rating = len(product.find_elements(
            "css selector",
            ".ratings .ws-icon.ws-icon-star"))
        product_key = (title, price)
        if product_key not in seen_key:
            seen_key.add(product_key)
            unique_products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating})

    driver.quit()
    return unique_products
