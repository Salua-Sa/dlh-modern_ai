#!/usr/bin/env python3
"""This module opens a static product category page
and returns a list of product dictionaries.
"""
import time
from selenium import webdriver


def scrape_products_list(url):
    """
    Opens a static product category page.

    Args:
        url: The login page.

    Return:
         A list of product dictionaries.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)

    driver.get(url)

    products = driver.find_elements(
        "css selector",
        "div.thumbnail"
        )

    list_product_dict = []
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
        rating = int(product.find_element(
            "css selector",
            ".ratings [data-rating]"
            ).get_attribute("data-rating"))
        list_product_dict.append({
            "title": title,
            "price": price,
            "description": description,
            "rating": rating
            })

    driver.quit()
    return list_product_dict
