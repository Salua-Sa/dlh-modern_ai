#!/usr/bin/env python3
"""This module loads documents from a folder
and returns them as a list using LangChain.
"""
from langchain_community import document_loaders


def load_documents(folder_path, file_pattern):
    """
    Load documents from a folder and returns them as a list using LangChain.

    Args:
        folder_path: A string representing the path to the folder.
        file_pattern: A string representing the glob pattern to match files.

    Returns:
        list: A list of LangChain document objects loaded from the folder.
    """
    loader = document_loaders.DirectoryLoader(
        path=folder_path,
        glob=file_pattern,
        loader_cls=document_loaders.TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    load_documents = loader.load()

    return load_documents
