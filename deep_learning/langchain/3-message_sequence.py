#!/usr/bin/env python3
"""This module  prepares a sequence of messages combining
a system message with one or more human messages.
"""
from langchain_core import messages


def setup_message_sequence(system_msg, human_msgs):
    """
    Prepare a sequence of messages combining a system message
    with one or more human messages.

    Args:
        system_msg: A string defining the system’s role or behavior.
        human_msgs: A list of strings representing human inputs.

    Returns:
        A list containing the SystemMessage followed by HumanMessage objects.
    """
    messages_list = [messages.SystemMessage(content=system_msg)]

    for h_msg in human_msgs:
        messages_list.append(messages.HumanMessage(content=h_msg))

    return messages_list
