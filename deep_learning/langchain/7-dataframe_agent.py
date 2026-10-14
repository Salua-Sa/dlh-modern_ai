#!/usr/bin/env python3
"""This module sets up a Pandas DataFrame Agent using a LangChain LLM.
"""
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent)


def create_dataframe_agent(llm, df):
    """
    Set up a a Pandas DataFrame Agent using a LangChain LLM.
    The agent should be configured with:
        allow_dangerous_code=True
        verbose=True
        handle_parsing_errors=True

    Args:
        llm: LangChain LLM instance
        df: Pandas DataFrame

    Returns:
        Pandas DataFrame Agent that can process queries on the DataFrame
    """
    agent_dataframe = create_pandas_dataframe_agent(
        llm,
        df,
        allow_dangerous_code=True,
        agent_executor_kwargs={"verbose": True,
                               "handle_parsing_errors": True},
    )

    return agent_dataframe
