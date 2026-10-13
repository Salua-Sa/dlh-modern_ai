#!/usr/bin/env python3
"""This module sets up a LangChain agent executor capable
of using multiple tools to answer user queries.
"""
from langchain import agents
from langchain_community.agent_toolkits import load_tools


def load_agent_with_toolkit(llm, toolkit_names, prompt):
    """
    Set up a LangChain agent executor capable of using multiple
    tools to answer user queries.

    Args:
        llm: A LangChain LLM instance
        toolkit_names: List of tool names to load
        prompt: A ChatPromptTemplate instance

    Returns:
        AgentExecutor: An agent executor ready to handle queries.
    """
    # Load tools: Make all tools in toolkit_names available to the agent.
    tools = load_tools.load_tools(toolkit_names, llm=llm)

    # Create tool-calling agent: Initialize an agent using the given LLM
    # and prompt with the loaded tools.
    agent = agents.create_tool_calling_agent(llm, tools, prompt)

    # Wrap in AgentExecutor: Encapsulate the agent to handle queries,
    # call tools dynamically, and show detailed logs with verbose=True.
    agent_executor = agents.AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    return agent_executor
