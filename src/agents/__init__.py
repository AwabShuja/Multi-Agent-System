"""
Agent implementations for the Multi-Agent Virtual Company.

This package provides all agent classes:
- BaseAgent: Abstract base class for all agents
- ToolEnabledAgent: Base class for agents with tools
- ResearcherAgent: Web research using Tavily
- AnalystAgent: Data analysis and summarization
- CriticAgent: Quality review and feedback
- WriterAgent: Final report generation
- SupervisorAgent: Workflow orchestration
"""

from .base import BaseAgent, ToolEnabledAgent, create_llm
from .researcher import ResearcherAgent, create_researcher_agent
from .analyst import AnalystAgent, create_analyst_agent
from .critic import CriticAgent, create_critic_agent
from .writer import WriterAgent, create_writer_agent
from .supervisor import SupervisorAgent, create_supervisor_agent

__all__ = [
    "BaseAgent",
    "ToolEnabledAgent",
    "create_llm",
    "ResearcherAgent",
    "create_researcher_agent",
    "AnalystAgent",
    "create_analyst_agent",
    "CriticAgent",
    "create_critic_agent",
    "WriterAgent",
    "create_writer_agent",
    "SupervisorAgent",
    "create_supervisor_agent",
]
