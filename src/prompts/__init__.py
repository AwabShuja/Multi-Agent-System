"""
System prompts for each agent in the Multi-Agent Virtual Company.

Each agent has specialized prompts that define their role and behavior.
"""

from .researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_TASK_PROMPT,
    RESEARCHER_SEARCH_PROMPT,
)
from .analyst import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_TASK_PROMPT,
    ANALYST_REVISION_PROMPT,
)

__all__ = [
    # Researcher prompts
    "RESEARCHER_SYSTEM_PROMPT",
    "RESEARCHER_TASK_PROMPT",
    "RESEARCHER_SEARCH_PROMPT",
    # Analyst prompts
    "ANALYST_SYSTEM_PROMPT",
    "ANALYST_TASK_PROMPT",
    "ANALYST_REVISION_PROMPT",
]
