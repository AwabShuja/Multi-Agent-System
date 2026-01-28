"""
LangGraph Node Functions for the Multi-Agent Virtual Company.

This module defines node functions that wrap each agent's process method.
Nodes are the fundamental units of computation in LangGraph.

Each node:
1. Receives the current state
2. Invokes the corresponding agent
3. Returns state updates
"""

from typing import Callable, Optional
from datetime import datetime
from loguru import logger

from src.graph.state import GraphState, AgentType
from src.schemas.models import AgentMessage
from src.agents import (
    ResearcherAgent,
    AnalystAgent,
    CriticAgent,
    WriterAgent,
    SupervisorAgent,
)


# =============================================================================
# Node Function Type
# =============================================================================

NodeFunction = Callable[[GraphState], GraphState]


# =============================================================================
# Agent Instance Cache
# =============================================================================

class AgentRegistry:
    """
    Registry to manage agent instances.
    
    Uses lazy initialization to create agents only when needed.
    Supports dependency injection for testing.
    """
    
    _instance: Optional["AgentRegistry"] = None
    
    def __init__(self, api_key: str, tavily_api_key: str):
        """
        Initialize the agent registry.
        
        Args:
            api_key: Groq API key for all agents
            tavily_api_key: Tavily API key for researcher
        """
        self.api_key = api_key
        self.tavily_api_key = tavily_api_key
        self._agents: dict[AgentType, object] = {}
        logger.info("AgentRegistry initialized")
    
    @classmethod
    def get_instance(cls, api_key: Optional[str] = None, tavily_api_key: Optional[str] = None) -> "AgentRegistry":
        """
        Get the singleton registry instance.
        
        Args:
            api_key: Groq API key (required on first call)
            tavily_api_key: Tavily API key (required on first call)
            
        Returns:
            AgentRegistry instance
        """
        if cls._instance is None:
            if api_key is None or tavily_api_key is None:
                raise ValueError("Both API keys required for first initialization")
            cls._instance = cls(api_key, tavily_api_key)
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
    
    def get_researcher(self) -> ResearcherAgent:
        """Get or create the Researcher agent."""
        if "researcher" not in self._agents:
            self._agents["researcher"] = ResearcherAgent(
                api_key=self.api_key,
                tavily_api_key=self.tavily_api_key,
            )
        return self._agents["researcher"]
    
    def get_analyst(self) -> AnalystAgent:
        """Get or create the Analyst agent."""
        if "analyst" not in self._agents:
            self._agents["analyst"] = AnalystAgent(api_key=self.api_key)
        return self._agents["analyst"]
    
    def get_critic(self) -> CriticAgent:
        """Get or create the Critic agent."""
        if "critic" not in self._agents:
            self._agents["critic"] = CriticAgent(api_key=self.api_key)
        return self._agents["critic"]
    
    def get_writer(self) -> WriterAgent:
        """Get or create the Writer agent."""
        if "writer" not in self._agents:
            self._agents["writer"] = WriterAgent(api_key=self.api_key)
        return self._agents["writer"]
    
    def get_supervisor(self) -> SupervisorAgent:
        """Get or create the Supervisor agent."""
        if "supervisor" not in self._agents:
            self._agents["supervisor"] = SupervisorAgent(api_key=self.api_key)
        return self._agents["supervisor"]


# =============================================================================
# Global Registry Access
# =============================================================================

def initialize_registry(api_key: str, tavily_api_key: str) -> AgentRegistry:
    """
    Initialize the global agent registry.
    
    Must be called before running the workflow.
    
    Args:
        api_key: Groq API key
        tavily_api_key: Tavily API key for research
        
    Returns:
        Initialized AgentRegistry
    """
    AgentRegistry.reset()  # Clear any existing instance
    return AgentRegistry.get_instance(api_key, tavily_api_key)


def get_registry() -> AgentRegistry:
    """
    Get the global agent registry.
    
    Returns:
        AgentRegistry instance
        
    Raises:
        ValueError: If registry not initialized
    """
    return AgentRegistry.get_instance()


# =============================================================================
# Node Functions
# =============================================================================

def supervisor_node(state: GraphState) -> GraphState:
    """
    Supervisor node - Orchestrates the workflow.
    
    Determines which agent should act next based on:
    - Current workflow state
    - Available data
    - Iteration count
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with next_agent set
    """
    logger.info("=" * 50)
    logger.info("SUPERVISOR NODE")
    logger.info("=" * 50)
    
    try:
        registry = get_registry()
        supervisor = registry.get_supervisor()
        
        # Update current agent
        state = {**state, "current_agent": "supervisor"}
        
        # Process and get routing decision
        updated_state = supervisor.process(state)
        
        logger.info(f"Supervisor decided next agent: {updated_state.get('next_agent')}")
        return updated_state
        
    except Exception as e:
        logger.error(f"Supervisor node error: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "supervisor",
            "workflow_status": "failed",
        }


def researcher_node(state: GraphState) -> GraphState:
    """
    Researcher node - Conducts web research.
    
    Uses Tavily API to search for relevant information
    based on the user's query.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with research_data
    """
    logger.info("=" * 50)
    logger.info("RESEARCHER NODE")
    logger.info("=" * 50)
    
    try:
        registry = get_registry()
        researcher = registry.get_researcher()
        
        # Update current agent
        state = {**state, "current_agent": "researcher"}
        
        # Process research
        updated_state = researcher.process(state)
        
        # Log research summary
        if updated_state.get("research_data"):
            rd = updated_state["research_data"]
            source_count = len(rd.sources) if hasattr(rd, 'sources') else 0
            logger.info(f"Research completed: {source_count} sources found")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Researcher node error: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "researcher",
            "workflow_status": "failed",
        }


def analyst_node(state: GraphState) -> GraphState:
    """
    Analyst node - Analyzes research data.
    
    Takes the research data and produces a structured
    analysis summary with key insights.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with analysis_summary
    """
    logger.info("=" * 50)
    logger.info("ANALYST NODE")
    logger.info("=" * 50)
    
    try:
        registry = get_registry()
        analyst = registry.get_analyst()
        
        # Update current agent
        state = {**state, "current_agent": "analyst"}
        
        # Check if this is a revision
        is_revision = state.get("critique_result") is not None
        if is_revision:
            logger.info("Analyst performing revision based on critique")
        
        # Process analysis
        updated_state = analyst.process(state)
        
        # Log analysis summary
        if updated_state.get("analysis_summary"):
            summary = updated_state["analysis_summary"]
            insight_count = len(summary.key_insights) if hasattr(summary, 'key_insights') else 0
            logger.info(f"Analysis completed: {insight_count} key insights")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Analyst node error: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "analyst",
            "workflow_status": "failed",
        }


def critic_node(state: GraphState) -> GraphState:
    """
    Critic node - Reviews analysis quality.
    
    Evaluates the analysis for completeness, accuracy,
    and potential bias. Can approve or request revision.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with critique_result
    """
    logger.info("=" * 50)
    logger.info("CRITIC NODE")
    logger.info("=" * 50)
    
    try:
        registry = get_registry()
        critic = registry.get_critic()
        
        # Update current agent
        state = {**state, "current_agent": "critic"}
        
        # Process critique
        updated_state = critic.process(state)
        
        # Log critique result
        if updated_state.get("critique_result"):
            critique = updated_state["critique_result"]
            status = "APPROVED" if critique.is_approved else "REVISION REQUESTED"
            score = critique.quality_score if hasattr(critique, 'quality_score') else 0
            logger.info(f"Critique completed: {status} (score: {score:.2f})")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Critic node error: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "critic",
            "workflow_status": "failed",
        }


def writer_node(state: GraphState) -> GraphState:
    """
    Writer node - Generates final report.
    
    Takes the approved analysis and produces a polished,
    professional report.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with final_report
    """
    logger.info("=" * 50)
    logger.info("WRITER NODE")
    logger.info("=" * 50)
    
    try:
        registry = get_registry()
        writer = registry.get_writer()
        
        # Update current agent
        state = {**state, "current_agent": "writer"}
        
        # Process report generation
        updated_state = writer.process(state)
        
        # Log report completion
        if updated_state.get("final_report"):
            report = updated_state["final_report"]
            section_count = len(report.sections) if hasattr(report, 'sections') else 0
            logger.info(f"Report completed: {section_count} sections")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Writer node error: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "writer",
            "workflow_status": "failed",
        }


def end_node(state: GraphState) -> GraphState:
    """
    End node - Finalizes the workflow.
    
    Marks the workflow as completed and records the completion time.
    
    Args:
        state: Current graph state
        
    Returns:
        Final state with workflow marked complete
    """
    logger.info("=" * 50)
    logger.info("END NODE - Workflow Complete")
    logger.info("=" * 50)
    
    return {
        **state,
        "workflow_status": "completed",
        "completed_at": datetime.now(),
    }


# =============================================================================
# Node Function Mapping
# =============================================================================

NODE_MAPPING: dict[str, NodeFunction] = {
    "supervisor": supervisor_node,
    "researcher": researcher_node,
    "analyst": analyst_node,
    "critic": critic_node,
    "writer": writer_node,
    "end": end_node,
}


def get_node_function(agent_type: str) -> NodeFunction:
    """
    Get the node function for an agent type.
    
    Args:
        agent_type: Agent type string
        
    Returns:
        Node function
        
    Raises:
        KeyError: If agent type not found
    """
    if agent_type not in NODE_MAPPING:
        raise KeyError(f"Unknown agent type: {agent_type}")
    return NODE_MAPPING[agent_type]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Registry
    "AgentRegistry",
    "initialize_registry",
    "get_registry",
    # Node functions
    "supervisor_node",
    "researcher_node",
    "analyst_node",
    "critic_node",
    "writer_node",
    "end_node",
    # Mapping
    "NODE_MAPPING",
    "get_node_function",
    # Type
    "NodeFunction",
]
