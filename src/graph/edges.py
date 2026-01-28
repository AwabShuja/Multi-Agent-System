"""
LangGraph Edge Functions for the Multi-Agent Virtual Company.

This module defines conditional edge functions that determine
the flow between nodes based on the current state.

Edges control the routing logic:
- Which agent to route to next
- Whether to continue or end the workflow
- How to handle errors and retries
"""

from typing import Literal, Union
from loguru import logger

from src.graph.state import GraphState, AgentType


# =============================================================================
# Edge Return Types
# =============================================================================

# Possible routing destinations
RouteDestination = Literal["researcher", "analyst", "critic", "writer", "end", "error"]


# =============================================================================
# Primary Routing Edge (From Supervisor)
# =============================================================================

def route_from_supervisor(state: GraphState) -> RouteDestination:
    """
    Determine next node based on supervisor's decision.
    
    This is the primary routing function called after the supervisor
    node to direct flow to the appropriate worker agent.
    
    Args:
        state: Current graph state (with next_agent set by supervisor)
        
    Returns:
        Destination node name
    """
    # Check for errors first
    if state.get("error"):
        logger.warning(f"Error detected, routing to error handler: {state['error']}")
        return "error"
    
    # Check workflow status
    if state.get("workflow_status") == "failed":
        logger.warning("Workflow failed, routing to error handler")
        return "error"
    
    # Get supervisor's routing decision
    next_agent = state.get("next_agent")
    
    # Handle END routing - can be explicit "END"/"end" or None when workflow is complete
    if next_agent is None or next_agent == "END" or next_agent == "end":
        # Check if we have a final report to determine if workflow should end
        if state.get("final_report"):
            logger.info("Routing to END node - workflow complete")
            return "end"
        else:
            logger.error("Supervisor did not set next_agent and no final report available")
            return "error"
    
    # Validate and route to worker agent
    valid_agents = {"researcher", "analyst", "critic", "writer"}
    
    if next_agent in valid_agents:
        logger.info(f"Routing to {next_agent}")
        return next_agent
    
    logger.error(f"Invalid next_agent: {next_agent}")
    return "error"


# =============================================================================
# Worker Agent Return Edges (Back to Supervisor)
# =============================================================================

def route_after_researcher(state: GraphState) -> Literal["supervisor", "error"]:
    """
    Route after researcher completes.
    
    Always returns to supervisor for next decision,
    unless there's an error.
    
    Args:
        state: Current graph state
        
    Returns:
        "supervisor" or "error"
    """
    if state.get("error"):
        logger.warning("Researcher encountered error")
        return "error"
    
    if state.get("research_data") is None:
        logger.warning("Researcher did not produce research_data")
        return "error"
    
    logger.info("Researcher completed, returning to supervisor")
    return "supervisor"


def route_after_analyst(state: GraphState) -> Literal["supervisor", "error"]:
    """
    Route after analyst completes.
    
    Always returns to supervisor for critique routing,
    unless there's an error.
    
    Args:
        state: Current graph state
        
    Returns:
        "supervisor" or "error"
    """
    if state.get("error"):
        logger.warning("Analyst encountered error")
        return "error"
    
    if state.get("analysis_summary") is None:
        logger.warning("Analyst did not produce analysis_summary")
        return "error"
    
    logger.info("Analyst completed, returning to supervisor")
    return "supervisor"


def route_after_critic(state: GraphState) -> Literal["supervisor", "error"]:
    """
    Route after critic completes.
    
    Returns to supervisor which will decide:
    - If approved: route to writer
    - If rejected: route back to analyst (if iterations remain)
    - If max iterations: route to writer anyway
    
    Args:
        state: Current graph state
        
    Returns:
        "supervisor" or "error"
    """
    if state.get("error"):
        logger.warning("Critic encountered error")
        return "error"
    
    if state.get("critique_result") is None:
        logger.warning("Critic did not produce critique_result")
        return "error"
    
    # Log critique decision
    critique = state.get("critique_result")
    if hasattr(critique, 'is_approved'):
        status = "APPROVED" if critique.is_approved else "REVISION NEEDED"
        logger.info(f"Critic decision: {status}")
    
    logger.info("Critic completed, returning to supervisor")
    return "supervisor"


def route_after_writer(state: GraphState) -> Literal["supervisor", "error"]:
    """
    Route after writer completes.
    
    Returns to supervisor which will end the workflow
    since the final report is ready.
    
    Args:
        state: Current graph state
        
    Returns:
        "supervisor" or "error"
    """
    if state.get("error"):
        logger.warning("Writer encountered error")
        return "error"
    
    if state.get("final_report") is None:
        logger.warning("Writer did not produce final_report")
        return "error"
    
    logger.info("Writer completed, returning to supervisor")
    return "supervisor"


# =============================================================================
# Shortcut Edge Functions (For Direct Routing Option)
# =============================================================================

def should_continue_workflow(state: GraphState) -> Literal["continue", "end", "error"]:
    """
    Determine if workflow should continue.
    
    This is an alternative routing approach that can be used
    for simpler workflow patterns.
    
    Args:
        state: Current graph state
        
    Returns:
        "continue", "end", or "error"
    """
    # Check for errors
    if state.get("error"):
        return "error"
    
    # Check if workflow is complete
    if state.get("workflow_status") == "completed":
        return "end"
    
    if state.get("final_report") is not None:
        return "end"
    
    return "continue"


def should_revise_analysis(state: GraphState) -> Literal["revise", "proceed", "force_proceed"]:
    """
    Determine if analysis needs revision.
    
    Based on critic's decision and iteration count.
    
    Args:
        state: Current graph state
        
    Returns:
        "revise", "proceed", or "force_proceed"
    """
    critique = state.get("critique_result")
    
    if critique is None:
        logger.warning("No critique result available")
        return "proceed"
    
    # Check if approved
    if hasattr(critique, 'is_approved') and critique.is_approved:
        logger.info("Analysis approved by critic")
        return "proceed"
    
    # Check iteration count
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, forcing proceed")
        return "force_proceed"
    
    logger.info(f"Analysis needs revision (iteration {iteration + 1}/{max_iterations})")
    return "revise"


# =============================================================================
# Error Handling Edge
# =============================================================================

def route_on_error(state: GraphState) -> Literal["retry", "end", "escalate"]:
    """
    Determine how to handle errors.
    
    Based on error type and retry count.
    
    Args:
        state: Current graph state
        
    Returns:
        "retry", "end", or "escalate"
    """
    error = state.get("error")
    error_agent = state.get("error_agent")
    
    if error is None:
        logger.warning("route_on_error called without error")
        return "end"
    
    logger.error(f"Error from {error_agent}: {error}")
    
    # For now, always end on error
    # Future: implement retry logic based on error type
    return "end"


# =============================================================================
# Composite Routing Function
# =============================================================================

def create_router(
    default_route: str = "supervisor"
) -> callable:
    """
    Factory function to create custom routers.
    
    Args:
        default_route: Default destination if no specific route matches
        
    Returns:
        Router function
    """
    def router(state: GraphState) -> str:
        """Custom router based on state."""
        # Check for errors
        if state.get("error"):
            return "error"
        
        # Check for completion
        if state.get("workflow_status") == "completed":
            return "end"
        
        # Use supervisor's decision if available
        next_agent = state.get("next_agent")
        if next_agent:
            if next_agent == "END":
                return "end"
            return next_agent
        
        return default_route
    
    return router


# =============================================================================
# Edge Configuration Helper
# =============================================================================

class EdgeConfig:
    """
    Configuration for workflow edges.
    
    Provides a structured way to define conditional edges.
    """
    
    @staticmethod
    def supervisor_edges() -> dict:
        """
        Get edge configuration for supervisor node.
        
        Returns:
            Dictionary mapping conditions to destinations
        """
        return {
            "researcher": "researcher",
            "analyst": "analyst",
            "critic": "critic",
            "writer": "writer",
            "end": "end",
            "error": "error",
        }
    
    @staticmethod
    def worker_return_edge() -> dict:
        """
        Get edge configuration for worker nodes returning to supervisor.
        
        Returns:
            Dictionary mapping conditions to destinations
        """
        return {
            "supervisor": "supervisor",
            "error": "error",
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "RouteDestination",
    # Primary routing
    "route_from_supervisor",
    # Worker return edges
    "route_after_researcher",
    "route_after_analyst",
    "route_after_critic",
    "route_after_writer",
    # Utility edges
    "should_continue_workflow",
    "should_revise_analysis",
    "route_on_error",
    # Factories
    "create_router",
    "EdgeConfig",
]
