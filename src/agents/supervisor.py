"""
Supervisor Agent for the Multi-Agent Virtual Company.

The Supervisor agent orchestrates the workflow, deciding which agent
should act next based on the current state.
"""

import json
import re
from typing import Optional, Literal
from datetime import datetime
from loguru import logger

from src.agents.base import BaseAgent
from src.graph.state import GraphState, AgentType
from src.schemas.models import AgentMessage
from src.prompts.supervisor import (
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_ROUTING_PROMPT,
    SUPERVISOR_ERROR_PROMPT,
)


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - Orchestrates the multi-agent workflow.
    
    This agent decides which agent should act next based on:
    - Current state of the workflow
    - Available data
    - Iteration count
    - Any errors or issues
    """
    
    # Valid agent routing options
    VALID_AGENTS = {"researcher", "analyst", "critic", "writer", "END"}
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,  # Low temperature for consistent routing
        max_tokens: int = 1024,
    ):
        """
        Initialize the Supervisor agent.
        
        Args:
            api_key: Groq API key
            model_name: LLM model to use
            temperature: LLM temperature (low for consistency)
            max_tokens: Maximum tokens for response
        """
        super().__init__(
            name="supervisor",
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        logger.info("SupervisorAgent initialized")
    
    @property
    def system_prompt(self) -> str:
        """Return the supervisor system prompt."""
        return SUPERVISOR_SYSTEM_PROMPT
    
    def process(self, state: GraphState) -> GraphState:
        """
        Determine the next agent and update state.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with next_agent set
        """
        self.log_action("Evaluating workflow state")
        
        try:
            # Check for errors first
            if state.get("error"):
                return self._handle_error_state(state)
            
            # Determine next agent
            routing_decision = self._decide_next_agent(state)
            
            next_agent = routing_decision.get("next_agent", "END")
            reasoning = routing_decision.get("reasoning", "")
            instructions = routing_decision.get("instructions", "")
            
            # Validate the decision
            if next_agent not in self.VALID_AGENTS:
                logger.warning(f"Invalid agent '{next_agent}', defaulting to END")
                next_agent = "END"
            
            # Create supervisor message
            message = self.create_message(
                receiver=next_agent if next_agent != "END" else "all",
                message_type="instruction",
                content=f"Routing to {next_agent}: {reasoning}",
                metadata={
                    "next_agent": next_agent,
                    "instructions": instructions,
                }
            )
            
            self.log_action(f"Routing to {next_agent}", reasoning)
            
            # Update state
            new_state = {
                **state,
                "current_agent": "supervisor",
                "next_agent": next_agent if next_agent != "END" else None,
                "messages": [message],
            }
            
            # Mark as completed if ending
            if next_agent == "END":
                new_state["workflow_status"] = "completed"
                new_state["completed_at"] = datetime.now()
            
            return new_state
            
        except Exception as e:
            return self.handle_error(state, e, "Supervisor routing failed")
    
    def _decide_next_agent(self, state: GraphState) -> dict:
        """
        Decide which agent should act next.
        
        Uses rule-based logic first, falls back to LLM for edge cases.
        
        Args:
            state: Current graph state
            
        Returns:
            Dictionary with next_agent, reasoning, instructions
        """
        # Rule-based routing for standard workflow
        decision = self._rule_based_routing(state)
        
        if decision:
            return decision
        
        # Fall back to LLM for complex decisions
        return self._llm_based_routing(state)
    
    def _rule_based_routing(self, state: GraphState) -> Optional[dict]:
        """
        Apply rule-based routing logic.
        
        Args:
            state: Current graph state
            
        Returns:
            Routing decision or None if LLM needed
        """
        research_data = state.get("research_data")
        analysis_summary = state.get("analysis_summary")
        critique_result = state.get("critique_result")
        final_report = state.get("final_report")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # Rule 1: If we have a final report, we're done
        if final_report:
            return {
                "next_agent": "END",
                "reasoning": "Final report is complete",
                "instructions": None,
            }
        
        # Rule 2: No research data yet - start with researcher
        if not research_data:
            return {
                "next_agent": "researcher",
                "reasoning": "Starting workflow - need to gather research data",
                "instructions": f"Research the topic: {state.get('user_query', '')}",
            }
        
        # Rule 3: Have research but no analysis - go to analyst
        if research_data and not analysis_summary:
            return {
                "next_agent": "analyst",
                "reasoning": "Research data available, need analysis",
                "instructions": "Analyze the research data and create summary",
            }
        
        # Rule 4: Have analysis but no critique - go to critic
        if analysis_summary and not critique_result:
            return {
                "next_agent": "critic",
                "reasoning": "Analysis ready for quality review",
                "instructions": "Review the analysis for quality and completeness",
            }
        
        # Rule 5: Critique exists - check if approved
        if critique_result:
            if critique_result.is_approved:
                return {
                    "next_agent": "writer",
                    "reasoning": "Analysis approved by critic, ready for report",
                    "instructions": "Create the final polished report",
                }
            else:
                # Check iteration limit
                if iteration_count >= max_iterations:
                    return {
                        "next_agent": "writer",
                        "reasoning": f"Max iterations ({max_iterations}) reached, proceeding to report",
                        "instructions": "Create report with current analysis (note limitations)",
                    }
                else:
                    return {
                        "next_agent": "analyst",
                        "reasoning": "Critic requested revision",
                        "instructions": critique_result.revision_instructions or "Revise based on feedback",
                    }
        
        # If no rule matches, use LLM
        return None
    
    def _llm_based_routing(self, state: GraphState) -> dict:
        """
        Use LLM for complex routing decisions.
        
        Args:
            state: Current graph state
            
        Returns:
            Routing decision dictionary
        """
        logger.debug("Using LLM for routing decision")
        
        # Format the prompt
        prompt = self._format_routing_prompt(state)
        
        # Invoke LLM
        llm_response = self.invoke_llm(prompt)
        
        # Parse response
        return self._parse_routing_response(llm_response)
    
    def _format_routing_prompt(self, state: GraphState) -> str:
        """
        Format the routing prompt with current state.
        
        Args:
            state: Current graph state
            
        Returns:
            Formatted prompt string
        """
        # Get latest message
        messages = state.get("messages", [])
        latest_message = "No messages yet"
        if messages:
            latest = messages[-1] if isinstance(messages[-1], AgentMessage) else messages[-1]
            if isinstance(latest, dict):
                latest_message = f"From {latest.get('sender', 'unknown')}: {latest.get('content', '')}"
            else:
                latest_message = f"From {latest.sender}: {latest.content}"
        
        return SUPERVISOR_ROUTING_PROMPT.format(
            user_query=state.get("user_query", "No query"),
            workflow_status=state.get("workflow_status", "in_progress"),
            iteration_count=state.get("iteration_count", 0),
            max_iterations=state.get("max_iterations", 3),
            has_research="Yes" if state.get("research_data") else "No",
            has_analysis="Yes" if state.get("analysis_summary") else "No",
            has_critique="Yes" if state.get("critique_result") else "No",
            has_report="Yes" if state.get("final_report") else "No",
            latest_message=latest_message,
        )
    
    def _parse_routing_response(self, llm_response: str) -> dict:
        """
        Parse LLM routing response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Routing decision dictionary
        """
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*?\}', llm_response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                next_agent = data.get("next_agent", "END")
                
                # Validate agent name
                if next_agent.lower() not in [a.lower() for a in self.VALID_AGENTS]:
                    next_agent = "END"
                
                return {
                    "next_agent": next_agent.lower(),
                    "reasoning": data.get("reasoning", "LLM decision"),
                    "instructions": data.get("instructions"),
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract agent name from text
        response_lower = llm_response.lower()
        for agent in ["researcher", "analyst", "critic", "writer", "end"]:
            if agent in response_lower:
                return {
                    "next_agent": agent if agent != "end" else "END",
                    "reasoning": "Extracted from LLM response",
                    "instructions": None,
                }
        
        # Default to END if can't parse
        return {
            "next_agent": "END",
            "reasoning": "Could not parse routing decision, ending workflow",
            "instructions": None,
        }
    
    def _handle_error_state(self, state: GraphState) -> GraphState:
        """
        Handle workflow error state.
        
        Args:
            state: State with error
            
        Returns:
            Updated state with error handling
        """
        error = state.get("error", "Unknown error")
        error_agent = state.get("error_agent", "unknown")
        iteration_count = state.get("iteration_count", 0)
        
        self.log_action("Handling error", f"From {error_agent}: {error}")
        
        # Try LLM for error recovery decision
        error_prompt = SUPERVISOR_ERROR_PROMPT.format(
            error=error,
            error_agent=error_agent,
            iteration_count=iteration_count,
        )
        
        try:
            llm_response = self.invoke_llm(error_prompt)
            decision = self._parse_routing_response(llm_response)
        except Exception:
            # If LLM fails too, just end
            decision = {
                "next_agent": "END",
                "reasoning": "Error recovery failed, ending workflow",
                "instructions": None,
            }
        
        message = self.create_message(
            receiver="all",
            message_type="error",
            content=f"Error recovery: routing to {decision['next_agent']}",
            metadata={"original_error": error, "error_agent": error_agent},
        )
        
        return {
            **state,
            "current_agent": "supervisor",
            "next_agent": decision["next_agent"] if decision["next_agent"] != "END" else None,
            "workflow_status": "failed" if decision["next_agent"] == "END" else "in_progress",
            "messages": [message],
            "error": None,  # Clear error after handling
        }
    
    def get_workflow_summary(self, state: GraphState) -> str:
        """
        Get a human-readable workflow summary.
        
        Args:
            state: Current graph state
            
        Returns:
            Summary string
        """
        status = state.get("workflow_status", "unknown")
        iterations = state.get("iteration_count", 0)
        
        steps_completed = []
        if state.get("research_data"):
            steps_completed.append("Research")
        if state.get("analysis_summary"):
            steps_completed.append("Analysis")
        if state.get("critique_result"):
            steps_completed.append("Critique")
        if state.get("final_report"):
            steps_completed.append("Report")
        
        return (
            f"Workflow Status: {status}\n"
            f"Iterations: {iterations}\n"
            f"Steps Completed: {', '.join(steps_completed) or 'None'}"
        )
    
    async def aprocess(self, state: GraphState) -> GraphState:
        """
        Async version of process.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with next_agent
        """
        self.log_action("Async evaluating workflow state")
        
        try:
            if state.get("error"):
                return await self._async_handle_error_state(state)
            
            routing_decision = self._decide_next_agent(state)
            
            next_agent = routing_decision.get("next_agent", "END")
            reasoning = routing_decision.get("reasoning", "")
            
            if next_agent not in self.VALID_AGENTS:
                next_agent = "END"
            
            message = self.create_message(
                receiver=next_agent if next_agent != "END" else "all",
                message_type="instruction",
                content=f"Routing to {next_agent}: {reasoning}",
                metadata={"next_agent": next_agent},
            )
            
            self.log_action(f"Async routing to {next_agent}")
            
            new_state = {
                **state,
                "current_agent": "supervisor",
                "next_agent": next_agent if next_agent != "END" else None,
                "messages": [message],
            }
            
            if next_agent == "END":
                new_state["workflow_status"] = "completed"
                new_state["completed_at"] = datetime.now()
            
            return new_state
            
        except Exception as e:
            return self.handle_error(state, e, "Async supervisor routing failed")
    
    async def _async_handle_error_state(self, state: GraphState) -> GraphState:
        """
        Async error handling.
        
        Args:
            state: State with error
            
        Returns:
            Updated state
        """
        error = state.get("error", "Unknown error")
        error_agent = state.get("error_agent", "unknown")
        
        self.log_action("Async handling error", f"From {error_agent}")
        
        # For async, use simple rule-based recovery
        decision = {
            "next_agent": "END",
            "reasoning": "Error occurred, ending workflow",
            "instructions": None,
        }
        
        message = self.create_message(
            receiver="all",
            message_type="error",
            content=f"Workflow ended due to error: {error}",
            metadata={"error_agent": error_agent},
        )
        
        return {
            **state,
            "current_agent": "supervisor",
            "next_agent": None,
            "workflow_status": "failed",
            "messages": [message],
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_supervisor_agent(
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
) -> SupervisorAgent:
    """
    Factory function to create a configured SupervisorAgent.
    
    Args:
        groq_api_key: Groq API key
        model_name: LLM model name
        temperature: LLM temperature
        
    Returns:
        Configured SupervisorAgent instance
    """
    return SupervisorAgent(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
    )


__all__ = [
    "SupervisorAgent",
    "create_supervisor_agent",
]
