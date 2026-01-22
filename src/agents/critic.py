"""
Critic Agent for the Multi-Agent Virtual Company.

The Critic agent reviews the Analyst's work for quality,
bias, and completeness, providing feedback for improvement.
"""

import json
import re
from typing import Optional
from datetime import datetime
from loguru import logger

from src.agents.base import BaseAgent
from src.graph.state import GraphState
from src.schemas.models import (
    AnalysisSummary,
    CritiqueResult,
    AgentMessage,
)
from src.prompts.critic import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_TASK_PROMPT,
    CRITIC_ITERATION_WARNING,
)


class CriticAgent(BaseAgent):
    """
    Critic Agent - Reviews analysis quality and provides feedback.
    
    This agent evaluates the Analyst's output for:
    - Completeness and accuracy
    - Potential bias
    - Missing elements
    - Overall quality
    
    It can approve the analysis or request revisions.
    """
    
    # Quality threshold for approval (0-1 scale)
    APPROVAL_THRESHOLD = 0.70
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,  # Lower temperature for consistent evaluation
        max_tokens: int = 4096,
        approval_threshold: float = 0.70,
    ):
        """
        Initialize the Critic agent.
        
        Args:
            api_key: Groq API key
            model_name: LLM model to use
            temperature: LLM temperature
            max_tokens: Maximum tokens for response
            approval_threshold: Minimum quality score for approval (0-1)
        """
        super().__init__(
            name="critic",
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self.approval_threshold = approval_threshold
        logger.info(f"CriticAgent initialized (approval threshold: {approval_threshold})")
    
    @property
    def system_prompt(self) -> str:
        """Return the critic system prompt."""
        return CRITIC_SYSTEM_PROMPT
    
    def process(self, state: GraphState) -> GraphState:
        """
        Review the analysis and provide feedback.
        
        Args:
            state: Current graph state with analysis_summary
            
        Returns:
            Updated state with critique_result populated
        """
        self.log_action("Starting critique review")
        
        try:
            # Get analysis summary from state
            analysis_summary = state.get("analysis_summary")
            if not analysis_summary:
                raise ValueError("No analysis summary available for review")
            
            # Get research data for context
            research_data = state.get("research_data")
            
            # Get iteration info
            iteration_count = state.get("iteration_count", 0) + 1
            max_iterations = state.get("max_iterations", 3)
            
            # Perform the critique
            critique_result = self._perform_critique(
                analysis_summary,
                research_data,
                iteration_count,
                max_iterations,
                state.get("messages", []),
            )
            
            # Determine message based on result
            if critique_result.is_approved:
                message_content = f"Analysis APPROVED. Quality score: {critique_result.quality_score:.2f}"
            else:
                message_content = f"Analysis needs REVISION. Quality score: {critique_result.quality_score:.2f}"
            
            message = self.create_message(
                receiver="supervisor",
                message_type="feedback",
                content=message_content,
                metadata={
                    "is_approved": critique_result.is_approved,
                    "quality_score": critique_result.quality_score,
                    "revision_required": critique_result.revision_required,
                    "iteration": iteration_count,
                }
            )
            
            self.log_action(
                "Critique completed",
                f"Approved: {critique_result.is_approved}, Score: {critique_result.quality_score:.2f}"
            )
            
            # Return updated state
            return {
                **state,
                "critique_result": critique_result,
                "current_agent": "critic",
                "iteration_count": iteration_count,
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Critique failed")
    
    def _perform_critique(
        self,
        analysis_summary: AnalysisSummary,
        research_data: Optional[any],
        iteration_count: int,
        max_iterations: int,
        previous_messages: list[AgentMessage],
    ) -> CritiqueResult:
        """
        Perform the critique of the analysis.
        
        Args:
            analysis_summary: Analysis to review
            research_data: Original research data for context
            iteration_count: Current iteration number
            max_iterations: Maximum allowed iterations
            previous_messages: Previous messages for context
            
        Returns:
            CritiqueResult with feedback
        """
        logger.info(f"Critiquing analysis (iteration {iteration_count}/{max_iterations})")
        
        # Format the analysis for review
        formatted_analysis = self._format_analysis_for_review(
            analysis_summary, research_data
        )
        
        # Build the critique prompt
        critique_prompt = formatted_analysis
        
        # Add iteration context if not first iteration
        if iteration_count > 1:
            previous_feedback = self._extract_previous_feedback(previous_messages)
            iteration_warning = CRITIC_ITERATION_WARNING.format(
                iteration=iteration_count,
                max_iterations=max_iterations,
                previous_feedback=previous_feedback,
            )
            critique_prompt = f"{iteration_warning}\n\n{critique_prompt}"
        
        # Invoke LLM for critique
        llm_response = self.invoke_llm(critique_prompt)
        
        # Parse the response
        critique_result = self._parse_critique_response(
            llm_response,
            iteration_count,
            max_iterations,
        )
        
        return critique_result
    
    def _format_analysis_for_review(
        self,
        analysis: AnalysisSummary,
        research_data: Optional[any],
    ) -> str:
        """
        Format analysis summary for critique review.
        
        Args:
            analysis: Analysis summary to format
            research_data: Research data for context
            
        Returns:
            Formatted prompt string
        """
        # Format key insights
        insights_str = "\n".join([
            f"- {i.insight} (confidence: {i.confidence})"
            for i in analysis.key_insights
        ]) if analysis.key_insights else "No insights provided"
        
        # Format trends
        trends_str = "\n".join([f"- {t}" for t in analysis.trends_identified]) \
            if analysis.trends_identified else "No trends identified"
        
        # Format risks
        risks_str = "\n".join([f"- {r}" for r in analysis.risks_identified]) \
            if analysis.risks_identified else "No risks identified"
        
        # Format opportunities
        opportunities_str = "\n".join([f"- {o}" for o in analysis.opportunities_identified]) \
            if analysis.opportunities_identified else "No opportunities identified"
        
        # Get research context
        sources_count = research_data.sources_count if research_data else "Unknown"
        research_quality = f"{analysis.data_quality_score:.2f}" if analysis.data_quality_score else "Unknown"
        
        return CRITIC_TASK_PROMPT.format(
            topic=analysis.topic,
            executive_summary=analysis.executive_summary,
            key_insights=insights_str,
            trends=trends_str,
            sentiment=analysis.sentiment,
            data_quality_score=analysis.data_quality_score,
            risks=risks_str,
            opportunities=opportunities_str,
            sources_count=sources_count,
            research_quality=research_quality,
        )
    
    def _extract_previous_feedback(self, messages: list[AgentMessage]) -> str:
        """
        Extract previous critique feedback from messages.
        
        Args:
            messages: List of agent messages
            
        Returns:
            Summary of previous feedback
        """
        feedback_parts = []
        for msg in messages:
            if msg.sender == "critic" and msg.message_type == "feedback":
                feedback_parts.append(msg.content)
        
        if feedback_parts:
            return "\n".join(feedback_parts)
        return "No previous feedback recorded."
    
    def _parse_critique_response(
        self,
        llm_response: str,
        iteration_count: int,
        max_iterations: int,
    ) -> CritiqueResult:
        """
        Parse LLM response into CritiqueResult.
        
        Args:
            llm_response: Raw LLM response
            iteration_count: Current iteration
            max_iterations: Max iterations
            
        Returns:
            CritiqueResult object
        """
        # Try to extract JSON
        json_data = self._extract_json(llm_response)
        
        if json_data:
            return self._create_critique_from_json(
                json_data, iteration_count, max_iterations
            )
        else:
            logger.warning("Could not parse JSON, creating critique from text")
            return self._create_critique_from_text(
                llm_response, iteration_count, max_iterations
            )
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON dict or None
        """
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _create_critique_from_json(
        self,
        json_data: dict,
        iteration_count: int,
        max_iterations: int,
    ) -> CritiqueResult:
        """
        Create CritiqueResult from parsed JSON.
        
        Args:
            json_data: Parsed JSON data
            iteration_count: Current iteration
            max_iterations: Max iterations
            
        Returns:
            CritiqueResult object
        """
        # Get quality score
        quality_score = json_data.get("quality_score", 0.5)
        if isinstance(quality_score, (int, float)):
            # Normalize if given as 0-100
            if quality_score > 1:
                quality_score = quality_score / 100
            quality_score = max(0.0, min(1.0, quality_score))
        else:
            quality_score = 0.5
        
        # Determine approval
        is_approved = json_data.get("is_approved", quality_score >= self.approval_threshold)
        
        # Force approval if max iterations reached and score is reasonable
        if iteration_count >= max_iterations and quality_score >= 0.5:
            is_approved = True
            revision_required = False
        else:
            revision_required = json_data.get("revision_required", not is_approved)
        
        return CritiqueResult(
            is_approved=is_approved,
            quality_score=quality_score,
            strengths=json_data.get("strengths", []),
            weaknesses=json_data.get("weaknesses", []),
            missing_elements=json_data.get("missing_elements", []),
            bias_detected=json_data.get("bias_detected", False),
            bias_details=json_data.get("bias_details"),
            suggestions=json_data.get("suggestions", []),
            revision_required=revision_required,
            revision_instructions=json_data.get("revision_instructions") if revision_required else None,
            timestamp=datetime.now(),
        )
    
    def _create_critique_from_text(
        self,
        text: str,
        iteration_count: int,
        max_iterations: int,
    ) -> CritiqueResult:
        """
        Create CritiqueResult from unstructured text.
        
        Fallback when JSON parsing fails.
        
        Args:
            text: Raw text response
            iteration_count: Current iteration
            max_iterations: Max iterations
            
        Returns:
            CritiqueResult object
        """
        text_lower = text.lower()
        
        # Try to determine approval from text
        is_approved = any(word in text_lower for word in ["approve", "approved", "acceptable", "good quality"])
        needs_revision = any(word in text_lower for word in ["revision", "revise", "improve", "needs work"])
        
        if needs_revision:
            is_approved = False
        
        # Estimate quality score from text sentiment
        quality_score = 0.75 if is_approved else 0.55
        
        # Force approval at max iterations
        if iteration_count >= max_iterations:
            is_approved = True
            needs_revision = False
        
        return CritiqueResult(
            is_approved=is_approved,
            quality_score=quality_score,
            strengths=["Analysis completed"],
            weaknesses=["Detailed feedback could not be parsed"] if not is_approved else [],
            missing_elements=[],
            bias_detected=False,
            bias_details=None,
            suggestions=["Review the full critique response for details"],
            revision_required=needs_revision,
            revision_instructions=text[:500] if needs_revision else None,
            timestamp=datetime.now(),
        )
    
    async def aprocess(self, state: GraphState) -> GraphState:
        """
        Async version of process.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with critique_result
        """
        self.log_action("Starting async critique review")
        
        try:
            analysis_summary = state.get("analysis_summary")
            if not analysis_summary:
                raise ValueError("No analysis summary available for review")
            
            research_data = state.get("research_data")
            iteration_count = state.get("iteration_count", 0) + 1
            max_iterations = state.get("max_iterations", 3)
            
            critique_result = await self._async_perform_critique(
                analysis_summary,
                research_data,
                iteration_count,
                max_iterations,
                state.get("messages", []),
            )
            
            message_content = (
                f"Analysis APPROVED. Score: {critique_result.quality_score:.2f}"
                if critique_result.is_approved
                else f"Analysis needs REVISION. Score: {critique_result.quality_score:.2f}"
            )
            
            message = self.create_message(
                receiver="supervisor",
                message_type="feedback",
                content=message_content,
                metadata={
                    "is_approved": critique_result.is_approved,
                    "quality_score": critique_result.quality_score,
                }
            )
            
            self.log_action("Async critique completed")
            
            return {
                **state,
                "critique_result": critique_result,
                "current_agent": "critic",
                "iteration_count": iteration_count,
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Async critique failed")
    
    async def _async_perform_critique(
        self,
        analysis_summary: AnalysisSummary,
        research_data: Optional[any],
        iteration_count: int,
        max_iterations: int,
        previous_messages: list[AgentMessage],
    ) -> CritiqueResult:
        """
        Async version of critique.
        
        Args:
            analysis_summary: Analysis to review
            research_data: Research data context
            iteration_count: Current iteration
            max_iterations: Max iterations
            previous_messages: Previous messages
            
        Returns:
            CritiqueResult
        """
        formatted_analysis = self._format_analysis_for_review(
            analysis_summary, research_data
        )
        
        critique_prompt = formatted_analysis
        
        if iteration_count > 1:
            previous_feedback = self._extract_previous_feedback(previous_messages)
            iteration_warning = CRITIC_ITERATION_WARNING.format(
                iteration=iteration_count,
                max_iterations=max_iterations,
                previous_feedback=previous_feedback,
            )
            critique_prompt = f"{iteration_warning}\n\n{critique_prompt}"
        
        llm_response = await self.ainvoke_llm(critique_prompt)
        
        return self._parse_critique_response(
            llm_response, iteration_count, max_iterations
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_critic_agent(
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    approval_threshold: float = 0.70,
) -> CriticAgent:
    """
    Factory function to create a configured CriticAgent.
    
    Args:
        groq_api_key: Groq API key
        model_name: LLM model name
        temperature: LLM temperature
        approval_threshold: Minimum quality score for approval
        
    Returns:
        Configured CriticAgent instance
    """
    return CriticAgent(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        approval_threshold=approval_threshold,
    )


__all__ = [
    "CriticAgent",
    "create_critic_agent",
]
