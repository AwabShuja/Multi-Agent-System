"""
Analyst Agent for the Multi-Agent Virtual Company.

The Analyst agent processes raw research data from the Researcher,
filters noise, identifies trends, and creates structured summaries.
"""

import json
import re
from typing import Optional, Literal
from datetime import datetime
from loguru import logger

from src.agents.base import BaseAgent
from src.graph.state import GraphState
from src.schemas.models import (
    ResearchData,
    AnalysisSummary,
    KeyInsight,
    CritiqueResult,
    AgentMessage,
)
from src.tools.analysis import ResearchDataProcessor, TextAnalyzer
from src.prompts.analyst import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_TASK_PROMPT,
    ANALYST_REVISION_PROMPT,
)


class AnalystAgent(BaseAgent):
    """
    Analyst Agent - Processes research data and creates summaries.
    
    This agent takes raw research data from the Researcher and:
    - Filters out noise and irrelevant information
    - Identifies key trends and patterns
    - Assesses sentiment
    - Creates a structured AnalysisSummary
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.5,  # Balanced for analysis
        max_tokens: int = 4096,
    ):
        """
        Initialize the Analyst agent.
        
        Args:
            api_key: Groq API key
            model_name: LLM model to use
            temperature: LLM temperature
            max_tokens: Maximum tokens for response
        """
        super().__init__(
            name="analyst",
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Initialize analysis tools
        self.data_processor = ResearchDataProcessor()
        self.text_analyzer = TextAnalyzer()
        
        logger.info("AnalystAgent initialized")
    
    @property
    def system_prompt(self) -> str:
        """Return the analyst system prompt."""
        return ANALYST_SYSTEM_PROMPT
    
    def process(self, state: GraphState) -> GraphState:
        """
        Process research data and create analysis summary.
        
        Args:
            state: Current graph state with research_data
            
        Returns:
            Updated state with analysis_summary populated
        """
        self.log_action("Starting analysis")
        
        try:
            # Get research data from state
            research_data = state.get("research_data")
            if not research_data:
                raise ValueError("No research data available for analysis")
            
            # Check if this is a revision (critic feedback exists)
            critique_result = state.get("critique_result")
            if critique_result and critique_result.revision_required:
                analysis_summary = self._perform_revision(
                    state, research_data, critique_result
                )
            else:
                analysis_summary = self._perform_analysis(research_data)
            
            # Create completion message
            message = self.create_message(
                receiver="supervisor",
                message_type="completion",
                content=f"Analysis completed. Sentiment: {analysis_summary.sentiment}, Quality: {analysis_summary.data_quality_score:.2f}",
                metadata={
                    "sentiment": analysis_summary.sentiment,
                    "quality_score": analysis_summary.data_quality_score,
                    "insights_count": len(analysis_summary.key_insights),
                }
            )
            
            self.log_action(
                "Analysis completed",
                f"Sentiment: {analysis_summary.sentiment}, {len(analysis_summary.key_insights)} insights"
            )
            
            # Return updated state
            return {
                **state,
                "analysis_summary": analysis_summary,
                "current_agent": "analyst",
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Analysis failed")
    
    def _perform_analysis(self, research_data: ResearchData) -> AnalysisSummary:
        """
        Perform the main analysis on research data.
        
        Args:
            research_data: Research data from Researcher agent
            
        Returns:
            AnalysisSummary with all analysis results
        """
        logger.info(f"Analyzing research data for: {research_data.topic}")
        
        # Prepare research data for LLM
        formatted_content = self.data_processor.format_for_llm(research_data)
        
        # Get preliminary analysis from text analyzer
        preliminary_data = self.data_processor.prepare_for_analysis(research_data)
        
        # Create analysis prompt
        task_prompt = ANALYST_TASK_PROMPT.format(
            topic=research_data.topic,
            research_content=formatted_content,
        )
        
        # Invoke LLM for analysis
        llm_response = self.invoke_llm(task_prompt)
        
        # Parse the LLM response into AnalysisSummary
        analysis_summary = self._parse_analysis_response(
            llm_response,
            research_data,
            preliminary_data,
        )
        
        return analysis_summary
    
    def _perform_revision(
        self,
        state: GraphState,
        research_data: ResearchData,
        critique_result: CritiqueResult,
    ) -> AnalysisSummary:
        """
        Perform a revision based on critic feedback.
        
        Args:
            state: Current graph state
            research_data: Original research data
            critique_result: Critique with feedback
            
        Returns:
            Revised AnalysisSummary
        """
        logger.info("Performing analysis revision based on critic feedback")
        
        # Get the previous analysis
        previous_analysis = state.get("analysis_summary")
        previous_analysis_str = self._format_previous_analysis(previous_analysis)
        
        # Format feedback
        feedback = "\n".join([
            f"- Weaknesses: {', '.join(critique_result.weaknesses)}",
            f"- Missing Elements: {', '.join(critique_result.missing_elements)}",
            f"- Suggestions: {', '.join(critique_result.suggestions)}",
        ])
        
        # Create revision prompt
        revision_prompt = ANALYST_REVISION_PROMPT.format(
            topic=research_data.topic,
            feedback=feedback,
            revision_instructions=critique_result.revision_instructions or "Address all feedback points",
            previous_analysis=previous_analysis_str,
        )
        
        # Invoke LLM for revision
        llm_response = self.invoke_llm(revision_prompt)
        
        # Parse the revised response
        preliminary_data = self.data_processor.prepare_for_analysis(research_data)
        analysis_summary = self._parse_analysis_response(
            llm_response,
            research_data,
            preliminary_data,
        )
        
        return analysis_summary
    
    def _parse_analysis_response(
        self,
        llm_response: str,
        research_data: ResearchData,
        preliminary_data: dict,
    ) -> AnalysisSummary:
        """
        Parse LLM response into AnalysisSummary.
        
        Args:
            llm_response: Raw LLM response
            research_data: Original research data
            preliminary_data: Preliminary analysis data
            
        Returns:
            AnalysisSummary object
        """
        # Try to extract JSON from response
        json_data = self._extract_json(llm_response)
        
        if json_data:
            return self._create_summary_from_json(
                json_data, research_data, preliminary_data
            )
        else:
            # Fallback: create summary from unstructured response
            logger.warning("Could not parse JSON, creating summary from text")
            return self._create_summary_from_text(
                llm_response, research_data, preliminary_data
            )
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON dict or None
        """
        # Try to find JSON block in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _create_summary_from_json(
        self,
        json_data: dict,
        research_data: ResearchData,
        preliminary_data: dict,
    ) -> AnalysisSummary:
        """
        Create AnalysisSummary from parsed JSON.
        
        Args:
            json_data: Parsed JSON data
            research_data: Original research data
            preliminary_data: Preliminary analysis data
            
        Returns:
            AnalysisSummary object
        """
        # Parse key insights
        key_insights = []
        for insight_data in json_data.get("key_insights", []):
            if isinstance(insight_data, dict):
                insight = KeyInsight(
                    insight=insight_data.get("insight", ""),
                    confidence=insight_data.get("confidence", "medium"),
                    supporting_sources=insight_data.get("supporting_sources", []),
                )
                key_insights.append(insight)
            elif isinstance(insight_data, str):
                key_insights.append(KeyInsight(
                    insight=insight_data,
                    confidence="medium",
                    supporting_sources=[],
                ))
        
        # Validate sentiment
        sentiment = json_data.get("sentiment", "neutral")
        if sentiment not in ["bullish", "bearish", "neutral", "mixed"]:
            sentiment = preliminary_data.get("preliminary_sentiment", "neutral")
        
        # Validate data quality score
        quality_score = json_data.get("data_quality_score", 0.5)
        if not isinstance(quality_score, (int, float)) or not (0 <= quality_score <= 1):
            quality_score = preliminary_data.get("quality_score", 0.5)
        
        return AnalysisSummary(
            topic=research_data.topic,
            executive_summary=json_data.get("executive_summary", "Analysis completed."),
            key_insights=key_insights,
            trends_identified=json_data.get("trends_identified", []),
            sentiment=sentiment,
            data_quality_score=quality_score,
            risks_identified=json_data.get("risks_identified", []),
            opportunities_identified=json_data.get("opportunities_identified", []),
            timestamp=datetime.now(),
        )
    
    def _create_summary_from_text(
        self,
        text: str,
        research_data: ResearchData,
        preliminary_data: dict,
    ) -> AnalysisSummary:
        """
        Create AnalysisSummary from unstructured text.
        
        Fallback when JSON parsing fails.
        
        Args:
            text: Raw text response
            research_data: Original research data
            preliminary_data: Preliminary analysis data
            
        Returns:
            AnalysisSummary object
        """
        # Extract what we can from the text
        lines = text.strip().split("\n")
        
        # Use first few lines as executive summary
        executive_summary = " ".join(lines[:3])[:500] if lines else "Analysis completed."
        
        # Use preliminary analysis as fallback
        sentiment_score, sentiment = self.text_analyzer.calculate_sentiment_score(
            research_data.raw_content
        )
        quality_score = self.text_analyzer.assess_data_quality(research_data)
        
        return AnalysisSummary(
            topic=research_data.topic,
            executive_summary=executive_summary,
            key_insights=[
                KeyInsight(
                    insight="Analysis completed - see executive summary for details",
                    confidence="medium",
                    supporting_sources=[],
                )
            ],
            trends_identified=preliminary_data.get("key_topics", []),
            sentiment=sentiment,
            data_quality_score=quality_score,
            risks_identified=[],
            opportunities_identified=[],
            timestamp=datetime.now(),
        )
    
    def _format_previous_analysis(self, analysis: Optional[AnalysisSummary]) -> str:
        """
        Format previous analysis for revision prompt.
        
        Args:
            analysis: Previous AnalysisSummary
            
        Returns:
            Formatted string
        """
        if not analysis:
            return "No previous analysis available."
        
        insights_str = "\n".join([
            f"- {i.insight} (confidence: {i.confidence})"
            for i in analysis.key_insights
        ])
        
        return f"""
Executive Summary: {analysis.executive_summary}

Key Insights:
{insights_str}

Trends: {', '.join(analysis.trends_identified)}
Sentiment: {analysis.sentiment}
Data Quality: {analysis.data_quality_score}
Risks: {', '.join(analysis.risks_identified)}
Opportunities: {', '.join(analysis.opportunities_identified)}
"""
    
    async def aprocess(self, state: GraphState) -> GraphState:
        """
        Async version of process.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with analysis_summary
        """
        self.log_action("Starting async analysis")
        
        try:
            research_data = state.get("research_data")
            if not research_data:
                raise ValueError("No research data available for analysis")
            
            critique_result = state.get("critique_result")
            if critique_result and critique_result.revision_required:
                analysis_summary = await self._async_perform_revision(
                    state, research_data, critique_result
                )
            else:
                analysis_summary = await self._async_perform_analysis(research_data)
            
            message = self.create_message(
                receiver="supervisor",
                message_type="completion",
                content=f"Analysis completed. Sentiment: {analysis_summary.sentiment}",
                metadata={
                    "sentiment": analysis_summary.sentiment,
                    "quality_score": analysis_summary.data_quality_score,
                }
            )
            
            self.log_action("Async analysis completed")
            
            return {
                **state,
                "analysis_summary": analysis_summary,
                "current_agent": "analyst",
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Async analysis failed")
    
    async def _async_perform_analysis(
        self,
        research_data: ResearchData,
    ) -> AnalysisSummary:
        """
        Async version of analysis.
        
        Args:
            research_data: Research data to analyze
            
        Returns:
            AnalysisSummary
        """
        formatted_content = self.data_processor.format_for_llm(research_data)
        preliminary_data = self.data_processor.prepare_for_analysis(research_data)
        
        task_prompt = ANALYST_TASK_PROMPT.format(
            topic=research_data.topic,
            research_content=formatted_content,
        )
        
        llm_response = await self.ainvoke_llm(task_prompt)
        
        return self._parse_analysis_response(
            llm_response, research_data, preliminary_data
        )
    
    async def _async_perform_revision(
        self,
        state: GraphState,
        research_data: ResearchData,
        critique_result: CritiqueResult,
    ) -> AnalysisSummary:
        """
        Async version of revision.
        
        Args:
            state: Current state
            research_data: Research data
            critique_result: Critique feedback
            
        Returns:
            Revised AnalysisSummary
        """
        previous_analysis = state.get("analysis_summary")
        previous_analysis_str = self._format_previous_analysis(previous_analysis)
        
        feedback = "\n".join([
            f"- Weaknesses: {', '.join(critique_result.weaknesses)}",
            f"- Missing Elements: {', '.join(critique_result.missing_elements)}",
            f"- Suggestions: {', '.join(critique_result.suggestions)}",
        ])
        
        revision_prompt = ANALYST_REVISION_PROMPT.format(
            topic=research_data.topic,
            feedback=feedback,
            revision_instructions=critique_result.revision_instructions or "Address all feedback",
            previous_analysis=previous_analysis_str,
        )
        
        llm_response = await self.ainvoke_llm(revision_prompt)
        preliminary_data = self.data_processor.prepare_for_analysis(research_data)
        
        return self._parse_analysis_response(
            llm_response, research_data, preliminary_data
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_analyst_agent(
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.5,
) -> AnalystAgent:
    """
    Factory function to create a configured AnalystAgent.
    
    Args:
        groq_api_key: Groq API key
        model_name: LLM model name
        temperature: LLM temperature
        
    Returns:
        Configured AnalystAgent instance
    """
    return AnalystAgent(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
    )


__all__ = [
    "AnalystAgent",
    "create_analyst_agent",
]
