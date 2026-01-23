"""
Writer Agent for the Multi-Agent Virtual Company.

The Writer agent produces the final polished report from
the approved analysis.
"""

import json
import re
from typing import Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.agents.base import BaseAgent
from src.graph.state import GraphState
from src.schemas.models import (
    AnalysisSummary,
    CritiqueResult,
    FinalReport,
    ReportSection,
    ResearchData,
    AgentMessage,
)
from src.prompts.writer import (
    WRITER_SYSTEM_PROMPT,
    WRITER_TASK_PROMPT,
    WRITER_QUICK_REPORT_PROMPT,
)


class WriterAgent(BaseAgent):
    """
    Writer Agent - Produces the final polished report.
    
    This agent takes the approved analysis and creates a
    professional, well-structured report ready for delivery.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,  # Higher for creative writing
        max_tokens: int = 4096,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the Writer agent.
        
        Args:
            api_key: Groq API key
            model_name: LLM model to use
            temperature: LLM temperature (higher for creativity)
            max_tokens: Maximum tokens for response
            output_dir: Directory to save reports (optional)
        """
        super().__init__(
            name="writer",
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self.output_dir = output_dir
        logger.info("WriterAgent initialized")
    
    @property
    def system_prompt(self) -> str:
        """Return the writer system prompt."""
        return WRITER_SYSTEM_PROMPT
    
    def process(self, state: GraphState) -> GraphState:
        """
        Generate the final report from approved analysis.
        
        Args:
            state: Current graph state with analysis_summary
            
        Returns:
            Updated state with final_report populated
        """
        self.log_action("Starting report generation")
        
        try:
            # Get analysis summary from state
            analysis_summary = state.get("analysis_summary")
            if not analysis_summary:
                raise ValueError("No analysis summary available for report")
            
            # Get additional context
            research_data = state.get("research_data")
            critique_result = state.get("critique_result")
            
            # Generate the report
            final_report = self._generate_report(
                analysis_summary,
                research_data,
                critique_result,
            )
            
            # Save report if output directory is configured
            if self.output_dir:
                self._save_report(final_report)
            
            # Create completion message
            message = self.create_message(
                receiver="supervisor",
                message_type="completion",
                content=f"Report generated: '{final_report.title}' ({final_report.word_count} words)",
                metadata={
                    "title": final_report.title,
                    "word_count": final_report.word_count,
                    "sections_count": len(final_report.sections),
                }
            )
            
            self.log_action(
                "Report generated",
                f"'{final_report.title}' - {final_report.word_count} words"
            )
            
            # Return updated state
            return {
                **state,
                "final_report": final_report,
                "current_agent": "writer",
                "workflow_status": "completed",
                "completed_at": datetime.now(),
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Report generation failed")
    
    def _generate_report(
        self,
        analysis_summary: AnalysisSummary,
        research_data: Optional[ResearchData],
        critique_result: Optional[CritiqueResult],
    ) -> FinalReport:
        """
        Generate the final report.
        
        Args:
            analysis_summary: Approved analysis
            research_data: Original research data
            critique_result: Critique feedback (if any)
            
        Returns:
            FinalReport object
        """
        logger.info(f"Generating report for: {analysis_summary.topic}")
        
        # Format the prompt
        task_prompt = self._format_task_prompt(
            analysis_summary, research_data, critique_result
        )
        
        # Invoke LLM
        llm_response = self.invoke_llm(task_prompt)
        
        # Parse response into FinalReport
        final_report = self._parse_report_response(
            llm_response,
            analysis_summary,
            research_data,
        )
        
        return final_report
    
    def _format_task_prompt(
        self,
        analysis: AnalysisSummary,
        research_data: Optional[ResearchData],
        critique: Optional[CritiqueResult],
    ) -> str:
        """
        Format the task prompt for report generation.
        
        Args:
            analysis: Analysis summary
            research_data: Research data
            critique: Critique result
            
        Returns:
            Formatted prompt string
        """
        # Format insights
        insights_str = "\n".join([
            f"- {i.insight} (confidence: {i.confidence})"
            for i in analysis.key_insights
        ]) if analysis.key_insights else "No specific insights"
        
        # Format trends
        trends_str = "\n".join([f"- {t}" for t in analysis.trends_identified]) \
            if analysis.trends_identified else "No trends identified"
        
        # Format risks
        risks_str = "\n".join([f"- {r}" for r in analysis.risks_identified]) \
            if analysis.risks_identified else "No specific risks noted"
        
        # Format opportunities
        opportunities_str = "\n".join([f"- {o}" for o in analysis.opportunities_identified]) \
            if analysis.opportunities_identified else "No specific opportunities noted"
        
        # Format critique notes
        critique_notes = "None"
        if critique:
            notes_parts = []
            if critique.strengths:
                notes_parts.append(f"Strengths: {', '.join(critique.strengths)}")
            if critique.suggestions:
                notes_parts.append(f"Suggestions incorporated: {', '.join(critique.suggestions)}")
            critique_notes = "\n".join(notes_parts) if notes_parts else "Analysis approved without notes"
        
        # Get sources count
        sources_count = research_data.sources_count if research_data else "Unknown"
        
        return WRITER_TASK_PROMPT.format(
            topic=analysis.topic,
            executive_summary=analysis.executive_summary,
            key_insights=insights_str,
            trends=trends_str,
            sentiment=analysis.sentiment,
            risks=risks_str,
            opportunities=opportunities_str,
            data_quality_score=f"{analysis.data_quality_score:.2f}",
            sources_count=sources_count,
            critique_notes=critique_notes,
        )
    
    def _parse_report_response(
        self,
        llm_response: str,
        analysis: AnalysisSummary,
        research_data: Optional[ResearchData],
    ) -> FinalReport:
        """
        Parse LLM response into FinalReport.
        
        Args:
            llm_response: Raw LLM response
            analysis: Original analysis
            research_data: Research data
            
        Returns:
            FinalReport object
        """
        # Try to extract JSON
        json_data = self._extract_json(llm_response)
        
        if json_data:
            return self._create_report_from_json(json_data, analysis, research_data)
        else:
            logger.warning("Could not parse JSON, creating report from text")
            return self._create_report_from_text(llm_response, analysis, research_data)
    
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
        
        # Try raw JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _create_report_from_json(
        self,
        json_data: dict,
        analysis: AnalysisSummary,
        research_data: Optional[ResearchData],
    ) -> FinalReport:
        """
        Create FinalReport from parsed JSON.
        
        Args:
            json_data: Parsed JSON data
            analysis: Original analysis
            research_data: Research data
            
        Returns:
            FinalReport object
        """
        # Parse sections
        sections = []
        for section_data in json_data.get("sections", []):
            if isinstance(section_data, dict):
                section = ReportSection(
                    title=section_data.get("title", "Section"),
                    content=section_data.get("content", ""),
                )
                sections.append(section)
        
        # If no sections, create from executive summary
        if not sections:
            sections = [
                ReportSection(
                    title="Overview",
                    content=json_data.get("executive_summary", analysis.executive_summary),
                )
            ]
        
        # Get sources from research data
        sources = json_data.get("sources", [])
        if not sources and research_data:
            sources = [r.url for r in research_data.search_results[:10]]
        
        # Calculate word count
        all_content = json_data.get("executive_summary", "") + " "
        all_content += " ".join([s.content for s in sections])
        word_count = len(all_content.split())
        
        return FinalReport(
            title=json_data.get("title", f"Research Report: {analysis.topic}"),
            topic=analysis.topic,
            executive_summary=json_data.get("executive_summary", analysis.executive_summary),
            sections=sections,
            key_takeaways=json_data.get("key_takeaways", []),
            recommendations=json_data.get("recommendations", []),
            sources=sources,
            generated_at=datetime.now(),
            word_count=word_count,
        )
    
    def _create_report_from_text(
        self,
        text: str,
        analysis: AnalysisSummary,
        research_data: Optional[ResearchData],
    ) -> FinalReport:
        """
        Create FinalReport from unstructured text.
        
        Fallback when JSON parsing fails.
        
        Args:
            text: Raw text response
            analysis: Original analysis
            research_data: Research data
            
        Returns:
            FinalReport object
        """
        # Parse markdown sections
        sections = self._parse_markdown_sections(text)
        
        # If no sections parsed, use text as single section
        if not sections:
            sections = [
                ReportSection(
                    title="Research Findings",
                    content=text[:3000],  # Limit content length
                )
            ]
        
        # Get sources
        sources = []
        if research_data:
            sources = [r.url for r in research_data.search_results[:10]]
        
        # Extract takeaways and recommendations from text
        takeaways = self._extract_bullet_points(text, ["takeaway", "key point", "finding"])
        recommendations = self._extract_bullet_points(text, ["recommend", "suggestion", "action"])
        
        word_count = len(text.split())
        
        return FinalReport(
            title=f"Research Report: {analysis.topic}",
            topic=analysis.topic,
            executive_summary=analysis.executive_summary,
            sections=sections,
            key_takeaways=takeaways[:5] if takeaways else [analysis.executive_summary[:100]],
            recommendations=recommendations[:5] if recommendations else [],
            sources=sources,
            generated_at=datetime.now(),
            word_count=word_count,
        )
    
    def _parse_markdown_sections(self, text: str) -> list[ReportSection]:
        """
        Parse markdown text into sections.
        
        Args:
            text: Markdown text
            
        Returns:
            List of ReportSection objects
        """
        sections = []
        current_title = None
        current_content = []
        
        for line in text.split("\n"):
            # Check for header
            if line.startswith("## "):
                # Save previous section
                if current_title:
                    sections.append(ReportSection(
                        title=current_title,
                        content="\n".join(current_content).strip(),
                    ))
                current_title = line[3:].strip()
                current_content = []
            elif line.startswith("# "):
                # Skip main title
                continue
            else:
                current_content.append(line)
        
        # Save last section
        if current_title:
            sections.append(ReportSection(
                title=current_title,
                content="\n".join(current_content).strip(),
            ))
        
        return sections
    
    def _extract_bullet_points(self, text: str, keywords: list[str]) -> list[str]:
        """
        Extract bullet points near keywords.
        
        Args:
            text: Text to search
            keywords: Keywords to look for
            
        Returns:
            List of extracted bullet points
        """
        points = []
        lines = text.split("\n")
        
        in_section = False
        for line in lines:
            line_lower = line.lower()
            
            # Check if entering relevant section
            if any(kw in line_lower for kw in keywords):
                in_section = True
                continue
            
            # Check if leaving section (new header)
            if line.startswith("#"):
                in_section = False
                continue
            
            # Extract bullet points
            if in_section and (line.strip().startswith("-") or line.strip().startswith("•")):
                point = line.strip().lstrip("-•").strip()
                if point and len(point) > 10:
                    points.append(point)
        
        return points
    
    def _save_report(self, report: FinalReport):
        """
        Save report to file.
        
        Args:
            report: Report to save
        """
        if not self.output_dir:
            return
        
        try:
            # Create filename
            safe_topic = re.sub(r'[^\w\s-]', '', report.topic)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_topic}_{timestamp}.md"
            
            filepath = self.output_dir / filename
            
            # Write markdown
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            
            logger.info(f"Report saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
    
    async def aprocess(self, state: GraphState) -> GraphState:
        """
        Async version of process.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with final_report
        """
        self.log_action("Starting async report generation")
        
        try:
            analysis_summary = state.get("analysis_summary")
            if not analysis_summary:
                raise ValueError("No analysis summary available")
            
            research_data = state.get("research_data")
            critique_result = state.get("critique_result")
            
            final_report = await self._async_generate_report(
                analysis_summary, research_data, critique_result
            )
            
            if self.output_dir:
                self._save_report(final_report)
            
            message = self.create_message(
                receiver="supervisor",
                message_type="completion",
                content=f"Report generated: '{final_report.title}'",
                metadata={"word_count": final_report.word_count}
            )
            
            self.log_action("Async report generation completed")
            
            return {
                **state,
                "final_report": final_report,
                "current_agent": "writer",
                "workflow_status": "completed",
                "completed_at": datetime.now(),
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Async report generation failed")
    
    async def _async_generate_report(
        self,
        analysis_summary: AnalysisSummary,
        research_data: Optional[ResearchData],
        critique_result: Optional[CritiqueResult],
    ) -> FinalReport:
        """
        Async report generation.
        
        Args:
            analysis_summary: Analysis to convert
            research_data: Research data
            critique_result: Critique feedback
            
        Returns:
            FinalReport
        """
        task_prompt = self._format_task_prompt(
            analysis_summary, research_data, critique_result
        )
        
        llm_response = await self.ainvoke_llm(task_prompt)
        
        return self._parse_report_response(
            llm_response, analysis_summary, research_data
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_writer_agent(
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
    output_dir: Optional[Path] = None,
) -> WriterAgent:
    """
    Factory function to create a configured WriterAgent.
    
    Args:
        groq_api_key: Groq API key
        model_name: LLM model name
        temperature: LLM temperature
        output_dir: Directory to save reports
        
    Returns:
        Configured WriterAgent instance
    """
    return WriterAgent(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        output_dir=output_dir,
    )


__all__ = [
    "WriterAgent",
    "create_writer_agent",
]
