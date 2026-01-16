"""
Pydantic models for structured data validation across the multi-agent system.

These models ensure type safety and data validation for:
- Research data from web searches
- Analysis summaries
- Critique feedback
- Final reports
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Search & Research Models
# =============================================================================

class SearchResult(BaseModel):
    """A single search result from Tavily or web scraping."""
    
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the source")
    content: str = Field(description="Snippet or content from the source")
    score: Optional[float] = Field(
        default=None, 
        description="Relevance score if available"
    )
    published_date: Optional[str] = Field(
        default=None, 
        description="Publication date if available"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Tesla Q4 2025 Earnings Report",
                "url": "https://example.com/tesla-earnings",
                "content": "Tesla reported record earnings...",
                "score": 0.95,
                "published_date": "2026-01-15"
            }
        }


class ResearchData(BaseModel):
    """Structured output from the Researcher agent."""
    
    topic: str = Field(description="The research topic/query")
    search_results: list[SearchResult] = Field(
        default_factory=list,
        description="List of search results gathered"
    )
    raw_content: str = Field(
        default="",
        description="Concatenated raw content from all sources"
    )
    sources_count: int = Field(
        default=0,
        description="Number of sources found"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the research was conducted"
    )
    researcher_notes: Optional[str] = Field(
        default=None,
        description="Additional notes from the researcher agent"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Tesla stock performance 2026",
                "search_results": [],
                "raw_content": "Tesla has shown significant growth...",
                "sources_count": 5,
                "researcher_notes": "Found multiple reliable financial sources"
            }
        }


# =============================================================================
# Analysis Models
# =============================================================================

class KeyInsight(BaseModel):
    """A key insight extracted during analysis."""
    
    insight: str = Field(description="The insight or finding")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this insight"
    )
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="URLs or sources supporting this insight"
    )


class AnalysisSummary(BaseModel):
    """Structured output from the Analyst agent."""
    
    topic: str = Field(description="The analyzed topic")
    executive_summary: str = Field(
        description="Brief executive summary (2-3 sentences)"
    )
    key_insights: list[KeyInsight] = Field(
        default_factory=list,
        description="List of key insights extracted"
    )
    trends_identified: list[str] = Field(
        default_factory=list,
        description="Trends identified in the data"
    )
    sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        description="Overall sentiment from the analysis"
    )
    data_quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Quality score of the source data (0-1)"
    )
    risks_identified: list[str] = Field(
        default_factory=list,
        description="Potential risks or concerns identified"
    )
    opportunities_identified: list[str] = Field(
        default_factory=list,
        description="Potential opportunities identified"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the analysis was completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Tesla stock performance",
                "executive_summary": "Tesla shows strong momentum with increasing deliveries.",
                "key_insights": [],
                "trends_identified": ["Increasing EV adoption", "Expanding market share"],
                "sentiment": "bullish",
                "data_quality_score": 0.85,
                "risks_identified": ["Competition increasing"],
                "opportunities_identified": ["New markets in Asia"]
            }
        }


# =============================================================================
# Critique Models
# =============================================================================

class CritiqueResult(BaseModel):
    """Structured output from the Critic agent."""
    
    is_approved: bool = Field(
        description="Whether the analysis passes quality review"
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall quality score (0-1)"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Strengths of the analysis"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Weaknesses or areas for improvement"
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="Important elements that are missing"
    )
    bias_detected: bool = Field(
        default=False,
        description="Whether potential bias was detected"
    )
    bias_details: Optional[str] = Field(
        default=None,
        description="Details about detected bias if any"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )
    revision_required: bool = Field(
        description="Whether the analyst should revise their work"
    )
    revision_instructions: Optional[str] = Field(
        default=None,
        description="Specific instructions for revision if required"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the critique was completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "is_approved": True,
                "quality_score": 0.88,
                "strengths": ["Comprehensive coverage", "Good source diversity"],
                "weaknesses": ["Could include more quantitative data"],
                "missing_elements": [],
                "bias_detected": False,
                "suggestions": ["Add more financial metrics"],
                "revision_required": False
            }
        }


# =============================================================================
# Final Report Models
# =============================================================================

class ReportSection(BaseModel):
    """A section of the final report."""
    
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown")


class FinalReport(BaseModel):
    """Structured output from the Writer agent - the final deliverable."""
    
    title: str = Field(description="Report title")
    topic: str = Field(description="The researched topic")
    executive_summary: str = Field(
        description="Executive summary for quick reading"
    )
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Report sections with content"
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Key takeaways (bullet points)"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of source URLs used"
    )
    disclaimer: str = Field(
        default="This report is for informational purposes only and should not be considered financial advice.",
        description="Legal disclaimer"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the report was generated"
    )
    word_count: int = Field(
        default=0,
        description="Total word count of the report"
    )

    def to_markdown(self) -> str:
        """Convert the report to a formatted markdown string."""
        md_parts = [
            f"# {self.title}",
            f"\n*Generated on: {self.generated_at.strftime('%Y-%m-%d %H:%M')}*\n",
            "---\n",
            "## Executive Summary\n",
            f"{self.executive_summary}\n",
        ]
        
        # Add sections
        for section in self.sections:
            md_parts.append(f"## {section.title}\n")
            md_parts.append(f"{section.content}\n")
        
        # Key takeaways
        if self.key_takeaways:
            md_parts.append("## Key Takeaways\n")
            for takeaway in self.key_takeaways:
                md_parts.append(f"- {takeaway}")
            md_parts.append("")
        
        # Recommendations
        if self.recommendations:
            md_parts.append("\n## Recommendations\n")
            for i, rec in enumerate(self.recommendations, 1):
                md_parts.append(f"{i}. {rec}")
            md_parts.append("")
        
        # Sources
        if self.sources:
            md_parts.append("\n## Sources\n")
            for source in self.sources:
                md_parts.append(f"- {source}")
            md_parts.append("")
        
        # Disclaimer
        md_parts.append(f"\n---\n*{self.disclaimer}*")
        
        return "\n".join(md_parts)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Tesla Stock Analysis Report - January 2026",
                "topic": "Tesla stock performance",
                "executive_summary": "A comprehensive analysis of Tesla's market position...",
                "sections": [],
                "key_takeaways": ["Strong growth trajectory", "Market leader in EVs"],
                "recommendations": ["Consider long-term holding"],
                "sources": ["https://example.com"],
                "word_count": 1500
            }
        }


# =============================================================================
# Agent Communication Models
# =============================================================================

class AgentMessage(BaseModel):
    """Message passed between agents in the workflow."""
    
    sender: Literal["supervisor", "researcher", "analyst", "critic", "writer"] = Field(
        description="Agent that sent this message"
    )
    receiver: Literal["supervisor", "researcher", "analyst", "critic", "writer", "all"] = Field(
        description="Intended recipient agent"
    )
    message_type: Literal["instruction", "data", "feedback", "completion", "error"] = Field(
        description="Type of message"
    )
    content: str = Field(
        description="Message content"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sender": "supervisor",
                "receiver": "researcher",
                "message_type": "instruction",
                "content": "Research the latest Tesla stock news",
                "metadata": {"priority": "high"}
            }
        }
