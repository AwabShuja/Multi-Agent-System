"""
Researcher Agent for the Multi-Agent Virtual Company.

The Researcher agent is responsible for gathering information from the web
using Tavily search API. It's the first agent in the research pipeline.
"""

from typing import Optional
from datetime import datetime
from loguru import logger

from src.agents.base import ToolEnabledAgent
from src.graph.state import GraphState
from src.schemas.models import ResearchData, SearchResult, AgentMessage
from src.tools.search import TavilySearchTool, create_tavily_tool
from src.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_TASK_PROMPT,
    RESEARCHER_SEARCH_PROMPT,
)


class ResearcherAgent(ToolEnabledAgent):
    """
    Researcher Agent - Gathers information from the web.
    
    This agent uses Tavily search to find relevant information
    about stocks, tech trends, and other topics. It compiles
    raw research data for the Analyst agent.
    """
    
    def __init__(
        self,
        api_key: str,
        tavily_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,  # Lower temperature for factual research
        max_tokens: int = 4096,
        max_search_results: int = 5,
    ):
        """
        Initialize the Researcher agent.
        
        Args:
            api_key: Groq API key for LLM
            tavily_api_key: Tavily API key for search
            model_name: LLM model to use
            temperature: LLM temperature (lower for factual output)
            max_tokens: Maximum tokens for LLM response
            max_search_results: Maximum results per search query
        """
        # Initialize Tavily search tool
        self.search_tool = create_tavily_tool(
            api_key=tavily_api_key,
            max_results=max_search_results,
        )
        self.tavily_api_key = tavily_api_key
        
        # Initialize base agent
        super().__init__(
            name="researcher",
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[],  # We'll use our custom search tool directly
        )
        
        logger.info("ResearcherAgent initialized with Tavily search")
    
    @property
    def system_prompt(self) -> str:
        """Return the researcher system prompt."""
        return RESEARCHER_SYSTEM_PROMPT
    
    def process(self, state: GraphState) -> GraphState:
        """
        Process the current state: perform research on the query.
        
        Args:
            state: Current graph state with user_query
            
        Returns:
            Updated state with research_data populated
        """
        self.log_action("Starting research", state.get("user_query", ""))
        
        try:
            # Get the research topic from state
            topic = state.get("user_query", "")
            if not topic:
                raise ValueError("No research topic provided in state")
            
            # Perform the research
            research_data = self._conduct_research(topic)
            
            # Create completion message
            message = self.create_message(
                receiver="supervisor",
                message_type="completion",
                content=f"Research completed. Found {research_data.sources_count} sources.",
                metadata={
                    "sources_count": research_data.sources_count,
                    "topic": topic,
                }
            )
            
            self.log_action(
                "Research completed",
                f"Found {research_data.sources_count} sources"
            )
            
            # Return updated state
            return {
                **state,
                "research_data": research_data,
                "current_agent": "researcher",
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Research failed")
    
    def _conduct_research(self, topic: str) -> ResearchData:
        """
        Conduct comprehensive research on the topic.
        
        Args:
            topic: Topic to research
            
        Returns:
            ResearchData with all findings
        """
        logger.info(f"Conducting research on: {topic}")
        
        # Generate search queries based on the topic
        search_queries = self._generate_search_queries(topic)
        
        # Collect all search results
        all_results: list[SearchResult] = []
        all_content_parts: list[str] = []
        seen_urls: set[str] = set()
        
        for query in search_queries:
            logger.debug(f"Searching: {query}")
            
            # Determine search type based on topic
            if self._is_finance_topic(topic):
                research_data = self.search_tool.search_finance(query)
            elif self._is_tech_topic(topic):
                research_data = self.search_tool.search_tech(query)
            else:
                research_data = self.search_tool.search(query)
            
            # Add unique results
            for result in research_data.search_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
                    all_content_parts.append(
                        f"## {result.title}\n"
                        f"Source: {result.url}\n"
                        f"{result.content}\n"
                    )
        
        # Generate researcher notes using LLM
        researcher_notes = self._generate_research_notes(topic, all_results)
        
        # Compile final research data
        return ResearchData(
            topic=topic,
            search_results=all_results,
            raw_content="\n---\n".join(all_content_parts),
            sources_count=len(all_results),
            timestamp=datetime.now(),
            researcher_notes=researcher_notes,
        )
    
    def _generate_search_queries(self, topic: str) -> list[str]:
        """
        Generate effective search queries for the topic.
        
        Args:
            topic: Research topic
            
        Returns:
            List of search queries
        """
        # Use LLM to generate targeted queries
        prompt = RESEARCHER_SEARCH_PROMPT.format(topic=topic)
        
        try:
            response = self.invoke_llm(prompt)
            
            # Parse queries from response (one per line)
            queries = [
                q.strip().strip("-").strip("•").strip("1234567890.").strip()
                for q in response.strip().split("\n")
                if q.strip() and len(q.strip()) > 5
            ]
            
            # Ensure we have at least the original topic as a query
            if not queries:
                queries = [topic]
            
            # Limit to 3 queries to avoid too many API calls
            return queries[:3]
            
        except Exception as e:
            logger.warning(f"Query generation failed, using topic directly: {e}")
            return [topic]
    
    def _generate_research_notes(
        self,
        topic: str,
        results: list[SearchResult],
    ) -> str:
        """
        Generate notes summarizing the research findings.
        
        Args:
            topic: Research topic
            results: Search results collected
            
        Returns:
            Research notes string
        """
        if not results:
            return "No relevant sources found for this topic."
        
        # Create a summary prompt
        sources_summary = "\n".join([
            f"- {r.title} ({r.url})"
            for r in results[:10]  # Limit to first 10 for prompt
        ])
        
        prompt = f"""Based on the following sources found for the topic "{topic}", 
provide brief research notes (2-3 sentences) about:
1. Overall data quality and source diversity
2. Any notable gaps or limitations
3. Key themes emerging from the sources

Sources found:
{sources_summary}

Research Notes:"""
        
        try:
            notes = self.invoke_llm(prompt)
            return notes.strip()
        except Exception as e:
            logger.warning(f"Could not generate research notes: {e}")
            return f"Found {len(results)} sources on topic: {topic}"
    
    def _is_finance_topic(self, topic: str) -> bool:
        """Check if topic is finance-related."""
        finance_keywords = [
            "stock", "share", "invest", "trading", "market", "earnings",
            "revenue", "profit", "dividend", "nasdaq", "nyse", "s&p",
            "dow", "crypto", "bitcoin", "ethereum", "forex", "bond",
            "etf", "fund", "portfolio", "financial", "banking",
        ]
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in finance_keywords)
    
    def _is_tech_topic(self, topic: str) -> bool:
        """Check if topic is technology-related."""
        tech_keywords = [
            "ai", "artificial intelligence", "machine learning", "tech",
            "software", "hardware", "startup", "innovation", "digital",
            "cloud", "saas", "app", "platform", "semiconductor", "chip",
            "data", "cyber", "robot", "automat", "computing", "internet",
        ]
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in tech_keywords)
    
    async def aprocess(self, state: GraphState) -> GraphState:
        """
        Async version of process for better performance.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with research_data
        """
        self.log_action("Starting async research", state.get("user_query", ""))
        
        try:
            topic = state.get("user_query", "")
            if not topic:
                raise ValueError("No research topic provided in state")
            
            # Use async search
            research_data = await self._async_conduct_research(topic)
            
            message = self.create_message(
                receiver="supervisor",
                message_type="completion",
                content=f"Research completed. Found {research_data.sources_count} sources.",
                metadata={
                    "sources_count": research_data.sources_count,
                    "topic": topic,
                }
            )
            
            self.log_action(
                "Async research completed",
                f"Found {research_data.sources_count} sources"
            )
            
            return {
                **state,
                "research_data": research_data,
                "current_agent": "researcher",
                "messages": [message],
            }
            
        except Exception as e:
            return self.handle_error(state, e, "Async research failed")
    
    async def _async_conduct_research(self, topic: str) -> ResearchData:
        """
        Async version of research conductor.
        
        Args:
            topic: Topic to research
            
        Returns:
            ResearchData with findings
        """
        logger.info(f"Conducting async research on: {topic}")
        
        search_queries = self._generate_search_queries(topic)
        
        all_results: list[SearchResult] = []
        all_content_parts: list[str] = []
        seen_urls: set[str] = set()
        
        for query in search_queries:
            logger.debug(f"Async searching: {query}")
            
            # Use async search
            research_data = await self.search_tool.asearch(query)
            
            for result in research_data.search_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
                    all_content_parts.append(
                        f"## {result.title}\n"
                        f"Source: {result.url}\n"
                        f"{result.content}\n"
                    )
        
        researcher_notes = self._generate_research_notes(topic, all_results)
        
        return ResearchData(
            topic=topic,
            search_results=all_results,
            raw_content="\n---\n".join(all_content_parts),
            sources_count=len(all_results),
            timestamp=datetime.now(),
            researcher_notes=researcher_notes,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_researcher_agent(
    groq_api_key: str,
    tavily_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_search_results: int = 5,
) -> ResearcherAgent:
    """
    Factory function to create a configured ResearcherAgent.
    
    Args:
        groq_api_key: Groq API key
        tavily_api_key: Tavily API key
        model_name: LLM model name
        temperature: LLM temperature
        max_search_results: Max results per search
        
    Returns:
        Configured ResearcherAgent instance
    """
    return ResearcherAgent(
        api_key=groq_api_key,
        tavily_api_key=tavily_api_key,
        model_name=model_name,
        temperature=temperature,
        max_search_results=max_search_results,
    )


__all__ = [
    "ResearcherAgent",
    "create_researcher_agent",
]
