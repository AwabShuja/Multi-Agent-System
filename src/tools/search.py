"""
Tavily Search Tool for the Multi-Agent Virtual Company.

This module provides web search capabilities using the Tavily API,
optimized for AI agent research tasks.
"""

import asyncio
from typing import Optional
from datetime import datetime
from loguru import logger

from tavily import TavilyClient, AsyncTavilyClient

from src.schemas.models import SearchResult, ResearchData


class TavilySearchTool:
    """
    Tavily-powered search tool for gathering research data.
    
    Tavily is specifically designed for AI agents and provides
    high-quality, relevant search results with content extraction.
    """
    
    def __init__(self, api_key: str, max_results: int = 5):
        """
        Initialize the Tavily search tool.
        
        Args:
            api_key: Tavily API key
            max_results: Maximum number of results per search (default: 5)
        """
        self.api_key = api_key
        self.max_results = max_results
        self._sync_client: Optional[TavilyClient] = None
        self._async_client: Optional[AsyncTavilyClient] = None
        
        logger.info(f"TavilySearchTool initialized with max_results={max_results}")
    
    @property
    def sync_client(self) -> TavilyClient:
        """Lazy initialization of sync client."""
        if self._sync_client is None:
            self._sync_client = TavilyClient(api_key=self.api_key)
        return self._sync_client
    
    @property
    def async_client(self) -> AsyncTavilyClient:
        """Lazy initialization of async client."""
        if self._async_client is None:
            self._async_client = AsyncTavilyClient(api_key=self.api_key)
        return self._async_client
    
    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = False,
        max_results: Optional[int] = None,
    ) -> ResearchData:
        """
        Perform a synchronous search using Tavily.
        
        Args:
            query: Search query string
            search_depth: "basic" or "advanced" (more thorough)
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            include_answer: Whether to include AI-generated answer
            include_raw_content: Whether to include full page content
            max_results: Override default max results
            
        Returns:
            ResearchData with search results and metadata
        """
        logger.info(f"Searching for: '{query}' (depth: {search_depth})")
        
        try:
            results = self.sync_client.search(
                query=query,
                search_depth=search_depth,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                max_results=max_results or self.max_results,
            )
            
            return self._process_results(query, results)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ResearchData(
                topic=query,
                search_results=[],
                raw_content=f"Search failed: {str(e)}",
                sources_count=0,
                researcher_notes=f"Error during search: {str(e)}"
            )
    
    async def asearch(
        self,
        query: str,
        search_depth: str = "advanced",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = False,
        max_results: Optional[int] = None,
    ) -> ResearchData:
        """
        Perform an asynchronous search using Tavily.
        
        Args:
            query: Search query string
            search_depth: "basic" or "advanced" (more thorough)
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            include_answer: Whether to include AI-generated answer
            include_raw_content: Whether to include full page content
            max_results: Override default max results
            
        Returns:
            ResearchData with search results and metadata
        """
        logger.info(f"Async searching for: '{query}' (depth: {search_depth})")
        
        try:
            results = await self.async_client.search(
                query=query,
                search_depth=search_depth,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                max_results=max_results or self.max_results,
            )
            
            return self._process_results(query, results)
            
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return ResearchData(
                topic=query,
                search_results=[],
                raw_content=f"Search failed: {str(e)}",
                sources_count=0,
                researcher_notes=f"Error during search: {str(e)}"
            )
    
    def _process_results(self, query: str, raw_results: dict) -> ResearchData:
        """
        Process raw Tavily results into structured ResearchData.
        
        Args:
            query: Original search query
            raw_results: Raw response from Tavily API
            
        Returns:
            Structured ResearchData object
        """
        search_results = []
        raw_content_parts = []
        
        # Process individual results
        for result in raw_results.get("results", []):
            search_result = SearchResult(
                title=result.get("title", "Untitled"),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score"),
                published_date=result.get("published_date"),
            )
            search_results.append(search_result)
            
            # Build raw content
            raw_content_parts.append(f"## {search_result.title}")
            raw_content_parts.append(f"Source: {search_result.url}")
            raw_content_parts.append(f"{search_result.content}")
            raw_content_parts.append("")
        
        # Include Tavily's AI-generated answer if available
        tavily_answer = raw_results.get("answer", "")
        researcher_notes = None
        if tavily_answer:
            researcher_notes = f"Tavily AI Summary: {tavily_answer}"
        
        research_data = ResearchData(
            topic=query,
            search_results=search_results,
            raw_content="\n".join(raw_content_parts),
            sources_count=len(search_results),
            timestamp=datetime.now(),
            researcher_notes=researcher_notes,
        )
        
        logger.info(f"Search completed: {len(search_results)} results found")
        return research_data
    
    def search_news(
        self,
        query: str,
        days: int = 7,
        max_results: Optional[int] = None,
    ) -> ResearchData:
        """
        Search specifically for recent news articles.
        
        Args:
            query: Search query
            days: Number of days to look back (default: 7)
            max_results: Maximum results to return
            
        Returns:
            ResearchData with news-focused results
        """
        logger.info(f"Searching news for: '{query}' (last {days} days)")
        
        try:
            results = self.sync_client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                days=days,
                max_results=max_results or self.max_results,
                include_answer=True,
            )
            
            return self._process_results(query, results)
            
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return ResearchData(
                topic=query,
                search_results=[],
                raw_content=f"News search failed: {str(e)}",
                sources_count=0,
                researcher_notes=f"Error during news search: {str(e)}"
            )
    
    def search_finance(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> ResearchData:
        """
        Search with focus on financial/stock information.
        
        Includes trusted financial domains and excludes unreliable sources.
        
        Args:
            query: Search query (e.g., "Tesla stock analysis")
            max_results: Maximum results to return
            
        Returns:
            ResearchData with finance-focused results
        """
        # Trusted financial domains
        finance_domains = [
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "cnbc.com",
            "finance.yahoo.com",
            "marketwatch.com",
            "seekingalpha.com",
            "fool.com",
            "investopedia.com",
            "barrons.com",
        ]
        
        logger.info(f"Searching finance sources for: '{query}'")
        
        return self.search(
            query=query,
            search_depth="advanced",
            include_domains=finance_domains,
            include_answer=True,
            max_results=max_results,
        )
    
    def search_tech(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> ResearchData:
        """
        Search with focus on technology news and analysis.
        
        Args:
            query: Search query (e.g., "AI trends 2026")
            max_results: Maximum results to return
            
        Returns:
            ResearchData with tech-focused results
        """
        # Trusted tech domains
        tech_domains = [
            "techcrunch.com",
            "theverge.com",
            "wired.com",
            "arstechnica.com",
            "venturebeat.com",
            "thenextweb.com",
            "zdnet.com",
            "cnet.com",
            "engadget.com",
            "mit.edu",
        ]
        
        logger.info(f"Searching tech sources for: '{query}'")
        
        return self.search(
            query=query,
            search_depth="advanced",
            include_domains=tech_domains,
            include_answer=True,
            max_results=max_results,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_tavily_tool(api_key: str, max_results: int = 5) -> TavilySearchTool:
    """
    Factory function to create a configured TavilySearchTool.
    
    Args:
        api_key: Tavily API key
        max_results: Maximum search results
        
    Returns:
        Configured TavilySearchTool instance
    """
    return TavilySearchTool(api_key=api_key, max_results=max_results)


# =============================================================================
# LangChain Tool Integration (for agent use)
# =============================================================================

def get_tavily_langchain_tool(api_key: str):
    """
    Get a LangChain-compatible Tavily search tool for agent binding.
    
    This integrates directly with LangChain's tool system for use
    with LLM agents.
    
    Args:
        api_key: Tavily API key
        
    Returns:
        LangChain Tool instance
    """
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    return TavilySearchResults(
        api_key=api_key,
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        name="tavily_search",
        description=(
            "Search the web for current information on any topic. "
            "Use this to find recent news, stock information, tech trends, "
            "and other up-to-date information. Input should be a search query."
        ),
    )


__all__ = [
    "TavilySearchTool",
    "create_tavily_tool",
    "get_tavily_langchain_tool",
]
