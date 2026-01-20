"""
Tools for search, scraping, and analysis in the Multi-Agent Virtual Company.

This package provides:
- TavilySearchTool: Web search using Tavily API
- WebScraper: Additional web scraping utilities
- TextAnalyzer: Text processing and analysis
- ResearchDataProcessor: Prepare research data for agents
"""

from .search import (
    TavilySearchTool,
    create_tavily_tool,
    get_tavily_langchain_tool,
)
from .scraper import (
    WebScraper,
    create_scraper,
    is_valid_url,
    normalize_url,
    get_domain,
)
from .analysis import (
    TextAnalyzer,
    ResearchDataProcessor,
    create_text_analyzer,
    create_data_processor,
)

__all__ = [
    # Search
    "TavilySearchTool",
    "create_tavily_tool",
    "get_tavily_langchain_tool",
    # Scraper
    "WebScraper",
    "create_scraper",
    "is_valid_url",
    "normalize_url",
    "get_domain",
    # Analysis
    "TextAnalyzer",
    "ResearchDataProcessor",
    "create_text_analyzer",
    "create_data_processor",
]
