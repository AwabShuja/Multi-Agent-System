"""
Web Scraper Utilities for the Multi-Agent Virtual Company.

This module provides additional web scraping capabilities
to supplement Tavily search when deeper content extraction is needed.
"""

import re
import asyncio
from typing import Optional
from urllib.parse import urlparse, urljoin
from datetime import datetime
from loguru import logger

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.schemas.models import SearchResult


class WebScraper:
    """
    Web scraper for extracting content from URLs.
    
    Used when Tavily results need to be supplemented with
    additional content from specific pages.
    """
    
    # Default headers to mimic a browser
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        headers: Optional[dict] = None,
    ):
        """
        Initialize the web scraper.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            headers: Custom headers (optional)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = headers or self.DEFAULT_HEADERS
        self._session: Optional[requests.Session] = None
        
        logger.info(f"WebScraper initialized (timeout={timeout}s, retries={max_retries})")
    
    @property
    def session(self) -> requests.Session:
        """Lazy initialization of requests session with retry logic."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        
        return self._session
    
    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string, or None if fetch fails
        """
        logger.debug(f"Fetching URL: {url}")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    async def afetch_url(self, url: str) -> Optional[str]:
        """
        Asynchronously fetch content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string, or None if fetch fails
        """
        logger.debug(f"Async fetching URL: {url}")
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                        
        except Exception as e:
            logger.warning(f"Async fetch failed for {url}: {e}")
            return None
    
    async def afetch_multiple(self, urls: list[str]) -> dict[str, Optional[str]]:
        """
        Asynchronously fetch content from multiple URLs.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            Dictionary mapping URL to content (or None if failed)
        """
        logger.info(f"Async fetching {len(urls)} URLs")
        
        tasks = [self.afetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        return dict(zip(urls, results))
    
    def extract_text(self, html: str) -> str:
        """
        Extract readable text from HTML content.
        
        This is a simple extraction that removes HTML tags.
        For production, consider using BeautifulSoup or similar.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Extracted text content
        """
        if not html:
            return ""
        
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode common HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_title(self, html: str) -> str:
        """
        Extract page title from HTML.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Page title or "Untitled"
        """
        if not html:
            return "Untitled"
        
        # Try to find title tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Try to find h1
        h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()
        
        return "Untitled"
    
    def extract_metadata(self, html: str) -> dict:
        """
        Extract metadata from HTML (description, keywords, etc.).
        
        Args:
            html: Raw HTML content
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "description": "",
            "keywords": "",
            "author": "",
            "published_date": "",
        }
        
        if not html:
            return metadata
        
        # Extract meta description
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if desc_match:
            metadata["description"] = desc_match.group(1)
        
        # Extract meta keywords
        keywords_match = re.search(
            r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if keywords_match:
            metadata["keywords"] = keywords_match.group(1)
        
        # Extract author
        author_match = re.search(
            r'<meta[^>]*name=["\']author["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if author_match:
            metadata["author"] = author_match.group(1)
        
        # Extract published date (common formats)
        date_patterns = [
            r'<meta[^>]*property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*name=["\']date["\'][^>]*content=["\']([^"\']+)["\']',
            r'<time[^>]*datetime=["\']([^"\']+)["\']',
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, html, re.IGNORECASE)
            if date_match:
                metadata["published_date"] = date_match.group(1)
                break
        
        return metadata
    
    def scrape_url(self, url: str) -> Optional[SearchResult]:
        """
        Scrape a URL and return structured SearchResult.
        
        Args:
            url: URL to scrape
            
        Returns:
            SearchResult with extracted content, or None if failed
        """
        html = self.fetch_url(url)
        if not html:
            return None
        
        title = self.extract_title(html)
        content = self.extract_text(html)
        metadata = self.extract_metadata(html)
        
        # Truncate content if too long
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        return SearchResult(
            title=title,
            url=url,
            content=content or metadata.get("description", "No content extracted"),
            published_date=metadata.get("published_date"),
        )
    
    async def ascrape_url(self, url: str) -> Optional[SearchResult]:
        """
        Asynchronously scrape a URL and return structured SearchResult.
        
        Args:
            url: URL to scrape
            
        Returns:
            SearchResult with extracted content, or None if failed
        """
        html = await self.afetch_url(url)
        if not html:
            return None
        
        title = self.extract_title(html)
        content = self.extract_text(html)
        metadata = self.extract_metadata(html)
        
        # Truncate content if too long
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        return SearchResult(
            title=title,
            url=url,
            content=content or metadata.get("description", "No content extracted"),
            published_date=metadata.get("published_date"),
        )
    
    async def ascrape_multiple(self, urls: list[str]) -> list[SearchResult]:
        """
        Asynchronously scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of SearchResult objects (failed URLs are excluded)
        """
        logger.info(f"Async scraping {len(urls)} URLs")
        
        tasks = [self.ascrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        return [r for r in results if r is not None]
    
    def close(self):
        """Close the session and clean up resources."""
        if self._session:
            self._session.close()
            self._session = None


# =============================================================================
# Utility Functions
# =============================================================================

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Normalize a URL (remove fragments, trailing slashes).
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
    """
    parsed = urlparse(url)
    # Remove fragment and normalize
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized.rstrip("/")


def get_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        Domain name
    """
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


# =============================================================================
# Factory Function
# =============================================================================

def create_scraper(
    timeout: int = 30,
    max_retries: int = 3,
) -> WebScraper:
    """
    Factory function to create a configured WebScraper.
    
    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        
    Returns:
        Configured WebScraper instance
    """
    return WebScraper(timeout=timeout, max_retries=max_retries)


__all__ = [
    "WebScraper",
    "create_scraper",
    "is_valid_url",
    "normalize_url",
    "get_domain",
]
