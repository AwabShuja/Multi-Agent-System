"""
Configuration settings for the Multi-Agent Virtual Company.

This module handles all configuration including API keys, model settings,
and application parameters using environment variables.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure outputs directory exists
OUTPUTS_DIR.mkdir(exist_ok=True)


@dataclass
class Settings:
    """Application settings and configuration."""
    
    # =============================================================================
    # API Keys
    # =============================================================================
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096"))
    )
    
    # =============================================================================
    # Application Settings
    # =============================================================================
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    api_timeout: int = field(
        default_factory=lambda: int(os.getenv("API_TIMEOUT", "60"))
    )
    max_critic_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_CRITIC_ITERATIONS", "3"))
    )
    
    # =============================================================================
    # Paths
    # =============================================================================
    project_root: Path = field(default=PROJECT_ROOT)
    outputs_dir: Path = field(default=OUTPUTS_DIR)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_api_keys()
        self._validate_model_settings()
    
    def _validate_api_keys(self):
        """Validate that required API keys are present."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file.\n"
                "Get your API key from: https://console.groq.com/keys"
            )
        
        if not self.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set. Please add it to your .env file.\n"
                "Get your API key from: https://app.tavily.com/"
            )
    
    def _validate_model_settings(self):
        """Validate model configuration settings."""
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, got {self.temperature}"
            )
        
        if self.max_tokens <= 0:
            raise ValueError(
                f"Max tokens must be positive, got {self.max_tokens}"
            )
        
        if self.max_critic_iterations <= 0:
            raise ValueError(
                f"Max critic iterations must be positive, got {self.max_critic_iterations}"
            )
    
    def get_groq_config(self) -> dict:
        """Get Groq LLM configuration as a dictionary."""
        return {
            "api_key": self.groq_api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.api_timeout,
        }
    
    def get_tavily_config(self) -> dict:
        """Get Tavily search configuration as a dictionary."""
        return {
            "api_key": self.tavily_api_key,
            "max_results": 5,
        }
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.log_level == "DEBUG"
    
    def __repr__(self) -> str:
        """String representation hiding sensitive information."""
        return (
            f"Settings(\n"
            f"  model_name='{self.model_name}',\n"
            f"  temperature={self.temperature},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  log_level='{self.log_level}',\n"
            f"  groq_api_key='***{self.groq_api_key[-4:] if self.groq_api_key else 'NOT_SET'}',\n"
            f"  tavily_api_key='***{self.tavily_api_key[-4:] if self.tavily_api_key else 'NOT_SET'}'\n"
            f")"
        )


# Global settings instance
try:
    settings = Settings()
except ValueError as e:
    # If API keys are not set, provide helpful error message
    print(f"\n⚠️  Configuration Error: {e}")
    print("\nPlease follow these steps:")
    print("1. Copy .env.example to .env")
    print("2. Add your API keys to the .env file")
    print("3. Run the application again\n")
    settings = None


# Export commonly used paths
__all__ = ["settings", "PROJECT_ROOT", "OUTPUTS_DIR"]
