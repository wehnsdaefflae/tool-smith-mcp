"""Configuration management for Tool Smith MCP."""

import logging
import os
from pathlib import Path
from typing import Optional

import toml
from pydantic import BaseModel, Field


class ClaudeConfig(BaseModel):
    """Claude API configuration."""

    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000
    temperature: float = 0.1
    structure_args_temperature: float = 0.0
    structure_args_max_tokens: int = 1000


class ToolsConfig(BaseModel):
    """Tools configuration."""

    similarity_threshold: float = 0.7
    search_top_k: int = 3
    tools_dir: str = "./tool-smith-mcp/tools"
    initial_tools_dir: str = "resources/initial_tools"


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    db_path: str = "./tool-smith-mcp/vector_db"
    collection_name: str = "tool_descriptions"
    model_name: str = "all-MiniLM-L6-v2"
    anonymized_telemetry: bool = False


class ServerConfig(BaseModel):
    """Server configuration."""

    name: str = "tool-smith-mcp"
    version: str = "0.1.0"
    description: str = (
        "MCP server that dynamically creates and manages tools using LLM assistance"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class DevelopmentConfig(BaseModel):
    """Development configuration."""

    auto_reload: bool = False
    debug_mode: bool = False


class DockerConfig(BaseModel):
    """Docker execution configuration."""

    image_name: str = "python:3.11-slim"
    container_timeout: int = 30
    memory_limit: str = "256m"
    cpu_limit: float = 0.5
    enabled: bool = True


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    cache_dir: str = "./tool-smith-mcp/cache"
    embedding_ttl: int = 3600  # 1 hour
    tool_code_ttl: int = 1800  # 30 minutes
    cleanup_interval: int = 3600  # 1 hour


class Config(BaseModel):
    """Main configuration class."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from TOML file.

    Args:
        config_path: Path to configuration file. If None, looks for tool-smith.toml
                    in current directory or project root.

    Returns:
        Loaded configuration
    """
    if config_path is None:
        # Look for config file in current directory first, then project root
        candidates = [
            Path("tool-smith.toml"),
            Path("./tool-smith.toml"),
            Path(__file__).parent.parent.parent.parent / "tool-smith.toml",
        ]

        config_path = None
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        logging.warning("No configuration file found, using defaults")
        return Config()

    try:
        with open(config_path, "r") as f:
            config_data = toml.load(f)

        return Config(**config_data)

    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        logging.info("Using default configuration")
        return Config()


def get_initial_tools_dir(config: Config) -> Path:
    """Get the path to initial tools directory.

    Args:
        config: Configuration object

    Returns:
        Path to initial tools directory
    """
    initial_tools_dir = config.tools.initial_tools_dir

    # If it's a relative path, resolve it relative to the package
    if not Path(initial_tools_dir).is_absolute():
        package_root = Path(__file__).parent.parent.parent.parent
        return package_root / initial_tools_dir

    return Path(initial_tools_dir)


def setup_logging(config: Config) -> None:
    """Set up logging based on configuration.

    Args:
        config: Configuration object
    """
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=config.logging.format, force=True)


def get_claude_api_key() -> str:
    """Get Claude API key from environment.

    Returns:
        Claude API key

    Raises:
        ValueError: If API key is not found
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is required")
    return api_key
