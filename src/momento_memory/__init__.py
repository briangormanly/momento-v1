"""
Momento Enhanced Memory System

An enhanced memory system for AI-assisted software engineering with:
- Entry preservation and semantic search
- Project-based context management
- Backward compatibility with mcp-neo4j-memory
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .server import main as server_main

# Load .env file from the project root (momento-v1 directory)
# This allows configuration to be managed via .env file
# The .env file should be in the momento-v1 directory (one level up from src)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: try loading from current directory or parent directories
    load_dotenv()

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Momento Enhanced Memory MCP Server"
    )
    
    # Neo4j connection
    parser.add_argument(
        "--db-uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j database URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name (default: neo4j)"
    )
    
    # Transport configuration
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "http", "sse"],
        default=os.getenv("NEO4J_TRANSPORT", "stdio"),
        help="Transport mode (default: stdio)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("NEO4J_MCP_SERVER_HOST", "127.0.0.1"),
        help="Host for HTTP/SSE transport (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("NEO4J_MCP_SERVER_PORT", "8000")),
        help="Port for HTTP/SSE transport (default: 8000)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.getenv("NEO4J_MCP_SERVER_PATH", "/mcp/"),
        help="Path for HTTP/SSE transport (default: /mcp/)"
    )
    
    # Security
    parser.add_argument(
        "--allow-origins",
        type=str,
        default=os.getenv("NEO4J_MCP_SERVER_ALLOW_ORIGINS", ""),
        help="Comma-separated CORS allowed origins"
    )
    parser.add_argument(
        "--allowed-hosts",
        type=str,
        default=os.getenv("NEO4J_MCP_SERVER_ALLOWED_HOSTS", "localhost,127.0.0.1"),
        help="Comma-separated allowed hosts for DNS rebinding protection"
    )
    
    # Embedding configuration
    parser.add_argument(
        "--embedding-mode",
        type=str,
        choices=["local", "api", "hybrid"],
        default=os.getenv("EMBEDDING_MODE", "local"),
        help="Embedding generation mode (default: local)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Embedding model name"
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=os.getenv("EMBEDDING_API_KEY"),
        help="API key for embedding service (OpenAI)"
    )
    
    return parser.parse_args()


def parse_list_arg(arg: str) -> List[str]:
    """Parse comma-separated list argument."""
    if not arg:
        return []
    return [item.strip() for item in arg.split(",") if item.strip()]


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse list arguments
    allow_origins = parse_list_arg(args.allow_origins)
    allowed_hosts = parse_list_arg(args.allowed_hosts)
    
    # Run server
    try:
        asyncio.run(
            server_main(
                neo4j_uri=args.db_uri,
                neo4j_user=args.username,
                neo4j_password=args.password,
                neo4j_database=args.database,
                transport=args.transport,
                host=args.host,
                port=args.port,
                path=args.path,
                allow_origins=allow_origins,
                allowed_hosts=allowed_hosts,
                embedding_mode=args.embedding_mode,
                embedding_model=args.embedding_model,
                embedding_api_key=args.embedding_api_key,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

