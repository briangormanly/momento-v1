# Momento Enhanced Memory System

An enhanced memory system for AI-assisted software engineering, built on Neo4j's mcp-neo4j-memory server. It adds entry preservation, semantic search, and project-based context management while maintaining full backward compatibility with the base system.

## Overview

Momento Enhanced Memory extends Neo4j's mcp-neo4j-memory with advanced features for preserving context, enabling semantic search, and organizing memories by project. Users can write natural journal entries that are automatically processed to extract entities, generate embeddings, and link to projects—all while preserving the original text for full context tracing.

**Key Enhancements:**
- **Entry Preservation**: Original text stored with semantic embeddings for similarity search
- **Semantic Relationship Tracking**: Entry nodes are connected to entities using typed relationships that mirror the relationships between entities (e.g., if extracting `John WORKS_AT Google`, the entry gets `MENTIONS_WORKS_AT` relationships to both John and Google)
- **Project Context**: Organize and scope memories by project with cross-project relevance
- **Temporal Tracking**: Chronological organization with timestamp-based queries
- **Source Tracing**: Bidirectional links between entries and extracted entities

## Running as MCP Server

### Quick Start

```bash
# Start Neo4j database
cd infra
docker-compose up -d

# Install package
cd ..
pip install -e .   # or: uv sync

# Configure environment (optional but recommended)
cp .env.example .env
# Edit .env with your Neo4j connection details

# Run MCP server (uses .env file automatically)
momento   # or with uv: uv run momento
```

Alternatively, you can specify connection details via command line arguments:

```bash
momento --db-uri bolt://localhost:7687 \
  --username neo4j \
  --password neo4j_password
```

### Environment Configuration

The easiest way to manage configuration is using a `.env` file:

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your Neo4j connection details:
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=neo4j_password
   NEO4J_DATABASE=neo4j
   EMBEDDING_MODE=local
   ```

3. Run the server - it will automatically load the `.env` file:
   ```bash
   # If installed with pip install -e .
   momento
   
   # If installed with uv sync
   uv run momento
   ```

All configuration options can be set in `.env` or via command-line arguments. See `.env.example` for all available options.

**Note:** If you installed the package with `uv sync`, use `uv run momento` instead of just `momento`. The `uv run` command ensures the script runs in the correct virtual environment managed by uv.

### With Claude Desktop

When using `.env` file, Claude Desktop configuration is simple:

```json
{
  "mcpServers": {
    "momento": {
      "command": "momento"
    }
  }
}
```

**Note for uv users:** If you installed with `uv sync`, use `uv run` as the command:
```json
{
  "mcpServers": {
    "momento": {
      "command": "uv",
      "args": ["run", "momento"]
    }
  }
}
```

Or specify connection details directly:

```json
{
  "mcpServers": {
    "momento": {
      "command": "momento",
      "args": [
        "--db-uri", "bolt://localhost:7687",
        "--username", "neo4j",
        "--password", "neo4j_password"
      ]
    }
  }
}
```

### HTTP Mode

For web/API access:

```bash
momento --transport http \
  --host 127.0.0.1 \
  --port 8000 \
  --db-uri bolt://localhost:7687 \
  --username neo4j \
  --password neo4j_password
```

### Configuration Options

```bash
# Embedding mode (local, api, or hybrid)
--embedding-mode local

# Embedding model
--embedding-model sentence-transformers/all-MiniLM-L6-v2

# For OpenAI embeddings
--embedding-mode api
--embedding-api-key sk-your-key-here
```

## Development Environment

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Neo4j 5.11+ (for vector index support)

### Setup

```bash
# Clone repository
cd momento-v1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Start Neo4j
cd infra
docker-compose up -d
cd ..
```

### Environment Variables

Configuration can be provided in three ways (in order of precedence):
1. Command-line arguments (highest priority)
2. Environment variables
3. `.env` file (lowest priority)

**Using `.env` file (Recommended):**
Create a `.env` file from `.env.example` and set your values:
```bash
cp .env.example .env
# Edit .env with your configuration
```

**Using environment variables:**
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=neo4j_password
export NEO4J_DATABASE=neo4j
export EMBEDDING_MODE=local
```

The `.env` file is automatically loaded by the application and is the easiest way to manage configuration. See `.env.example` for all available options.

## Building

### Install Dependencies

```bash
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

### Run Tests

```bash
pytest
```

### Type Checking

```bash
pyright
```

### Run Examples

```bash
python examples/basic_usage.py
```

## Quick Example

```python
from momento_memory import MomentoMemory

memory = MomentoMemory(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="neo4j_password"
)

# Create entry
entry = await memory.create_entry(
    content="Today I completed the authentication module using JWT tokens.",
    author="Brian Gormanly",
    project_id="my-project"
)

# Extract entities and link them to the entry
# (In production, an LLM would extract these automatically)
entities = [
    Entity(name="authentication module", type="component", observations=["Uses JWT"]),
    Entity(name="JWT tokens", type="technology", observations=["Used for auth"])
]
await memory.create_entities(entities)

# Link the entities to the entry
await memory.link_entities_to_entry(
    entry_id=str(entry.id),
    entity_names=["authentication module", "JWT tokens"]
)

# Semantic search
results = await memory.search_entries_semantic(
    query="How did I implement authentication?",
    project_id="my-project"
)
```

### MCP Workflow with Claude

When using Momento via MCP with Claude:

1. **Create an entry** with `create_entry` tool (include `author` field)
2. Claude extracts entities and relationships from the content
3. **Create entities** with `create_entities` tool
4. **Create relationships** between entities with `create_relations` tool
5. **Link entities to entry** with `link_entities_to_entry` tool

This ensures Entry nodes are properly connected to Memory nodes with semantic relationships based on their roles in the story.

## Documentation

See [DESIGN.md](DESIGN.md) for detailed architecture, schema, and implementation details.

## License

MIT License - Building upon Neo4j's mcp-neo4j-memory (also MIT licensed).
