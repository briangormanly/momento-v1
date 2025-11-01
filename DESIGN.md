# Momento Enhanced Memory System - Design Document

## Overview

Momento Enhanced Memory is an enhanced memory system for AI-assisted software engineering, built on Neo4j's mcp-neo4j-memory server. It adds entry preservation, semantic search, and project-based context management while maintaining full backward compatibility with the base system.

## Base System Analysis

### Original mcp-neo4j-memory Schema

```cypher
(:Memory {
  name: string,
  type: string,
  observations: [string]
})
```

### Strengths

- Simple, flexible entity-relationship model
- Full-text search via Neo4j index
- Automatic entity extraction creates graph structure
- Good for storing facts about entities
- Relationships capture connections between entities

### Limitations Addressed

1. **Lost Source Context**: Original text/entry not preserved
   - Can't trace back from extracted entities to source
   - No semantic search on original content
   - Can't re-analyze entries with new extraction logic

2. **No Project Scoping**: All memories in one flat namespace
   - Can't filter by project context
   - No way to prioritize project-relevant memories
   - Cross-project insights require manual filtering

3. **No Temporal Context**: Limited time-based organization
   - Observations have timestamps in text, not as structured data
   - Can't query "what happened during this time period"
   - No chronological narrative reconstruction

4. **No Embedding Support**: Only keyword-based search
   - Can't find semantically similar content
   - Limited to exact or fuzzy text matches
   - No vector similarity search

## Enhanced Schema

### Core Nodes

#### Entry Node

```cypher
(:Entry {
  id: uuid,
  content: string,              // Original text
  embedding: vector,            // Semantic embedding
  timestamp: datetime,
  type: string,                 // 'journal', 'code_comment', 'meeting_notes', etc.
  project_id: string,           // Project association
  metadata: map                 // Flexible additional data
})
```

**Purpose**: 
- Preserve original context
- Enable semantic search
- Support re-extraction with improved logic
- Chronological organization

#### Enhanced Memory Node

```cypher
(:Memory {
  name: string,
  type: string,
  observations: [string],
  first_seen: datetime,         // When first mentioned
  last_updated: datetime,       // Last modification
  relevance_scores: map         // Project-specific relevance
})
```

#### Project Node

```cypher
(:Project {
  id: uuid,
  name: string,
  description: string,
  created: datetime,
  status: string,               // 'active', 'archived', 'completed'
  metadata: map
})
```

### Relationships

#### Existing
- `(:Memory)-[:RELATES_TO]->(:Memory)` - Various typed relationships

#### New
- `(:Entry)-[:EXTRACTED_ENTITY]->(:Memory)` - Links entries to extracted entities
- `(:Entry)-[:BELONGS_TO]->(:Project)` - Project association
- `(:Memory)-[:RELEVANT_TO {score: float}]->(:Project)` - Entity relevance to projects
- `(:Entry)-[:FOLLOWED_BY]->(:Entry)` - Chronological chain
- `(:Memory)-[:MENTIONED_IN]->(:Entry)` - Which entries mention this entity

## Architecture

### Data Flow

```
User Input → Entry Creation
  ├─ Store original text
  ├─ Generate embedding
  ├─ Extract entities (LLM-based)
  ├─ Create Memory nodes
  ├─ Link Entry ↔ Memory
  ├─ Associate with Project
  └─ Update timestamps and relevance
```

### Component Structure

```
momento_memory/
├── models.py         # Pydantic models (Entry, Project, Memory)
├── embeddings.py     # Embedding generation (local/API)
├── memory.py         # Core MomentoMemory class
├── server.py         # MCP server with all tools
└── utils.py          # Utility functions
```

## Features

### 1. Entry Preservation

Every journal entry is stored with:
- Full original text
- Semantic embedding for similarity search
- Timestamp for temporal queries
- Link to extracted entities

### 2. Semantic Search

Find information by meaning, not just keywords:
- Vector similarity search using embeddings
- Falls back to text search if vector index unavailable
- Supports project-scoped searches

### 3. Project Context

Organize work by project:
- Projects group related entries and entities
- Cross-project relevance scoring
- Project-scoped retrieval with optional cross-project insights

### 4. Temporal Tracking

- All entries timestamped
- Entities track first_seen and last_updated
- Enable chronological queries and time-based analysis

### 5. Source Tracing

Bidirectional links between entries and entities:
- Entry → Extracted Entity
- Entity → Mentioned In Entry
- Full context available for any entity

## MCP Tools

### Entry Management

#### `create_entry`
Create a new entry with automatic entity extraction and embedding generation.

**Input:**
```json
{
  "content": "string",
  "type": "journal|code_comment|meeting|task",
  "project_id": "optional_uuid",
  "extract_entities": true,
  "metadata": {}
}
```

**Process:**
1. Store original entry
2. Generate embedding (using configured model)
3. Extract entities using LLM (when implemented)
4. Create Entry node and relationships
5. Link to project if specified
6. Update Memory nodes with timestamps

#### `search_entries_semantic`
Search entries by semantic meaning.

**Input:**
```json
{
  "query": "string",
  "project_id": "optional",
  "limit": 10,
  "similarity_threshold": 0.7
}
```

**Returns:** Entries ranked by semantic similarity

#### `get_entry_context`
Retrieve full context for a memory/entity.

**Input:**
```json
{
  "entity_name": "string"
}
```

**Returns:** All entries that mention this entity

### Project Management

#### `create_project`
Create a new project context.

**Input:**
```json
{
  "name": "string",
  "description": "string",
  "metadata": {}
}
```

#### `get_project_memories`
Retrieve all memories relevant to a project.

**Input:**
```json
{
  "project_id": "uuid",
  "include_cross_project": true,
  "relevance_threshold": 0.5
}
```

**Returns:** 
- Primary: Memories directly from this project
- Secondary: Relevant memories from other projects

### Backward Compatible Tools

All original mcp-neo4j-memory tools preserved:
- `create_entities`, `create_relations`, `add_observations`
- `read_graph`, `search_memories`, `find_memories_by_name`
- `delete_entities`, `delete_relations`, `delete_observations`

## Technical Decisions

### Embedding Generation

**Default**: sentence-transformers/all-MiniLM-L6-v2 (local)
- Pros: No API costs, privacy, fast, good quality
- Cons: Less accurate than large models
- Dimensions: 384

**Alternative**: OpenAI embeddings (text-embedding-3-small/large)
- Pros: Higher quality
- Cons: API costs, latency, requires API key
- Dimensions: 1536 (small) or 3072 (large)

**Mode Options**:
- `local`: Use sentence-transformers (default)
- `api`: Use OpenAI API
- `hybrid`: Local by default, API for important entries

### Vector Search in Neo4j

- Neo4j 5.11+ supports vector similarity search
- Vector index on `Entry.embedding` property
- Falls back to text search automatically if index unavailable
- Can combine with graph traversal for hybrid retrieval

### Project Relevance Scoring

**Automatic Calculation** (when implemented):
```
relevance_score = 
  0.4 * (mentions in project entries) +
  0.3 * (connections to project entities) +
  0.2 * (recent activity in project) +
  0.1 * (manual boost)
```

### Performance Optimizations

1. Index on `Entry.timestamp` for temporal queries
2. Index on `Entry.project_id` for project filtering
3. Vector index on `Entry.embedding` for semantic search
4. Compound indexes for common query patterns

## Configuration

Configuration can be provided in three ways, with the following precedence order (highest to lowest):
1. **Command-line arguments** - Highest priority, overrides everything
2. **Environment variables** - Second priority
3. **`.env` file** - Lowest priority, convenient for default configuration

The application automatically loads the `.env` file from the project root directory.

### Using `.env` File (Recommended)

The easiest way to manage configuration is using a `.env` file:

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   # Neo4j Connection
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=password
   NEO4J_DATABASE=neo4j

   # Embedding Configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_API_KEY=optional_openai_key
   EMBEDDING_MODE=local

   # Transport (for HTTP/SSE)
   NEO4J_TRANSPORT=stdio
   NEO4J_MCP_SERVER_HOST=127.0.0.1
   NEO4J_MCP_SERVER_PORT=8000
   NEO4J_MCP_SERVER_PATH=/mcp/
   ```

3. Run the server - it automatically loads the `.env` file:
   ```bash
   momento
   ```

The `.env` file is ignored by git (via `.gitignore`) to prevent committing sensitive credentials. The `.env.example` file contains all available options with placeholder values.

### Environment Variables

All configuration options can also be set as environment variables:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password
export NEO4J_DATABASE=neo4j
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export EMBEDDING_API_KEY=optional_openai_key
export EMBEDDING_MODE=local
export NEO4J_TRANSPORT=stdio
export NEO4J_MCP_SERVER_HOST=127.0.0.1
export NEO4J_MCP_SERVER_PORT=8000
export NEO4J_MCP_SERVER_PATH=/mcp/
```

### Command Line Arguments

All configuration options can also be specified via command-line arguments:

```bash
momento \
  --db-uri bolt://localhost:7687 \
  --username neo4j \
  --password password \
  --embedding-mode local \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --transport stdio
```

Command-line arguments take highest precedence and override `.env` file and environment variables.

## Implementation Status

### Completed ✅

- Entry node schema and storage
- Embedding generation (local + API options)
- Semantic search with vector index support
- Project nodes and association
- Backward compatible tool set
- MCP server integration (stdio, HTTP, SSE)
- Temporal tracking (timestamps)

### TODO

1. **Entity Extraction**: Currently placeholder - needs LLM-based extraction using Claude API
2. **Cross-Project Relevance**: Structure in place, scoring algorithm needs implementation
3. **Temporal Queries**: Data structure ready, query tool not implemented
4. **Graph Traversal**: Tool for exploring entity connections not implemented

## Migration Path

### Existing Data

Current Memory nodes can coexist with new Entry system:
1. Keep existing Memory nodes as-is
2. New entries create both Entry and Memory nodes
3. Gradually deprecate direct Memory creation
4. Optional: Create synthetic Entry nodes from existing observations

### Backward Compatibility

All existing tools continue to work:
- `create_entities`, `create_relations`, etc. unchanged
- New tools are additive
- Clients can adopt incrementally

## Benefits

### For AI Engineering

1. **Context Preservation**: Full original text for re-analysis
2. **Semantic Search**: Find relevant information by meaning
3. **Project Isolation**: Focus on relevant project context
4. **Cross-Project Insights**: Still access relevant info from other projects
5. **Temporal Understanding**: Track evolution of ideas and implementations
6. **Better GraphRAG**: Richer context for LLM augmentation

### For Users

1. **Natural Input**: Journal-style entries, not forced entity extraction
2. **Time Travel**: See what you were working on at any point
3. **Project Dashboard**: Comprehensive project memory view
4. **Serendipity**: Discover related work from other projects
5. **Audit Trail**: Full history of all inputs and extractions

## Example Data Flow

### User Input
```
"Today I completed the authentication module for the Momento project.
I used JWT tokens and implemented refresh token rotation for security."
```

### Processing
1. **Create Entry**
   - Store original text
   - Generate embedding
   - Link to "Momento Enhanced Memory Project"
   - Timestamp: 2025-10-31T14:30:00Z

2. **Extract Entities**
   - Entity: "JWT tokens" (type: technology)
   - Entity: "authentication module" (type: component)
   - Entity: "refresh token rotation" (type: security_pattern)

3. **Create Relationships**
   - `(:Entry)-[:EXTRACTED_ENTITY]->(:Memory {name: "JWT tokens"})`
   - `(:Entry)-[:BELONGS_TO]->(:Project {name: "Momento"})`
   - `(:Memory {name: "authentication module"})-[:USES]->(:Memory {name: "JWT tokens"})`

4. **Update Relevance**
   - Update relevance scores for all entities to Momento project

### Later Retrieval

**Query**: "How did I implement security in authentication?"

**Process**:
1. Generate query embedding
2. Find similar entries (semantic search)
3. Get entities from those entries
4. Rank by project relevance
5. Return context-rich results with source entries
