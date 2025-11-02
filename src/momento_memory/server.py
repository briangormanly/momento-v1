"""
MCP server for Momento Enhanced Memory System.

This module provides the FastMCP server with all tools for:
- Entry management (create_entry, search_entries_semantic, get_entry_context)
- Project management (create_project, get_project_memories)
- Backward compatible entity/relation management
- Enhanced querying capabilities
"""

import json
import logging
from typing import List, Literal, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent, ToolAnnotations
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .embeddings import EmbeddingGenerator, EmbeddingMode
from .memory import MomentoMemory
from .models import (
    Entity,
    ObservationAddition,
    ObservationDeletion,
    Project,
    Relation,
)

logger = logging.getLogger('momento_memory')
logger.setLevel(logging.INFO)


def create_mcp_server(memory: MomentoMemory) -> FastMCP:
    """Create MCP server with all Momento tools."""
    
    mcp = FastMCP(
        "momento-memory",
        dependencies=["neo4j", "pydantic", "sentence-transformers"],
        stateless_http=True
    )
    
    # ========================================================================
    # Entry Management Tools
    # ========================================================================
    
    @mcp.tool(
        name="create_entry",
        annotations=ToolAnnotations(
            title="Create Entry",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        )
    )
    async def create_entry(
        content: str = Field(..., description="Original text content of the entry"),
        entry_type: str = Field(default="journal", description="Type: journal, code_comment, meeting, task, note"),
        project_id: Optional[str] = Field(None, description="ID of associated project"),
        extract_entities: bool = Field(True, description="Whether to extract entities automatically"),
        metadata: Optional[dict] = Field(None, description="Additional metadata")
    ) -> ToolResult:
        """Create a new entry with automatic entity extraction and embedding generation.
        
        This is the primary way to add information to Momento. It preserves your original
        text while extracting entities and relationships for the knowledge graph.
        
        Example:
        {
            "content": "Today I completed the authentication module using JWT tokens",
            "entry_type": "journal",
            "project_id": "momento-001",
            "extract_entities": true
        }
        """
        logger.info(f"MCP tool: create_entry (type={entry_type}, project={project_id})")
        
        try:
            result = await memory.create_entry(
                content=content,
                entry_type=entry_type,
                project_id=project_id,
                extract_entities=extract_entities,
                metadata=metadata
            )
            
            response_text = f"Entry created successfully (ID: {result.id})"
            if result.entities:
                response_text += f"\nExtracted {len(result.entities)} entities"
            if result.relations:
                response_text += f" and {len(result.relations)} relationships"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content=result.model_dump()
            )
            
        except Neo4jError as e:
            logger.error(f"Neo4j error creating entry: {e}")
            raise ToolError(f"Neo4j error creating entry: {e}")
        except Exception as e:
            logger.error(f"Error creating entry: {e}")
            raise ToolError(f"Error creating entry: {e}")
    
    @mcp.tool(
        name="search_entries_semantic",
        annotations=ToolAnnotations(
            title="Search Entries Semantically",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def search_entries_semantic(
        query: str = Field(..., description="Search query"),
        project_id: Optional[str] = Field(None, description="Filter by project ID"),
        limit: int = Field(10, ge=1, le=100, description="Maximum number of results"),
        similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    ) -> ToolResult:
        """Search entries using semantic similarity.
        
        Finds entries that are semantically similar to your query, even if they don't
        contain the exact words. This uses embeddings to understand meaning.
        
        Example:
        {
            "query": "How did I implement security features?",
            "project_id": "momento-001",
            "limit": 10
        }
        """
        logger.info(f"MCP tool: search_entries_semantic ('{query}')")
        
        try:
            results = await memory.search_entries_semantic(
                query=query,
                project_id=project_id,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            response_text = f"Found {len(results)} entries matching '{query}'"
            if results:
                response_text += "\n\nTop results:"
                for i, entry in enumerate(results[:5], 1):
                    preview = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
                    response_text += f"\n{i}. [{entry.type}] {entry.timestamp.strftime('%Y-%m-%d')}: {preview}"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content={"entries": [e.model_dump() for e in results]}
            )
            
        except Neo4jError as e:
            logger.error(f"Neo4j error searching entries: {e}")
            raise ToolError(f"Neo4j error searching entries: {e}")
        except Exception as e:
            logger.error(f"Error searching entries: {e}")
            raise ToolError(f"Error searching entries: {e}")
    
    @mcp.tool(
        name="get_entry_context",
        annotations=ToolAnnotations(
            title="Get Entry Context",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def get_entry_context(
        entity_name: str = Field(..., description="Name of entity to get context for")
    ) -> ToolResult:
        """Get all entries that mention a specific entity.
        
        This traces back from an extracted entity to all the original entries
        that mentioned it, giving you the full context.
        
        Example:
        {
            "entity_name": "JWT tokens"
        }
        """
        logger.info(f"MCP tool: get_entry_context ('{entity_name}')")
        
        try:
            entries = await memory.get_entry_context(entity_name)
            
            response_text = f"Found {len(entries)} entries mentioning '{entity_name}'"
            if entries:
                response_text += "\n\nEntries:"
                for i, entry in enumerate(entries, 1):
                    preview = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
                    response_text += f"\n{i}. [{entry.type}] {entry.timestamp.strftime('%Y-%m-%d')}: {preview}"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content={"entries": [e.model_dump() for e in entries]}
            )
            
        except Neo4jError as e:
            logger.error(f"Neo4j error getting entry context: {e}")
            raise ToolError(f"Neo4j error getting entry context: {e}")
        except Exception as e:
            logger.error(f"Error getting entry context: {e}")
            raise ToolError(f"Error getting entry context: {e}")
    
    # ========================================================================
    # Project Management Tools
    # ========================================================================
    
    @mcp.tool(
        name="create_project",
        annotations=ToolAnnotations(
            title="Create Project",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def create_project(
        project_id: str = Field(..., description="Unique project identifier"),
        name: str = Field(..., description="Human-readable project name"),
        description: str = Field("", description="Project description"),
        status: str = Field("active", description="Status: active, archived, completed"),
        metadata: Optional[dict] = Field(None, description="Additional metadata")
    ) -> ToolResult:
        """Create a new project context.
        
        Projects help organize memories by context. Entries and entities can be
        associated with projects for better organization and retrieval.
        
        Example:
        {
            "project_id": "momento-001",
            "name": "Momento Enhanced Memory",
            "description": "Building an enhanced memory system",
            "status": "active"
        }
        """
        logger.info(f"MCP tool: create_project ({project_id})")
        
        try:
            project = await memory.create_project(
                project_id=project_id,
                name=name,
                description=description,
                status=status,
                metadata=metadata
            )
            
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Project created: {name} (ID: {project_id})"
                )],
                structured_content=project.model_dump()
            )
            
        except Neo4jError as e:
            logger.error(f"Neo4j error creating project: {e}")
            raise ToolError(f"Neo4j error creating project: {e}")
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise ToolError(f"Error creating project: {e}")
    
    @mcp.tool(
        name="get_project_memories",
        annotations=ToolAnnotations(
            title="Get Project Memories",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def get_project_memories(
        project_id: str = Field(..., description="Project ID"),
        include_cross_project: bool = Field(True, description="Include relevant memories from other projects"),
        relevance_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum relevance score")
    ) -> ToolResult:
        """Get all memories associated with a project.
        
        Returns entries, entities, and relationships for a specific project.
        Optionally includes relevant information from other projects.
        
        Example:
        {
            "project_id": "momento-001",
            "include_cross_project": true
        }
        """
        logger.info(f"MCP tool: get_project_memories ({project_id})")
        
        try:
            result = await memory.get_project_memories(
                project_id=project_id,
                include_cross_project=include_cross_project,
                relevance_threshold=relevance_threshold
            )
            
            response_text = f"Project: {result.project.name}\n"
            response_text += f"- {len(result.entries)} entries\n"
            response_text += f"- {len(result.entities)} entities\n"
            response_text += f"- {len(result.relations)} relationships"
            
            if include_cross_project and result.cross_project_entities:
                response_text += f"\n- {len(result.cross_project_entities)} cross-project entities"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content=result.model_dump()
            )
            
        except ValueError as e:
            logger.error(f"Project not found: {e}")
            raise ToolError(str(e))
        except Neo4jError as e:
            logger.error(f"Neo4j error getting project memories: {e}")
            raise ToolError(f"Neo4j error getting project memories: {e}")
        except Exception as e:
            logger.error(f"Error getting project memories: {e}")
            raise ToolError(f"Error getting project memories: {e}")
    
    # ========================================================================
    # Backward Compatible Tools (from base mcp-neo4j-memory)
    # ========================================================================
    
    @mcp.tool(
        name="create_entities",
        annotations=ToolAnnotations(
            title="Create Entities",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def create_entities(
        entities: List[Entity] = Field(..., description="List of entities to create")
    ) -> ToolResult:
        """Create multiple entities in the knowledge graph (backward compatible)."""
        logger.info(f"MCP tool: create_entities ({len(entities)})")
        
        try:
            result = await memory.create_entities(entities)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Created {len(result)} entities"
                )],
                structured_content={"entities": [e.model_dump() for e in result]}
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error creating entities: {e}")
            raise ToolError(f"Neo4j error creating entities: {e}")
        except Exception as e:
            logger.error(f"Error creating entities: {e}")
            raise ToolError(f"Error creating entities: {e}")
    
    @mcp.tool(
        name="create_relations",
        annotations=ToolAnnotations(
            title="Create Relations",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def create_relations(
        relations: List[Relation] = Field(..., description="List of relations to create")
    ) -> ToolResult:
        """Create multiple relationships between entities (backward compatible)."""
        logger.info(f"MCP tool: create_relations ({len(relations)})")
        
        try:
            result = await memory.create_relations(relations)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Created {len(result)} relationships"
                )],
                structured_content={"relations": [r.model_dump() for r in result]}
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error creating relations: {e}")
            raise ToolError(f"Neo4j error creating relations: {e}")
        except Exception as e:
            logger.error(f"Error creating relations: {e}")
            raise ToolError(f"Error creating relations: {e}")
    
    @mcp.tool(
        name="read_graph",
        annotations=ToolAnnotations(
            title="Read Graph",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def read_graph() -> ToolResult:
        """Read the entire knowledge graph (backward compatible)."""
        logger.info("MCP tool: read_graph")
        
        try:
            result = await memory.read_graph()
            response_text = f"Knowledge graph contains:\n"
            response_text += f"- {len(result.entities)} entities\n"
            response_text += f"- {len(result.relations)} relationships"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content=result.model_dump()
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error reading graph: {e}")
            raise ToolError(f"Neo4j error reading graph: {e}")
        except Exception as e:
            logger.error(f"Error reading graph: {e}")
            raise ToolError(f"Error reading graph: {e}")
    
    @mcp.tool(
        name="search_memories",
        annotations=ToolAnnotations(
            title="Search Memories",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def search_memories(
        query: str = Field(..., description="Fulltext search query")
    ) -> ToolResult:
        """Search memories using fulltext search (backward compatible)."""
        logger.info(f"MCP tool: search_memories ('{query}')")
        
        try:
            result = await memory.search_memories(query)
            response_text = f"Found {len(result.entities)} entities matching '{query}'"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content=result.model_dump()
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error searching memories: {e}")
            raise ToolError(f"Neo4j error searching memories: {e}")
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise ToolError(f"Error searching memories: {e}")
    
    @mcp.tool(
        name="find_memories_by_name",
        annotations=ToolAnnotations(
            title="Find Memories by Name",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def find_memories_by_name(
        names: List[str] = Field(..., description="Entity names to find")
    ) -> ToolResult:
        """Find specific entities by name (backward compatible)."""
        logger.info(f"MCP tool: find_memories_by_name ({len(names)} names)")
        
        try:
            result = await memory.find_memories_by_name(names)
            response_text = f"Found {len(result.entities)} entities"
            
            return ToolResult(
                content=[TextContent(type="text", text=response_text)],
                structured_content=result.model_dump()
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error finding memories: {e}")
            raise ToolError(f"Neo4j error finding memories: {e}")
        except Exception as e:
            logger.error(f"Error finding memories: {e}")
            raise ToolError(f"Error finding memories: {e}")
    
    @mcp.tool(
        name="add_observations",
        annotations=ToolAnnotations(
            title="Add Observations",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def add_observations(
        observations: List[ObservationAddition] = Field(..., description="Observations to add")
    ) -> ToolResult:
        """Add observations to entities (backward compatible)."""
        logger.info(f"MCP tool: add_observations ({len(observations)})")
        
        try:
            result = await memory.add_observations(observations)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Added observations to {len(result)} entities"
                )],
                structured_content={"result": result}
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error adding observations: {e}")
            raise ToolError(f"Neo4j error adding observations: {e}")
        except Exception as e:
            logger.error(f"Error adding observations: {e}")
            raise ToolError(f"Error adding observations: {e}")
    
    @mcp.tool(
        name="delete_entities",
        annotations=ToolAnnotations(
            title="Delete Entities",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def delete_entities(
        entityNames: List[str] = Field(..., description="Entity names to delete")
    ) -> ToolResult:
        """Delete entities (backward compatible)."""
        logger.info(f"MCP tool: delete_entities ({len(entityNames)})")
        
        try:
            await memory.delete_entities(entityNames)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Deleted {len(entityNames)} entities"
                )]
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting entities: {e}")
            raise ToolError(f"Neo4j error deleting entities: {e}")
        except Exception as e:
            logger.error(f"Error deleting entities: {e}")
            raise ToolError(f"Error deleting entities: {e}")
    
    @mcp.tool(
        name="delete_observations",
        annotations=ToolAnnotations(
            title="Delete Observations",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def delete_observations(
        deletions: List[ObservationDeletion] = Field(..., description="Observations to delete")
    ) -> ToolResult:
        """Delete observations (backward compatible)."""
        logger.info(f"MCP tool: delete_observations ({len(deletions)})")
        
        try:
            await memory.delete_observations(deletions)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Deleted observations from {len(deletions)} entities"
                )]
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting observations: {e}")
            raise ToolError(f"Neo4j error deleting observations: {e}")
        except Exception as e:
            logger.error(f"Error deleting observations: {e}")
            raise ToolError(f"Error deleting observations: {e}")
    
    @mcp.tool(
        name="delete_relations",
        annotations=ToolAnnotations(
            title="Delete Relations",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True
        )
    )
    async def delete_relations(
        relations: List[Relation] = Field(..., description="Relations to delete")
    ) -> ToolResult:
        """Delete relations (backward compatible)."""
        logger.info(f"MCP tool: delete_relations ({len(relations)})")
        
        try:
            await memory.delete_relations(relations)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Deleted {len(relations)} relationships"
                )]
            )
        except Neo4jError as e:
            logger.error(f"Neo4j error deleting relations: {e}")
            raise ToolError(f"Neo4j error deleting relations: {e}")
        except Exception as e:
            logger.error(f"Error deleting relations: {e}")
            raise ToolError(f"Error deleting relations: {e}")
    
    return mcp


async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    allow_origins: List[str] = [],
    allowed_hosts: List[str] = [],
    embedding_mode: str = "local",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_api_key: Optional[str] = None,
) -> None:
    """Main entry point for Momento Memory MCP server."""
    
    logger.info("Starting Momento Enhanced Memory Server")
    logger.info(f"Connecting to Neo4j: {neo4j_uri}")
    
    # Connect to Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password),
        database=neo4j_database
    )
    
    # Verify connection
    try:
        await neo4j_driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise
    
    # Create embedding generator
    try:
        mode = EmbeddingMode(embedding_mode)
    except ValueError:
        logger.warning(f"Invalid embedding mode '{embedding_mode}', using 'local'")
        mode = EmbeddingMode.LOCAL
    
    embedding_generator = EmbeddingGenerator(
        mode=mode,
        model_name=embedding_model,
        api_key=embedding_api_key
    )
    logger.info(f"Embedding generator initialized (mode={mode}, model={embedding_model})")
    
    # Create memory system
    memory = MomentoMemory(neo4j_driver, embedding_generator)
    await memory.initialize()
    logger.info("Memory system initialized")
    
    # Configure middleware
    custom_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        ),
        Middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts if allowed_hosts else ["localhost", "127.0.0.1"]
        )
    ]
    
    # Create MCP server
    mcp = create_mcp_server(memory)
    logger.info("MCP server created")
    
    # Run server
    logger.info(f"Starting server with transport: {transport}")
    match transport:
        case "http":
            logger.info(f"HTTP server starting on {host}:{port}{path}")
            await mcp.run_http_async(
                host=host,
                port=port,
                path=path,
                middleware=custom_middleware
            )
        case "stdio":
            logger.info("STDIO server starting")
            await mcp.run_stdio_async()
        case "sse":
            logger.info(f"SSE server starting on {host}:{port}{path}")
            await mcp.run_http_async(
                host=host,
                port=port,
                path=path,
                middleware=custom_middleware,
                transport="sse"
            )
        case _:
            raise ValueError(f"Unsupported transport: {transport}")

