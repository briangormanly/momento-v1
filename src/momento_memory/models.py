"""
Data models for Momento Enhanced Memory System.

This module defines all Pydantic models used throughout the system:
- Entry: Original text with embeddings and metadata
- Project: Project context for organizing memories
- Enhanced Memory, Relation, etc. (backward compatible with base system)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Core Models (Backward Compatible with mcp-neo4j-memory)
# ============================================================================

class Entity(BaseModel):
    """Represents a memory entity in the knowledge graph.
    
    Backward compatible with base mcp-neo4j-memory system.
    
    Example:
    {
        "name": "John Smith",
        "type": "person", 
        "observations": ["Works at Neo4j", "Lives in San Francisco"]
    }
    """
    name: str = Field(
        description="Unique identifier/name for the entity",
        min_length=1,
        examples=["John Smith", "Neo4j Inc", "Authentication Module"]
    )
    type: str = Field(
        description="Category: 'person', 'company', 'location', 'concept', 'component', 'technology', etc.",
        min_length=1,
        examples=["person", "company", "location", "component", "technology"]
    )
    observations: List[str] = Field(
        description="List of facts or observations about this entity",
        default_factory=list
    )


class Relation(BaseModel):
    """Represents a relationship between two entities.
    
    Backward compatible with base mcp-neo4j-memory system.
    
    Example:
    {
        "source": "John Smith",
        "target": "Neo4j Inc", 
        "relationType": "WORKS_AT"
    }
    """
    source: str = Field(
        description="Name of the source entity",
        min_length=1
    )
    target: str = Field(
        description="Name of the target entity",
        min_length=1
    )
    relationType: str = Field(
        description="Type of relationship (uppercase with underscores)",
        min_length=1,
        examples=["WORKS_AT", "USES", "IMPLEMENTED", "RELATES_TO"]
    )


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph with entities and relationships."""
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


class ObservationAddition(BaseModel):
    """Request to add observations to an existing entity."""
    entityName: str = Field(min_length=1)
    observations: List[str] = Field(min_length=1)


class ObservationDeletion(BaseModel):
    """Request to delete observations from an existing entity."""
    entityName: str = Field(min_length=1)
    observations: List[str] = Field(min_length=1)


# ============================================================================
# Enhanced Models (New for Momento)
# ============================================================================

class Entry(BaseModel):
    """Represents an original entry (journal, code comment, meeting notes, etc.)
    
    New for Momento: Preserves original text and enables semantic search.
    
    Example:
    {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "content": "Today I completed the authentication module...",
        "embedding": [0.123, 0.456, ...],
        "timestamp": "2025-10-31T14:30:00Z",
        "type": "journal",
        "project_id": "abc-123",
        "metadata": {"mood": "productive"}
    }
    """
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this entry"
    )
    content: str = Field(
        description="Original text content of the entry",
        min_length=1
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Semantic embedding vector for similarity search"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this entry was created"
    )
    type: str = Field(
        default="journal",
        description="Type of entry: journal, code_comment, meeting, task, note",
        examples=["journal", "code_comment", "meeting", "task", "note"]
    )
    project_id: Optional[str] = Field(
        default=None,
        description="ID of associated project, if any"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tags, mood, etc.)"
    )


class Project(BaseModel):
    """Represents a project context for organizing memories.
    
    New for Momento: Enables project-scoped memory access.
    
    Example:
    {
        "id": "momento-001",
        "name": "Momento Enhanced Memory",
        "description": "Building an enhanced memory system",
        "created": "2025-10-31T00:00:00Z",
        "status": "active"
    }
    """
    id: str = Field(
        description="Unique identifier for this project",
        min_length=1
    )
    name: str = Field(
        description="Human-readable project name",
        min_length=1
    )
    description: str = Field(
        default="",
        description="Project description"
    )
    created: datetime = Field(
        default_factory=datetime.now,
        description="When this project was created"
    )
    status: str = Field(
        default="active",
        description="Project status: active, archived, completed",
        examples=["active", "archived", "completed"]
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional project metadata"
    )


class EnhancedMemory(Entity):
    """Enhanced memory entity with temporal and relevance tracking.
    
    Extends base Entity with additional fields for Momento.
    """
    first_seen: Optional[datetime] = Field(
        default=None,
        description="When this entity was first mentioned"
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="When this entity was last updated"
    )
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Project ID -> relevance score mapping"
    )


class EntryWithContext(Entry):
    """Entry with its extracted entities and relationships."""
    entities: List[Entity] = Field(
        default_factory=list,
        description="Entities extracted from this entry"
    )
    relations: List[Relation] = Field(
        default_factory=list,
        description="Relationships extracted from this entry"
    )


class ProjectMemories(BaseModel):
    """Memories associated with a project."""
    project: Project
    entries: List[Entry] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    cross_project_entities: List[Entity] = Field(
        default_factory=list,
        description="Relevant entities from other projects"
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateEntryRequest(BaseModel):
    """Request to create a new entry."""
    content: str = Field(min_length=1)
    type: str = Field(default="journal")
    project_id: Optional[str] = None
    extract_entities: bool = Field(
        default=True,
        description="Whether to automatically extract entities"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(min_length=1)
    project_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class TemporalQueryRequest(BaseModel):
    """Request for temporal query."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    project_id: Optional[str] = None
    entity_types: Optional[List[str]] = None


class GraphTraversalRequest(BaseModel):
    """Request for graph traversal."""
    start_entity: str = Field(min_length=1)
    max_depth: int = Field(default=3, ge=1, le=10)
    relationship_types: Optional[List[str]] = None
    project_filter: Optional[str] = None

