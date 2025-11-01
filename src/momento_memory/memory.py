"""
Core memory management for Momento Enhanced Memory System.

This module provides the MomentoMemory class which extends the base Neo4j memory
system with:
- Entry preservation and semantic search
- Project-based context management
- Temporal querying
- Enhanced entity extraction with source tracking
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from neo4j import AsyncDriver, RoutingControl

from .models import (
    Entry,
    Entity,
    EnhancedMemory,
    EntryWithContext,
    KnowledgeGraph,
    ObservationAddition,
    ObservationDeletion,
    Project,
    ProjectMemories,
    Relation,
)
from .embeddings import EmbeddingGenerator

logger = logging.getLogger('momento_memory')


class MomentoMemory:
    """Enhanced memory system with entry preservation and project context."""
    
    def __init__(
        self,
        neo4j_driver: AsyncDriver,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """Initialize Momento memory system.
        
        Args:
            neo4j_driver: Neo4j async driver
            embedding_generator: Optional embedding generator for semantic search
        """
        self.driver = neo4j_driver
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
    async def initialize(self):
        """Initialize database indexes and constraints."""
        logger.info("Initializing Momento memory database")
        
        # Create fulltext index for entities (backward compatible)
        await self._create_fulltext_index()
        
        # Create vector index for entries (semantic search)
        await self._create_vector_index()
        
        # Create constraints and indexes
        await self._create_constraints()
        
        logger.info("Database initialization complete")
    
    async def _create_fulltext_index(self):
        """Create fulltext index for Memory nodes."""
        try:
            query = """
            CREATE FULLTEXT INDEX search IF NOT EXISTS 
            FOR (m:Memory) 
            ON EACH [m.name, m.type, m.observations]
            """
            await self.driver.execute_query(query, routing_control=RoutingControl.WRITE)
            logger.info("Created fulltext index for Memory nodes")
        except Exception as e:
            logger.debug(f"Fulltext index creation: {e}")
    
    async def _create_vector_index(self):
        """Create vector index for Entry nodes."""
        try:
            # Get embedding dimension
            dimension = self.embedding_generator.get_dimension()
            
            query = f"""
            CREATE VECTOR INDEX entry_embeddings IF NOT EXISTS
            FOR (e:Entry)
            ON e.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
            await self.driver.execute_query(query, routing_control=RoutingControl.WRITE)
            logger.info(f"Created vector index for Entry nodes (dim={dimension})")
        except Exception as e:
            logger.warning(f"Vector index creation (requires Neo4j 5.11+): {e}")
    
    async def _create_constraints(self):
        """Create uniqueness constraints and indexes."""
        queries = [
            "CREATE CONSTRAINT entry_id IF NOT EXISTS FOR (e:Entry) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT memory_name IF NOT EXISTS FOR (m:Memory) REQUIRE m.name IS UNIQUE",
            "CREATE INDEX entry_timestamp IF NOT EXISTS FOR (e:Entry) ON (e.timestamp)",
            "CREATE INDEX entry_project IF NOT EXISTS FOR (e:Entry) ON (e.project_id)",
        ]
        
        for query in queries:
            try:
                await self.driver.execute_query(query, routing_control=RoutingControl.WRITE)
            except Exception as e:
                logger.debug(f"Constraint/index creation: {e}")
    
    # ========================================================================
    # Entry Management
    # ========================================================================
    
    async def create_entry(
        self,
        content: str,
        entry_type: str = "journal",
        project_id: Optional[str] = None,
        extract_entities: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EntryWithContext:
        """Create a new entry with optional entity extraction.
        
        Args:
            content: Original text content
            entry_type: Type of entry (journal, code_comment, etc.)
            project_id: Optional project ID
            extract_entities: Whether to extract entities
            metadata: Additional metadata
            
        Returns:
            Entry with extracted entities and relations
        """
        logger.info(f"Creating entry (type={entry_type}, project={project_id})")
        
        # Generate embedding
        embedding = await self.embedding_generator.generate(content)
        
        # Create Entry object
        entry = Entry(
            content=content,
            embedding=embedding,
            type=entry_type,
            project_id=project_id,
            metadata=metadata or {}
        )
        
        # Store entry in database
        query = """
        CREATE (e:Entry {
            id: $id,
            content: $content,
            embedding: $embedding,
            timestamp: datetime($timestamp),
            type: $type,
            project_id: $project_id,
            metadata: $metadata
        })
        RETURN e
        """
        
        await self.driver.execute_query(
            query,
            {
                "id": str(entry.id),
                "content": entry.content,
                "embedding": entry.embedding,
                "timestamp": entry.timestamp.isoformat(),
                "type": entry.type,
                "project_id": entry.project_id,
                "metadata": entry.metadata
            },
            routing_control=RoutingControl.WRITE
        )
        
        # Link to project if specified
        if project_id:
            await self._link_entry_to_project(entry.id, project_id)
        
        # Extract entities if requested
        entities = []
        relations = []
        if extract_entities:
            # Note: Entity extraction would typically use an LLM
            # For now, we'll provide a simple implementation
            # In production, this would call Claude or another LLM
            entities, relations = await self._extract_entities_from_entry(entry)
            
            # Link entry to extracted entities
            for entity in entities:
                await self._link_entry_to_entity(entry.id, entity.name)
        
        logger.info(f"Entry created: {entry.id} ({len(entities)} entities, {len(relations)} relations)")
        
        return EntryWithContext(
            **entry.model_dump(),
            entities=entities,
            relations=relations
        )
    
    async def _extract_entities_from_entry(
        self,
        entry: Entry
    ) -> tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from entry content.
        
        Note: This is a placeholder. In production, this would use an LLM
        (like Claude) to extract entities and relationships.
        
        Args:
            entry: Entry to extract from
            
        Returns:
            Tuple of (entities, relations)
        """
        # TODO: Implement LLM-based entity extraction
        # For now, return empty lists
        # In production, this would:
        # 1. Call Claude with entry.content
        # 2. Ask it to extract entities and relationships
        # 3. Parse the response into Entity and Relation objects
        # 4. Create those entities using create_entities()
        # 5. Create relationships using create_relations()
        
        return [], []
    
    async def _link_entry_to_project(self, entry_id: UUID, project_id: str):
        """Link an entry to a project."""
        query = """
        MATCH (e:Entry {id: $entry_id})
        MATCH (p:Project {id: $project_id})
        MERGE (e)-[:BELONGS_TO]->(p)
        """
        await self.driver.execute_query(
            query,
            {"entry_id": str(entry_id), "project_id": project_id},
            routing_control=RoutingControl.WRITE
        )
    
    async def _link_entry_to_entity(self, entry_id: UUID, entity_name: str):
        """Link an entry to an extracted entity."""
        query = """
        MATCH (e:Entry {id: $entry_id})
        MATCH (m:Memory {name: $entity_name})
        MERGE (e)-[:EXTRACTED_ENTITY]->(m)
        MERGE (m)-[:MENTIONED_IN]->(e)
        """
        await self.driver.execute_query(
            query,
            {"entry_id": str(entry_id), "entity_name": entity_name},
            routing_control=RoutingControl.WRITE
        )
    
    async def search_entries_semantic(
        self,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Entry]:
        """Search entries using semantic similarity.
        
        Args:
            query: Search query
            project_id: Optional project filter
            limit: Maximum results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching entries
        """
        logger.info(f"Semantic search: '{query}' (project={project_id})")
        
        # Generate query embedding
        query_embedding = await self.embedding_generator.generate(query)
        
        # Build Cypher query
        cypher_query = """
        CALL db.index.vector.queryNodes('entry_embeddings', $limit, $query_embedding)
        YIELD node as e, score
        WHERE score >= $threshold
        """
        
        if project_id:
            cypher_query += " AND e.project_id = $project_id"
        
        cypher_query += """
        RETURN e.id as id, 
               e.content as content,
               e.embedding as embedding,
               e.timestamp as timestamp,
               e.type as type,
               e.project_id as project_id,
               e.metadata as metadata,
               score
        ORDER BY score DESC
        """
        
        params = {
            "query_embedding": query_embedding,
            "limit": limit,
            "threshold": similarity_threshold
        }
        if project_id:
            params["project_id"] = project_id
        
        try:
            result = await self.driver.execute_query(
                cypher_query,
                params,
                routing_control=RoutingControl.READ
            )
            
            entries = []
            for record in result.records:
                entries.append(Entry(
                    id=record["id"],
                    content=record["content"],
                    embedding=record["embedding"],
                    timestamp=record["timestamp"],
                    type=record["type"],
                    project_id=record.get("project_id"),
                    metadata=record.get("metadata", {})
                ))
            
            logger.info(f"Found {len(entries)} entries")
            return entries
            
        except Exception as e:
            logger.warning(f"Vector search failed (requires Neo4j 5.11+ with vector index): {e}")
            # Fallback to text search
            return await self._fallback_text_search(query, project_id, limit)
    
    async def _fallback_text_search(
        self,
        query: str,
        project_id: Optional[str],
        limit: int
    ) -> List[Entry]:
        """Fallback text search when vector search unavailable."""
        logger.info("Using fallback text search")
        
        cypher_query = """
        MATCH (e:Entry)
        WHERE e.content CONTAINS $query
        """
        
        if project_id:
            cypher_query += " AND e.project_id = $project_id"
        
        cypher_query += """
        RETURN e.id as id,
               e.content as content,
               e.embedding as embedding,
               e.timestamp as timestamp,
               e.type as type,
               e.project_id as project_id,
               e.metadata as metadata
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        
        result = await self.driver.execute_query(
            cypher_query,
            {"query": query, "project_id": project_id, "limit": limit},
            routing_control=RoutingControl.READ
        )
        
        entries = []
        for record in result.records:
            entries.append(Entry(
                id=record["id"],
                content=record["content"],
                embedding=record.get("embedding"),
                timestamp=record["timestamp"],
                type=record["type"],
                project_id=record.get("project_id"),
                metadata=record.get("metadata", {})
            ))
        
        return entries
    
    async def get_entry_context(self, entity_name: str) -> List[Entry]:
        """Get all entries that mention a specific entity.
        
        Args:
            entity_name: Name of entity
            
        Returns:
            List of entries mentioning this entity
        """
        query = """
        MATCH (m:Memory {name: $entity_name})-[:MENTIONED_IN]->(e:Entry)
        RETURN e.id as id,
               e.content as content,
               e.embedding as embedding,
               e.timestamp as timestamp,
               e.type as type,
               e.project_id as project_id,
               e.metadata as metadata
        ORDER BY e.timestamp DESC
        """
        
        result = await self.driver.execute_query(
            query,
            {"entity_name": entity_name},
            routing_control=RoutingControl.READ
        )
        
        entries = []
        for record in result.records:
            entries.append(Entry(
                id=record["id"],
                content=record["content"],
                embedding=record.get("embedding"),
                timestamp=record["timestamp"],
                type=record["type"],
                project_id=record.get("project_id"),
                metadata=record.get("metadata", {})
            ))
        
        return entries
    
    # ========================================================================
    # Project Management
    # ========================================================================
    
    async def create_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Project:
        """Create a new project.
        
        Args:
            project_id: Unique project ID
            name: Project name
            description: Project description
            status: Project status
            metadata: Additional metadata
            
        Returns:
            Created project
        """
        logger.info(f"Creating project: {name} ({project_id})")
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            status=status,
            metadata=metadata or {}
        )
        
        query = """
        MERGE (p:Project {id: $id})
        SET p.name = $name,
            p.description = $description,
            p.created = datetime($created),
            p.status = $status,
            p.metadata = $metadata
        RETURN p
        """
        
        await self.driver.execute_query(
            query,
            {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created": project.created.isoformat(),
                "status": project.status,
                "metadata": project.metadata
            },
            routing_control=RoutingControl.WRITE
        )
        
        logger.info(f"Project created: {project_id}")
        return project
    
    async def get_project_memories(
        self,
        project_id: str,
        include_cross_project: bool = True,
        relevance_threshold: float = 0.5
    ) -> ProjectMemories:
        """Get all memories for a project.
        
        Args:
            project_id: Project ID
            include_cross_project: Include relevant memories from other projects
            relevance_threshold: Minimum relevance score for cross-project
            
        Returns:
            Project memories
        """
        logger.info(f"Getting memories for project: {project_id}")
        
        # Get project
        project_query = "MATCH (p:Project {id: $id}) RETURN p"
        project_result = await self.driver.execute_query(
            project_query,
            {"id": project_id},
            routing_control=RoutingControl.READ
        )
        
        if not project_result.records:
            raise ValueError(f"Project not found: {project_id}")
        
        p = project_result.records[0]["p"]
        project = Project(
            id=p["id"],
            name=p["name"],
            description=p.get("description", ""),
            created=p["created"],
            status=p.get("status", "active"),
            metadata=p.get("metadata", {})
        )
        
        # Get entries
        entries_query = """
        MATCH (e:Entry)-[:BELONGS_TO]->(p:Project {id: $project_id})
        RETURN e.id as id,
               e.content as content,
               e.embedding as embedding,
               e.timestamp as timestamp,
               e.type as type,
               e.project_id as project_id,
               e.metadata as metadata
        ORDER BY e.timestamp DESC
        """
        
        entries_result = await self.driver.execute_query(
            entries_query,
            {"project_id": project_id},
            routing_control=RoutingControl.READ
        )
        
        entries = [
            Entry(
                id=r["id"],
                content=r["content"],
                embedding=r.get("embedding"),
                timestamp=r["timestamp"],
                type=r["type"],
                project_id=r.get("project_id"),
                metadata=r.get("metadata", {})
            )
            for r in entries_result.records
        ]
        
        # Get entities and relations (using base system methods)
        # For now, get all entities linked to project entries
        entities_query = """
        MATCH (e:Entry)-[:BELONGS_TO]->(:Project {id: $project_id})
        MATCH (e)-[:EXTRACTED_ENTITY]->(m:Memory)
        RETURN DISTINCT m.name as name,
                        m.type as type,
                        m.observations as observations
        """
        
        entities_result = await self.driver.execute_query(
            entities_query,
            {"project_id": project_id},
            routing_control=RoutingControl.READ
        )
        
        entities = [
            Entity(
                name=r["name"],
                type=r["type"],
                observations=r.get("observations", [])
            )
            for r in entities_result.records
        ]
        
        # Get relations between project entities
        entity_names = [e.name for e in entities]
        relations_query = """
        MATCH (source:Memory)-[r]->(target:Memory)
        WHERE source.name IN $names AND target.name IN $names
        RETURN source.name as source,
               target.name as target,
               type(r) as relationType
        """
        
        relations_result = await self.driver.execute_query(
            relations_query,
            {"names": entity_names},
            routing_control=RoutingControl.READ
        )
        
        relations = [
            Relation(
                source=r["source"],
                target=r["target"],
                relationType=r["relationType"]
            )
            for r in relations_result.records
        ]
        
        # TODO: Implement cross-project entity retrieval with relevance scoring
        cross_project_entities = []
        
        return ProjectMemories(
            project=project,
            entries=entries,
            entities=entities,
            relations=relations,
            cross_project_entities=cross_project_entities
        )
    
    # ========================================================================
    # Backward Compatible Methods (from base mcp-neo4j-memory)
    # ========================================================================
    
    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create entities (backward compatible)."""
        logger.info(f"Creating {len(entities)} entities")
        
        for entity in entities:
            query = f"""
            MERGE (m:Memory {{name: $name}})
            SET m.type = $type,
                m.observations = $observations,
                m:`{entity.type}`,
                m.last_updated = datetime()
            ON CREATE SET m.first_seen = datetime()
            RETURN m
            """
            
            await self.driver.execute_query(
                query,
                entity.model_dump(),
                routing_control=RoutingControl.WRITE
            )
        
        return entities
    
    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create relations (backward compatible)."""
        logger.info(f"Creating {len(relations)} relations")
        
        for relation in relations:
            query = f"""
            MATCH (source:Memory {{name: $source}})
            MATCH (target:Memory {{name: $target}})
            MERGE (source)-[r:`{relation.relationType}`]->(target)
            RETURN r
            """
            
            await self.driver.execute_query(
                query,
                relation.model_dump(),
                routing_control=RoutingControl.WRITE
            )
        
        return relations
    
    async def read_graph(self) -> KnowledgeGraph:
        """Read entire graph (backward compatible)."""
        logger.info("Reading full knowledge graph")
        
        query = """
        MATCH (m:Memory)
        OPTIONAL MATCH (m)-[r]->(other:Memory)
        RETURN collect(DISTINCT {
            name: m.name,
            type: m.type,
            observations: m.observations
        }) as entities,
        collect(DISTINCT {
            source: startNode(r).name,
            target: endNode(r).name,
            relationType: type(r)
        }) as relations
        """
        
        result = await self.driver.execute_query(
            query,
            routing_control=RoutingControl.READ
        )
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        
        entities = [
            Entity(**e)
            for e in record["entities"]
            if e.get("name")
        ]
        
        relations = [
            Relation(**r)
            for r in record["relations"]
            if r.get("relationType")
        ]
        
        return KnowledgeGraph(entities=entities, relations=relations)
    
    async def search_memories(self, query: str) -> KnowledgeGraph:
        """Search memories using fulltext (backward compatible)."""
        logger.info(f"Searching memories: '{query}'")
        
        cypher_query = """
        CALL db.index.fulltext.queryNodes('search', $query) 
        YIELD node as m, score
        OPTIONAL MATCH (m)-[r]-(other:Memory)
        RETURN collect(DISTINCT {
            name: m.name,
            type: m.type,
            observations: m.observations
        }) as entities,
        collect(DISTINCT {
            source: startNode(r).name,
            target: endNode(r).name,
            relationType: type(r)
        }) as relations
        """
        
        result = await self.driver.execute_query(
            cypher_query,
            {"query": query},
            routing_control=RoutingControl.READ
        )
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        
        entities = [
            Entity(**e)
            for e in record["entities"]
            if e.get("name")
        ]
        
        relations = [
            Relation(**r)
            for r in record["relations"]
            if r.get("relationType")
        ]
        
        return KnowledgeGraph(entities=entities, relations=relations)
    
    async def find_memories_by_name(self, names: List[str]) -> KnowledgeGraph:
        """Find memories by name (backward compatible)."""
        logger.info(f"Finding {len(names)} memories by name")
        
        query = """
        MATCH (m:Memory)
        WHERE m.name IN $names
        OPTIONAL MATCH (m)-[r]-(other:Memory)
        WHERE other.name IN $names
        RETURN collect(DISTINCT {
            name: m.name,
            type: m.type,
            observations: m.observations
        }) as entities,
        collect(DISTINCT {
            source: startNode(r).name,
            target: endNode(r).name,
            relationType: type(r)
        }) as relations
        """
        
        result = await self.driver.execute_query(
            query,
            {"names": names},
            routing_control=RoutingControl.READ
        )
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        
        entities = [
            Entity(**e)
            for e in record["entities"]
            if e.get("name")
        ]
        
        relations = [
            Relation(**r)
            for r in record["relations"]
            if r.get("relationType")
        ]
        
        return KnowledgeGraph(entities=entities, relations=relations)
    
    async def add_observations(
        self,
        observations: List[ObservationAddition]
    ) -> List[Dict[str, Any]]:
        """Add observations (backward compatible)."""
        logger.info(f"Adding observations to {len(observations)} entities")
        
        query = """
        UNWIND $observations as obs
        MATCH (m:Memory {name: obs.entityName})
        WITH m, [o in obs.observations WHERE NOT o IN coalesce(m.observations, [])] as new
        SET m.observations = coalesce(m.observations, []) + new,
            m.last_updated = datetime()
        RETURN m.name as name, new
        """
        
        result = await self.driver.execute_query(
            query,
            {"observations": [obs.model_dump() for obs in observations]},
            routing_control=RoutingControl.WRITE
        )
        
        return [
            {"entityName": r["name"], "addedObservations": r["new"]}
            for r in result.records
        ]
    
    async def delete_entities(self, entity_names: List[str]) -> None:
        """Delete entities (backward compatible)."""
        logger.info(f"Deleting {len(entity_names)} entities")
        
        query = """
        UNWIND $names as name
        MATCH (m:Memory {name: name})
        DETACH DELETE m
        """
        
        await self.driver.execute_query(
            query,
            {"names": entity_names},
            routing_control=RoutingControl.WRITE
        )
    
    async def delete_observations(
        self,
        deletions: List[ObservationDeletion]
    ) -> None:
        """Delete observations (backward compatible)."""
        logger.info(f"Deleting observations from {len(deletions)} entities")
        
        query = """
        UNWIND $deletions as d
        MATCH (m:Memory {name: d.entityName})
        SET m.observations = [o in coalesce(m.observations, []) WHERE NOT o IN d.observations],
            m.last_updated = datetime()
        """
        
        await self.driver.execute_query(
            query,
            {"deletions": [d.model_dump() for d in deletions]},
            routing_control=RoutingControl.WRITE
        )
    
    async def delete_relations(self, relations: List[Relation]) -> None:
        """Delete relations (backward compatible)."""
        logger.info(f"Deleting {len(relations)} relations")
        
        for relation in relations:
            query = f"""
            MATCH (source:Memory {{name: $source}})-[r:`{relation.relationType}`]->(target:Memory {{name: $target}})
            DELETE r
            """
            
            await self.driver.execute_query(
                query,
                {"source": relation.source, "target": relation.target},
                routing_control=RoutingControl.WRITE
            )

