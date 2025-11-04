"""
Basic usage examples for Momento Enhanced Memory.

This script demonstrates direct usage of the MomentoMemory class
without the MCP server layer.
"""

import asyncio
from datetime import datetime

from neo4j import AsyncGraphDatabase

from momento_memory.embeddings import EmbeddingGenerator
from momento_memory.memory import MomentoMemory
from momento_memory.models import Entity, Relation


async def example_1_basic_entries():
    """Example 1: Create basic journal entries."""
    print("\n=== Example 1: Basic Journal Entries ===\n")
    
    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "neo4j_password")
    )
    
    # Create memory system
    embedding_gen = EmbeddingGenerator()
    memory = MomentoMemory(driver, embedding_gen)
    await memory.initialize()
    
    # Create a project
    project = await memory.create_project(
        project_id="example-001",
        name="Example Project",
        description="Testing Momento features"
    )
    print(f"Created project: {project.name}")
    
    # Create journal entries
    entry1 = await memory.create_entry(
        content="Today I learned about Neo4j graph databases. They're perfect for storing interconnected knowledge.",
        entry_type="journal",
        project_id="example-001"
    )
    print(f"\nCreated entry 1: {entry1.id}")
    print(f"Content preview: {entry1.content[:60]}...")
    
    entry2 = await memory.create_entry(
        content="I implemented vector similarity search using sentence transformers. The embeddings work great for semantic search.",
        entry_type="journal",
        project_id="example-001"
    )
    print(f"\nCreated entry 2: {entry2.id}")
    print(f"Content preview: {entry2.content[:60]}...")
    
    await driver.close()
    print("\n✅ Example 1 complete!")


async def example_2_semantic_search():
    """Example 2: Semantic search across entries."""
    print("\n=== Example 2: Semantic Search ===\n")
    
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "neo4j_password")
    )
    
    embedding_gen = EmbeddingGenerator()
    memory = MomentoMemory(driver, embedding_gen)
    
    # Search for entries
    query = "graph databases and knowledge storage"
    print(f"Searching for: '{query}'")
    
    results = await memory.search_entries_semantic(
        query=query,
        project_id="example-001",
        limit=5
    )
    
    print(f"\nFound {len(results)} results:")
    for i, entry in enumerate(results, 1):
        print(f"\n{i}. [{entry.type}] {entry.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"   {entry.content[:100]}...")
    
    await driver.close()
    print("\n✅ Example 2 complete!")


async def example_3_traditional_entities():
    """Example 3: Traditional entity and relationship creation."""
    print("\n=== Example 3: Traditional Entities ===\n")
    
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "neo4j_password")
    )
    
    embedding_gen = EmbeddingGenerator()
    memory = MomentoMemory(driver, embedding_gen)
    
    # Create entities
    entities = [
        Entity(
            name="Alice Johnson",
            type="person",
            observations=["Software engineer", "Specializes in AI", "Lives in Seattle"]
        ),
        Entity(
            name="Momento Project",
            type="project",
            observations=["Memory system", "Uses Neo4j", "Started Oct 2025"]
        )
    ]
    
    created = await memory.create_entities(entities)
    print(f"Created {len(created)} entities:")
    for entity in created:
        print(f"  - {entity.name} ({entity.type})")
    
    # Create relationship
    relations = [
        Relation(
            source="Alice Johnson",
            target="Momento Project",
            relationType="WORKS_ON"
        )
    ]
    
    created_rels = await memory.create_relations(relations)
    print(f"\nCreated {len(created_rels)} relationships:")
    for rel in created_rels:
        print(f"  - {rel.source} --[{rel.relationType}]--> {rel.target}")
    
    # Read graph
    graph = await memory.read_graph()
    print(f"\nKnowledge graph now contains:")
    print(f"  - {len(graph.entities)} entities")
    print(f"  - {len(graph.relations)} relationships")
    
    await driver.close()
    print("\n✅ Example 3 complete!")


async def example_4_project_memories():
    """Example 4: Retrieve project memories."""
    print("\n=== Example 4: Project Memories ===\n")
    
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "neo4j_password")
    )
    
    embedding_gen = EmbeddingGenerator()
    memory = MomentoMemory(driver, embedding_gen)
    
    # Get all memories for project
    project_memories = await memory.get_project_memories(
        project_id="example-001",
        include_cross_project=True
    )
    
    print(f"Project: {project_memories.project.name}")
    print(f"Status: {project_memories.project.status}")
    print(f"\nStatistics:")
    print(f"  - {len(project_memories.entries)} entries")
    print(f"  - {len(project_memories.entities)} entities")
    print(f"  - {len(project_memories.relations)} relationships")
    
    if project_memories.entries:
        print(f"\nRecent entries:")
        for entry in project_memories.entries[:3]:
            print(f"  - [{entry.type}] {entry.timestamp.strftime('%Y-%m-%d')}")
            print(f"    {entry.content[:80]}...")
    
    await driver.close()
    print("\n✅ Example 4 complete!")


async def example_5_semantic_relationships():
    """Example 5: Semantic relationship tracking between entries and entities."""
    print("\n=== Example 5: Semantic Relationship Tracking ===\n")
    
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "neo4j_password")
    )
    
    embedding_gen = EmbeddingGenerator()
    memory = MomentoMemory(driver, embedding_gen)
    
    # First, create an entry with entity extraction enabled
    # In production, this would use an LLM to extract entities and relationships
    entry = await memory.create_entry(
        content="Sarah Chen joined TechCorp as a senior engineer. She will be working on the AI platform.",
        entry_type="journal",
        project_id="example-001",
        extract_entities=True  # This would trigger LLM extraction in production
    )
    print(f"Created entry: {entry.id}")
    
    # For demonstration, manually create entities and relationships
    # This simulates what an LLM would extract
    entities = [
        Entity(name="Sarah Chen", type="person", observations=["Senior engineer"]),
        Entity(name="TechCorp", type="company", observations=["Tech company"]),
        Entity(name="AI Platform", type="technology", observations=["Software project"])
    ]
    await memory.create_entities(entities)
    print(f"\nManually created entities (simulating LLM extraction):")
    for entity in entities:
        print(f"  - {entity.name} ({entity.type})")
    
    # Create relationships between entities
    relations = [
        Relation(source="Sarah Chen", target="TechCorp", relationType="WORKS_AT"),
        Relation(source="Sarah Chen", target="AI Platform", relationType="WORKS_ON")
    ]
    await memory.create_relations(relations)
    print(f"\nCreated relationships between entities:")
    for rel in relations:
        print(f"  - {rel.source} --[{rel.relationType}]--> {rel.target}")
    
    # Link the entry to entities with semantic relationships
    # This is what the system does automatically when extract_entities=True
    for entity in entities:
        await memory._link_entry_to_entity(entry.id, entity.name)
    
    for relation in relations:
        await memory._link_entry_to_entities_via_relation(entry.id, relation)
    
    print(f"\n✨ Semantic relationship tracking created:")
    print(f"   Entry is now connected to entities with typed relationships:")
    print(f"   - (Entry)-[:MENTIONS_WORKS_AT {{role: 'source'}}]->(Sarah Chen)")
    print(f"   - (Entry)-[:MENTIONS_WORKS_AT {{role: 'target'}}]->(TechCorp)")
    print(f"   - (Entry)-[:MENTIONS_WORKS_ON {{role: 'source'}}]->(Sarah Chen)")
    print(f"   - (Entry)-[:MENTIONS_WORKS_ON {{role: 'target'}}]->(AI Platform)")
    
    print(f"\n🔍 Query examples you can now run in Neo4j Browser:")
    print(f"   // Find all entries about employment relationships")
    print(f"   MATCH (e:Entry)-[:MENTIONS_WORKS_AT]->(m:Memory)")
    print(f"   RETURN e.content, m.name, m.type")
    print(f"")
    print(f"   // Find entries about specific people working on technology")
    print(f"   MATCH (e:Entry)-[:MENTIONS_WORKS_ON]->(person:Memory {{type: 'person'}})")
    print(f"   MATCH (e)-[:MENTIONS_WORKS_ON]->(tech:Memory {{type: 'technology'}})")
    print(f"   RETURN e.content, person.name, tech.name")
    
    await driver.close()
    print("\n✅ Example 5 complete!")


async def main():
    """Run all examples."""
    print("=" * 70)
    print("MOMENTO ENHANCED MEMORY - USAGE EXAMPLES")
    print("=" * 70)
    
    try:
        await example_1_basic_entries()
        await asyncio.sleep(1)  # Give Neo4j time to process
        
        await example_2_semantic_search()
        await asyncio.sleep(1)
        
        await example_3_traditional_entities()
        await asyncio.sleep(1)
        
        await example_4_project_memories()
        await asyncio.sleep(1)
        
        await example_5_semantic_relationships()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check Neo4j browser: http://localhost:7474")
        print("2. Try your own queries")
        print("3. Integrate with Claude Desktop")
        print("4. Run the Cypher queries from Example 5 to see semantic relationships")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Neo4j is running:")
        print("  cd infra && docker-compose up -d")


if __name__ == "__main__":
    asyncio.run(main())

