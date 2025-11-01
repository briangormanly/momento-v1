"""
Utility functions for Momento Enhanced Memory System.
"""

from datetime import datetime
from typing import Any, Dict, Optional


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO string.
    
    Args:
        dt: Datetime to format (default: now)
        
    Returns:
        ISO formatted datetime string
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO datetime string.
    
    Args:
        ts: ISO datetime string
        
    Returns:
        Parsed datetime
    """
    return datetime.fromisoformat(ts)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_relevance_score(
    mentions_in_project: int,
    connections_to_project: int,
    recent_activity_score: float,
    manual_boost: float = 0.0
) -> float:
    """Calculate relevance score for an entity to a project.
    
    Args:
        mentions_in_project: Number of times mentioned in project entries
        connections_to_project: Number of connections to project entities
        recent_activity_score: Score based on recency (0.0-1.0)
        manual_boost: Manual relevance adjustment
        
    Returns:
        Relevance score (0.0-1.0)
    """
    # Normalize components
    mention_score = min(mentions_in_project / 10.0, 1.0)  # Cap at 10 mentions
    connection_score = min(connections_to_project / 10.0, 1.0)  # Cap at 10 connections
    
    # Weighted combination
    score = (
        0.4 * mention_score +
        0.3 * connection_score +
        0.2 * recent_activity_score +
        0.1 * manual_boost
    )
    
    return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]


def merge_metadata(
    existing: Optional[Dict[str, Any]],
    new: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge metadata dictionaries.
    
    Args:
        existing: Existing metadata
        new: New metadata to merge
        
    Returns:
        Merged metadata
    """
    result = existing.copy() if existing else {}
    if new:
        result.update(new)
    return result


def sanitize_entity_name(name: str) -> str:
    """Sanitize entity name for database storage.
    
    Args:
        name: Entity name
        
    Returns:
        Sanitized name
    """
    # Remove leading/trailing whitespace
    name = name.strip()
    
    # Normalize whitespace
    name = " ".join(name.split())
    
    return name


def sanitize_relation_type(rel_type: str) -> str:
    """Sanitize relationship type.
    
    Args:
        rel_type: Relationship type
        
    Returns:
        Sanitized relationship type (uppercase with underscores)
    """
    # Convert to uppercase
    rel_type = rel_type.upper()
    
    # Replace spaces with underscores
    rel_type = rel_type.replace(" ", "_")
    
    # Remove special characters except underscores
    rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_")
    
    return rel_type

