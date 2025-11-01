"""
Embedding generation for semantic search.

Supports multiple backends:
- Local: sentence-transformers (default)
- API: OpenAI embeddings (optional)
- Hybrid: Local for most, API for important entries
"""

import logging
from typing import List, Optional
from enum import Enum

logger = logging.getLogger('momento_memory')


class EmbeddingMode(str, Enum):
    """Embedding generation mode."""
    LOCAL = "local"
    API = "api"
    HYBRID = "hybrid"


class EmbeddingGenerator:
    """Generates embeddings for text content."""
    
    def __init__(
        self,
        mode: EmbeddingMode = EmbeddingMode.LOCAL,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None
    ):
        """Initialize embedding generator.
        
        Args:
            mode: Generation mode (local, api, hybrid)
            model_name: Model name for sentence-transformers or OpenAI
            api_key: OpenAI API key (required for api/hybrid modes)
        """
        self.mode = mode
        self.model_name = model_name
        self.api_key = api_key
        self._model = None
        self._client = None
        
        # Lazy load model when first needed
        
    def _load_local_model(self):
        """Lazy load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Local embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise
    
    def _load_api_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("API key required for API embedding mode")
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                raise ImportError(
                    "openai not installed. Install with: pip install openai"
                )
    
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if self.mode == EmbeddingMode.LOCAL or self.mode == EmbeddingMode.HYBRID:
            return self._generate_local(text)
        elif self.mode == EmbeddingMode.API:
            return await self._generate_api(text)
        else:
            raise ValueError(f"Unknown embedding mode: {self.mode}")
    
    def _generate_local(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        self._load_local_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def _generate_api(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        self._load_api_client()
        
        # Use text-embedding-3-small by default (good balance of quality/cost)
        model = self.model_name if "text-embedding" in self.model_name else "text-embedding-3-small"
        
        response = self._client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.mode == EmbeddingMode.LOCAL or self.mode == EmbeddingMode.HYBRID:
            return self._generate_batch_local(texts)
        elif self.mode == EmbeddingMode.API:
            return await self._generate_batch_api(texts)
        else:
            raise ValueError(f"Unknown embedding mode: {self.mode}")
    
    def _generate_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch using local model."""
        self._load_local_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    
    async def _generate_batch_api(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch using OpenAI API."""
        self._load_api_client()
        
        model = self.model_name if "text-embedding" in self.model_name else "text-embedding-3-small"
        
        response = self._client.embeddings.create(
            model=model,
            input=texts
        )
        return [data.embedding for data in response.data]
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator.
        
        Returns:
            Embedding dimension
        """
        if self.mode == EmbeddingMode.LOCAL or self.mode == EmbeddingMode.HYBRID:
            self._load_local_model()
            # Generate a dummy embedding to get dimension
            dummy = self._model.encode("test", convert_to_numpy=True)
            return len(dummy)
        elif self.mode == EmbeddingMode.API:
            # OpenAI embedding dimensions
            if "text-embedding-3-small" in self.model_name:
                return 1536
            elif "text-embedding-3-large" in self.model_name:
                return 3072
            elif "text-embedding-ada-002" in self.model_name:
                return 1536
            else:
                # Default for OpenAI
                return 1536
        else:
            raise ValueError(f"Unknown embedding mode: {self.mode}")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np
    
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

