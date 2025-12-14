"""OpenAI embedding service for generating vector embeddings from text chunks."""

import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import openai
from openai import OpenAI
import os
from dataclasses import dataclass, asdict
import time

from ..models.embedding_chunk import EmbeddingChunk


@dataclass
class VectorEmbedding:
    """Enhanced embedding structure with vector data and comprehensive metadata."""

    # Vector Data
    embedding: List[float]
    embedding_model: str
    embedding_created_at: str

    # Content Identity
    chunk_id: str
    text: str
    text_hash: str
    text_length: int

    # Hierarchical Context
    path: str
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = None

    # Source Tracking
    source_file: Optional[str] = None
    document_id: str = "document"
    collection_name: str = "default"

    # Content Classification
    content_type: str = "mixed"
    value_types: List[str] = None
    key_count: int = 0

    # Strategy & Quality
    strategy: str = "unknown"
    confidence: float = 0.0
    semantic_density: float = 0.0

    # Search & Filtering
    domain_type: str = "general"
    entity_types: List[str] = None
    performance_metrics: List[str] = None
    reasoning_content: List[str] = None

    # Technical Metadata
    created_at: str = ""
    version: str = "1.0"
    processing_pipeline: str = "vector_embeddings"

    def __post_init__(self):
        """Initialize computed fields."""
        if self.children_ids is None:
            self.children_ids = []
        if self.value_types is None:
            self.value_types = []
        if self.entity_types is None:
            self.entity_types = []
        if self.performance_metrics is None:
            self.performance_metrics = []
        if self.reasoning_content is None:
            self.reasoning_content = []
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.text_hash:
            self.text_hash = hashlib.sha256(self.text.encode('utf-8')).hexdigest()


class EmbeddingService:
    """Service for generating embeddings using OpenAI models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: OpenAI embedding model name
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Usage tracking
        self.usage_stats = {
            'total_tokens': 0,
            'total_requests': 0,
            'total_embeddings': 0,
            'failed_requests': 0,
            'last_reset': datetime.utcnow().isoformat()
        }

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic."""

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Generating embeddings for batch of {len(texts)} texts (attempt {attempt + 1})")

                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=1536,  # Configure text-embedding-3-large to return 1536 dimensions
                    timeout=self.timeout
                )

                # Update usage statistics
                self.usage_stats['total_requests'] += 1
                self.usage_stats['total_embeddings'] += len(texts)
                if hasattr(response, 'usage') and response.usage:
                    self.usage_stats['total_tokens'] += response.usage.total_tokens

                # Extract embeddings
                embeddings = [data.embedding for data in response.data]

                self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings

            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)

            except openai.APITimeoutError as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}: {e}")

            except openai.APIError as e:
                self.logger.error(f"OpenAI API error on attempt {attempt + 1}/{self.max_retries}: {e}")

            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")

        # All retries failed
        self.usage_stats['failed_requests'] += 1
        raise Exception(f"Failed to generate embeddings after {self.max_retries} attempts")

    def embed_chunk(
        self,
        chunk: EmbeddingChunk,
        collection_name: str = "default",
        document_id: str = "document"
    ) -> VectorEmbedding:
        """Convert an EmbeddingChunk to a VectorEmbedding with generated embedding.

        Args:
            chunk: EmbeddingChunk from document chunking
            collection_name: Collection to store in
            document_id: Document identifier

        Returns:
            VectorEmbedding with generated embedding vector
        """
        # Generate embedding for the chunk text
        embedding_vector = self.generate_embedding(chunk.text)

        # Create VectorEmbedding with comprehensive metadata
        vector_embedding = VectorEmbedding(
            # Vector Data
            embedding=embedding_vector,
            embedding_model=self.model,
            embedding_created_at=datetime.utcnow().isoformat(),

            # Content Identity
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            text_hash=hashlib.sha256(chunk.text.encode('utf-8')).hexdigest(),
            text_length=chunk.text_length,

            # Hierarchical Context
            path=chunk.path,
            level=chunk.level,
            parent_id=chunk.parent_id,
            children_ids=chunk.children_ids.copy(),

            # Source Tracking
            source_file=chunk.source_file,
            document_id=document_id,
            collection_name=collection_name,

            # Content Classification
            content_type=chunk.content_type,
            value_types=chunk.value_types.copy(),
            key_count=chunk.key_count,

            # Strategy & Quality
            strategy=chunk.strategy,
            confidence=chunk.confidence,
            semantic_density=chunk.semantic_density,

            # Additional metadata (populated from content analysis if available)
            domain_type="general",  # Could be enhanced with domain detection
            entity_types=[],        # Could be enhanced with entity extraction
            performance_metrics=[], # Could be enhanced with metric detection
            reasoning_content=[],   # Could be enhanced with reasoning detection
        )

        self.logger.debug(f"Generated embedding for chunk {chunk.chunk_id}: {len(embedding_vector)} dimensions")

        return vector_embedding

    def embed_chunks(
        self,
        chunks: List[EmbeddingChunk],
        collection_name: str = "default",
        document_id: str = "document"
    ) -> List[VectorEmbedding]:
        """Convert multiple EmbeddingChunks to VectorEmbeddings.

        Args:
            chunks: List of EmbeddingChunks from document chunking
            collection_name: Collection to store in
            document_id: Document identifier

        Returns:
            List of VectorEmbeddings with generated embedding vectors
        """
        if not chunks:
            return []

        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Extract texts for batch processing
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        embedding_vectors = self.generate_embeddings(texts)

        # Create VectorEmbedding objects
        vector_embeddings = []
        for chunk, embedding_vector in zip(chunks, embedding_vectors):
            vector_embedding = VectorEmbedding(
                # Vector Data
                embedding=embedding_vector,
                embedding_model=self.model,
                embedding_created_at=datetime.utcnow().isoformat(),

                # Content Identity
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                text_hash=hashlib.sha256(chunk.text.encode('utf-8')).hexdigest(),
                text_length=chunk.text_length,

                # Hierarchical Context
                path=chunk.path,
                level=chunk.level,
                parent_id=chunk.parent_id,
                children_ids=chunk.children_ids.copy(),

                # Source Tracking
                source_file=chunk.source_file,
                document_id=document_id,
                collection_name=collection_name,

                # Content Classification
                content_type=chunk.content_type,
                value_types=chunk.value_types.copy(),
                key_count=chunk.key_count,

                # Strategy & Quality
                strategy=chunk.strategy,
                confidence=chunk.confidence,
                semantic_density=chunk.semantic_density,
            )

            vector_embeddings.append(vector_embedding)

        self.logger.info(f"Successfully generated {len(vector_embeddings)} vector embeddings")

        return vector_embeddings

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding service configuration."""
        return {
            'model': self.model,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'usage_stats': self.usage_stats.copy(),
            'api_key_configured': bool(self.api_key),
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            'total_tokens': 0,
            'total_requests': 0,
            'total_embeddings': 0,
            'failed_requests': 0,
            'last_reset': datetime.utcnow().isoformat()
        }

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of generate_embeddings for high-throughput scenarios."""
        # For now, wrapping sync version - could be enhanced with async OpenAI client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings, texts)

    async def embed_chunks_async(
        self,
        chunks: List[EmbeddingChunk],
        collection_name: str = "default",
        document_id: str = "document"
    ) -> List[VectorEmbedding]:
        """Async version of embed_chunks for high-throughput scenarios."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_chunks, chunks, collection_name, document_id)