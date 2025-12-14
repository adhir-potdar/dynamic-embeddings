"""Vector store service for managing embeddings in PGVector database."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from sqlalchemy import text, desc, asc, and_, or_
from sqlalchemy.orm import Session

from ..database.schema import EmbeddingRecord
from ..database.connection import DatabaseConnection
from .embedding_service import VectorEmbedding


class VectorStore:
    """Service for storing and retrieving vector embeddings in PGVector."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize vector store.

        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
        self.logger = logging.getLogger(__name__)

    def insert_embedding(self, vector_embedding: VectorEmbedding) -> str:
        """Insert a single vector embedding into the database.

        Args:
            vector_embedding: VectorEmbedding to store

        Returns:
            Database ID of the inserted record
        """
        try:
            with self.db.get_session() as session:
                # Create database record
                record = EmbeddingRecord(
                    # Vector Data
                    embedding=vector_embedding.embedding,
                    embedding_model=vector_embedding.embedding_model,
                    embedding_created_at=datetime.fromisoformat(vector_embedding.embedding_created_at),

                    # Content Identity
                    chunk_id=vector_embedding.chunk_id,
                    text=vector_embedding.text,
                    text_hash=vector_embedding.text_hash,
                    text_length=vector_embedding.text_length,

                    # Hierarchical Context
                    path=vector_embedding.path,
                    level=vector_embedding.level,
                    parent_id=vector_embedding.parent_id,
                    children_ids=vector_embedding.children_ids,

                    # Source Tracking
                    source_file=vector_embedding.source_file,
                    document_id=vector_embedding.document_id,
                    collection_name=vector_embedding.collection_name,

                    # Content Classification
                    content_type=vector_embedding.content_type,
                    value_types=vector_embedding.value_types,
                    key_count=vector_embedding.key_count,

                    # Strategy & Quality
                    strategy=vector_embedding.strategy,
                    confidence=vector_embedding.confidence,
                    semantic_density=vector_embedding.semantic_density,

                    # Domain & Analysis
                    domain_type=vector_embedding.domain_type,
                    entity_types=vector_embedding.entity_types,
                    performance_metrics=vector_embedding.performance_metrics,
                    reasoning_content=vector_embedding.reasoning_content,

                    # Technical Metadata
                    version=vector_embedding.version,
                    processing_pipeline=vector_embedding.processing_pipeline,
                )

                session.add(record)
                session.commit()

                self.logger.debug(f"Inserted embedding with chunk_id: {vector_embedding.chunk_id}")
                return str(record.id)

        except Exception as e:
            self.logger.error(f"Failed to insert embedding {vector_embedding.chunk_id}: {e}")
            raise

    def insert_embeddings(self, vector_embeddings: List[VectorEmbedding]) -> List[str]:
        """Insert multiple vector embeddings into the database.

        Args:
            vector_embeddings: List of VectorEmbeddings to store

        Returns:
            List of database IDs for the inserted records
        """
        if not vector_embeddings:
            return []

        try:
            with self.db.get_session() as session:
                records = []

                for embedding in vector_embeddings:
                    record = EmbeddingRecord(
                        # Vector Data
                        embedding=embedding.embedding,
                        embedding_model=embedding.embedding_model,
                        embedding_created_at=datetime.fromisoformat(embedding.embedding_created_at),

                        # Content Identity
                        chunk_id=embedding.chunk_id,
                        text=embedding.text,
                        text_hash=embedding.text_hash,
                        text_length=embedding.text_length,

                        # Hierarchical Context
                        path=embedding.path,
                        level=embedding.level,
                        parent_id=embedding.parent_id,
                        children_ids=embedding.children_ids,

                        # Source Tracking
                        source_file=embedding.source_file,
                        document_id=embedding.document_id,
                        collection_name=embedding.collection_name,

                        # Content Classification
                        content_type=embedding.content_type,
                        value_types=embedding.value_types,
                        key_count=embedding.key_count,

                        # Strategy & Quality
                        strategy=embedding.strategy,
                        confidence=embedding.confidence,
                        semantic_density=embedding.semantic_density,

                        # Domain & Analysis
                        domain_type=embedding.domain_type,
                        entity_types=embedding.entity_types,
                        performance_metrics=embedding.performance_metrics,
                        reasoning_content=embedding.reasoning_content,

                        # Technical Metadata
                        version=embedding.version,
                        processing_pipeline=embedding.processing_pipeline,
                    )
                    records.append(record)

                # Bulk insert
                session.add_all(records)
                session.commit()

                # Get the IDs
                record_ids = [str(record.id) for record in records]

                self.logger.info(f"Inserted {len(vector_embeddings)} embeddings into database")
                return record_ids

        except Exception as e:
            self.logger.error(f"Failed to insert {len(vector_embeddings)} embeddings: {e}")
            raise

    def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[EmbeddingRecord, float]]:
        """Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            collection_name: Filter by collection name
            filters: Additional filters (strategy, content_type, etc.)
            similarity_threshold: Minimum similarity score

        Returns:
            List of (EmbeddingRecord, similarity_score) tuples
        """
        try:
            with self.db.get_session() as session:
                # Build base query with similarity
                query = session.query(
                    EmbeddingRecord,
                    (1 - EmbeddingRecord.embedding.cosine_distance(query_vector)).label('similarity')
                )

                # Apply filters
                if collection_name:
                    query = query.filter(EmbeddingRecord.collection_name == collection_name)

                if filters:
                    for key, value in filters.items():
                        if hasattr(EmbeddingRecord, key):
                            if isinstance(value, list):
                                query = query.filter(getattr(EmbeddingRecord, key).in_(value))
                            else:
                                query = query.filter(getattr(EmbeddingRecord, key) == value)

                # Get all results ordered by similarity (no limit yet)
                results = query.order_by(desc('similarity')).all()

                # Filter by threshold and apply limit
                detached_results = []
                for record, similarity in results:
                    similarity_score = float(similarity)

                    # Apply similarity threshold filter
                    if similarity_threshold > 0 and similarity_score < similarity_threshold:
                        continue

                    # Access all needed attributes while in session
                    detached_record = type('EmbeddingRecord', (), {
                        'id': record.id,
                        'chunk_id': record.chunk_id,
                        'text': record.text,
                        'strategy': record.strategy,
                        'content_type': record.content_type,
                        'collection_name': record.collection_name,
                        'document_id': record.document_id,
                        'confidence': record.confidence,
                        'semantic_density': record.semantic_density,
                        'created_at': record.created_at,
                        'path': record.path,
                        'level': record.level
                    })()
                    detached_results.append((detached_record, similarity_score))

                    # Apply limit after threshold filtering
                    if len(detached_results) >= limit:
                        break

                # Get total embeddings count for context
                total_count = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.collection_name == collection_name if collection_name else True
                ).count()

                self.logger.info(f"Similarity search returned {len(detached_results)} results from {total_count} total embeddings")
                return detached_results, total_count

        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            raise

    def similarity_search_by_text(
        self,
        query_text: str,
        embedding_service,
        limit: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> Tuple[List[Tuple[Any, float]], int]:
        """Perform similarity search using query text.

        Args:
            query_text: Text to search for
            embedding_service: EmbeddingService to generate query embedding
            limit: Maximum number of results
            collection_name: Filter by collection name
            filters: Additional filters
            similarity_threshold: Minimum similarity score

        Returns:
            Tuple of (List of (EmbeddingRecord, similarity_score) tuples, total_count)
        """
        # Generate embedding for query text
        query_vector = embedding_service.generate_embedding(query_text)

        return self.similarity_search(
            query_vector=query_vector,
            limit=limit,
            collection_name=collection_name,
            filters=filters,
            similarity_threshold=similarity_threshold
        )

    def get_by_chunk_id(self, chunk_id: str) -> Optional[EmbeddingRecord]:
        """Get embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            EmbeddingRecord if found, None otherwise
        """
        try:
            with self.db.get_session() as session:
                return session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.chunk_id == chunk_id
                ).first()

        except Exception as e:
            self.logger.error(f"Failed to get embedding by chunk_id {chunk_id}: {e}")
            return None

    def get_by_document_id(self, document_id: str, collection_name: Optional[str] = None) -> List[EmbeddingRecord]:
        """Get all embeddings for a document.

        Args:
            document_id: Document identifier
            collection_name: Optional collection filter

        Returns:
            List of EmbeddingRecords
        """
        try:
            with self.db.get_session() as session:
                query = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.document_id == document_id
                )

                if collection_name:
                    query = query.filter(EmbeddingRecord.collection_name == collection_name)

                return query.order_by(EmbeddingRecord.level, EmbeddingRecord.path).all()

        except Exception as e:
            self.logger.error(f"Failed to get embeddings for document {document_id}: {e}")
            return []

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection.

        Args:
            collection_name: Collection name

        Returns:
            Statistics dictionary
        """
        try:
            with self.db.get_session() as session:
                # Basic counts
                total_embeddings = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.collection_name == collection_name
                ).count()

                if total_embeddings == 0:
                    return {
                        'collection_name': collection_name,
                        'total_embeddings': 0,
                        'strategies': {},
                        'content_types': {},
                        'avg_semantic_density': 0.0,
                        'size_stats': {}
                    }

                # Strategy distribution
                strategy_stats = session.query(
                    EmbeddingRecord.strategy,
                    text('COUNT(*)')
                ).filter(
                    EmbeddingRecord.collection_name == collection_name
                ).group_by(EmbeddingRecord.strategy).all()

                # Content type distribution
                content_type_stats = session.query(
                    EmbeddingRecord.content_type,
                    text('COUNT(*)')
                ).filter(
                    EmbeddingRecord.collection_name == collection_name
                ).group_by(EmbeddingRecord.content_type).all()

                # Quality metrics
                quality_stats = session.query(
                    text('AVG(semantic_density)'),
                    text('AVG(confidence)'),
                    text('AVG(text_length)'),
                    text('MIN(text_length)'),
                    text('MAX(text_length)')
                ).filter(
                    EmbeddingRecord.collection_name == collection_name
                ).first()

                return {
                    'collection_name': collection_name,
                    'total_embeddings': total_embeddings,
                    'strategies': {strategy: count for strategy, count in strategy_stats},
                    'content_types': {content_type: count for content_type, count in content_type_stats},
                    'avg_semantic_density': float(quality_stats[0] or 0),
                    'avg_confidence': float(quality_stats[1] or 0),
                    'size_stats': {
                        'avg_text_length': float(quality_stats[2] or 0),
                        'min_text_length': int(quality_stats[3] or 0),
                        'max_text_length': int(quality_stats[4] or 0),
                    }
                }

        except Exception as e:
            self.logger.error(f"Failed to get collection stats for {collection_name}: {e}")
            return {'error': str(e)}

    def delete_by_collection(self, collection_name: str) -> int:
        """Delete all embeddings in a collection.

        Args:
            collection_name: Collection name

        Returns:
            Number of deleted records
        """
        try:
            with self.db.get_session() as session:
                deleted_count = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.collection_name == collection_name
                ).delete()

                self.logger.info(f"Deleted {deleted_count} embeddings from collection '{collection_name}'")
                return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

    def delete_by_document(self, document_id: str, collection_name: Optional[str] = None) -> int:
        """Delete all embeddings for a document.

        Args:
            document_id: Document identifier
            collection_name: Optional collection filter

        Returns:
            Number of deleted records
        """
        try:
            with self.db.get_session() as session:
                query = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.document_id == document_id
                )

                if collection_name:
                    query = query.filter(EmbeddingRecord.collection_name == collection_name)

                deleted_count = query.delete()

                self.logger.info(f"Deleted {deleted_count} embeddings for document '{document_id}'")
                return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            raise

    def update_embedding(self, chunk_id: str, updates: Dict[str, Any]) -> bool:
        """Update an embedding record.

        Args:
            chunk_id: Chunk identifier
            updates: Dictionary of fields to update

        Returns:
            True if update was successful
        """
        try:
            with self.db.get_session() as session:
                record = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.chunk_id == chunk_id
                ).first()

                if not record:
                    self.logger.warning(f"Embedding with chunk_id {chunk_id} not found for update")
                    return False

                # Update fields
                for key, value in updates.items():
                    if hasattr(record, key):
                        setattr(record, key, value)

                # Update timestamp
                record.updated_at = datetime.utcnow()

                session.commit()
                self.logger.debug(f"Updated embedding {chunk_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to update embedding {chunk_id}: {e}")
            return False

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their statistics.

        Returns:
            List of collection information dictionaries
        """
        try:
            with self.db.get_session() as session:
                # Get all collections
                collections = session.query(EmbeddingRecord.collection_name).distinct().all()

                collection_info = []
                for (collection_name,) in collections:
                    stats = self.get_collection_stats(collection_name)
                    collection_info.append(stats)

                return collection_info

        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []

    def get_hierarchical_chunks(
        self,
        parent_path: str,
        collection_name: Optional[str] = None,
        max_depth: Optional[int] = None
    ) -> List[EmbeddingRecord]:
        """Get chunks in a hierarchical structure.

        Args:
            parent_path: Parent path to search under
            collection_name: Optional collection filter
            max_depth: Maximum depth to search

        Returns:
            List of EmbeddingRecords in hierarchical order
        """
        try:
            with self.db.get_session() as session:
                query = session.query(EmbeddingRecord).filter(
                    EmbeddingRecord.path.like(f"{parent_path}%")
                )

                if collection_name:
                    query = query.filter(EmbeddingRecord.collection_name == collection_name)

                if max_depth is not None:
                    base_level = len(parent_path.split('.'))
                    query = query.filter(EmbeddingRecord.level <= base_level + max_depth)

                return query.order_by(EmbeddingRecord.level, EmbeddingRecord.path).all()

        except Exception as e:
            self.logger.error(f"Failed to get hierarchical chunks for path {parent_path}: {e}")
            return []