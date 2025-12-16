"""Database schema for PGVector embeddings storage."""

from sqlalchemy import (
    Column, Integer, String, Text, REAL, TIMESTAMP, BOOLEAN, JSON,
    Index, UniqueConstraint, create_engine, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import logging
from typing import Optional

Base = declarative_base()


class EmbeddingRecord(Base):
    """SQLAlchemy model for embedding storage in PGVector."""

    __tablename__ = 'embeddings'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Vector Data
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-large dimension (configured)
    embedding_model = Column(String(100), nullable=False)
    embedding_created_at = Column(TIMESTAMP(timezone=True), nullable=False)

    # Content Identity
    chunk_id = Column(String(4096), unique=True, nullable=False)
    text = Column(Text, nullable=False)
    text_hash = Column(String(64), unique=True, nullable=False)
    text_length = Column(Integer)

    # Hierarchical Context
    path = Column(Text)
    level = Column(Integer)
    parent_id = Column(String(4096))
    children_ids = Column(JSONB)

    # Source Tracking
    source_file = Column(Text)
    document_id = Column(String(4096))
    collection_name = Column(String(100))

    # Content Classification
    content_type = Column(String(50))
    value_types = Column(JSONB)
    key_count = Column(Integer)

    # Strategy & Quality
    strategy = Column(String(50))
    confidence = Column(REAL)
    semantic_density = Column(REAL)

    # Domain & Analysis
    domain_type = Column(String(100))
    entity_types = Column(JSONB)
    performance_metrics = Column(JSONB)
    reasoning_content = Column(JSONB)

    # Technical Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    version = Column(String(20), default='1.0')
    processing_pipeline = Column(String(50), default='vector_embeddings')

    __table_args__ = (
        # Unique constraints
        UniqueConstraint('chunk_id', name='uq_embeddings_chunk_id'),
        UniqueConstraint('text_hash', 'document_id', name='uq_embeddings_text_hash_document'),

        # Vector similarity index (created separately due to pgvector requirements)
        # CREATE INDEX embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops)

        # Search indexes
        Index('ix_embeddings_source', 'source_file', 'document_id'),
        Index('ix_embeddings_strategy', 'strategy', 'confidence'),
        Index('ix_embeddings_content_type', 'content_type', 'domain_type'),
        Index('ix_embeddings_path', 'path'),
        Index('ix_embeddings_collection', 'collection_name'),
        Index('ix_embeddings_level', 'level'),
        Index('ix_embeddings_created', 'created_at'),
    )

    def __repr__(self):
        return f"<EmbeddingRecord(id={self.id}, chunk_id='{self.chunk_id}', strategy='{self.strategy}')>"


class EmbeddingSchema:
    """Manages database schema creation and migration for embeddings."""

    def __init__(self, database_url: str):
        """Initialize schema manager.

        Args:
            database_url: PostgreSQL connection URL with pgvector extension
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.logger = logging.getLogger(__name__)

    def create_extension(self) -> None:
        """Create the pgvector extension if it doesn't exist."""
        try:
            with self.engine.connect() as conn:
                # Check if pgvector extension exists
                result = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )).scalar()

                if not result:
                    self.logger.info("Creating pgvector extension...")
                    conn.execute(text("CREATE EXTENSION vector"))
                    conn.commit()
                    self.logger.info("pgvector extension created successfully")
                else:
                    self.logger.info("pgvector extension already exists")

        except Exception as e:
            self.logger.error(f"Failed to create pgvector extension: {e}")
            raise

    def create_tables(self) -> None:
        """Create all embedding tables and indexes."""
        try:
            self.logger.info("Creating database tables...")

            # Create tables
            Base.metadata.create_all(self.engine)

            # Create vector index separately (pgvector specific)
            self._create_vector_indexes()

            self.logger.info("Database schema created successfully")

        except Exception as e:
            self.logger.error(f"Failed to create database schema: {e}")
            raise

    def _create_vector_indexes(self) -> None:
        """Create pgvector-specific indexes."""
        try:
            with self.engine.connect() as conn:
                # Check if vector index already exists
                result = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'embeddings_vector_idx')"
                )).scalar()

                if not result:
                    self.logger.info("Creating vector similarity index...")

                    # Create IVFFlat index for approximate nearest neighbor search
                    conn.execute(text(
                        "CREATE INDEX embeddings_vector_idx ON embeddings "
                        "USING ivfflat (embedding vector_cosine_ops) "
                        "WITH (lists = 100)"
                    ))

                    conn.commit()
                    self.logger.info("Vector similarity index created")
                else:
                    self.logger.info("Vector similarity index already exists")

        except Exception as e:
            self.logger.warning(f"Failed to create vector index: {e}")
            # Vector index is optional for basic functionality

    def drop_tables(self) -> None:
        """Drop all embedding tables (use with caution!)."""
        try:
            self.logger.warning("Dropping database tables...")
            Base.metadata.drop_all(self.engine)
            self.logger.info("Database tables dropped")

        except Exception as e:
            self.logger.error(f"Failed to drop database tables: {e}")
            raise

    def upgrade_schema(self, target_version: Optional[str] = None) -> None:
        """Upgrade database schema to target version.

        Args:
            target_version: Target schema version (future enhancement)
        """
        # Placeholder for future migration system
        self.logger.info(f"Schema upgrade to version {target_version or 'latest'}")

        # For now, just ensure all tables and indexes exist
        self.create_tables()

    def get_schema_info(self) -> dict:
        """Get information about current schema state."""
        try:
            with self.engine.connect() as conn:
                # Check if tables exist
                tables_exist = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'embeddings')"
                )).scalar()

                # Check if pgvector extension exists
                vector_extension = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )).scalar()

                # Check if vector index exists
                vector_index = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'embeddings_vector_idx')"
                )).scalar()

                # Get row count if table exists
                row_count = 0
                if tables_exist:
                    row_count = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()

                return {
                    'database_url': self.database_url.split('@')[1] if '@' in self.database_url else 'configured',
                    'pgvector_extension': vector_extension,
                    'tables_exist': tables_exist,
                    'vector_index_exists': vector_index,
                    'embedding_count': row_count,
                    'schema_version': '1.0'
                }

        except Exception as e:
            self.logger.error(f"Failed to get schema info: {e}")
            return {
                'database_url': 'error',
                'pgvector_extension': False,
                'tables_exist': False,
                'vector_index_exists': False,
                'embedding_count': 0,
                'schema_version': 'unknown'
            }

    def vacuum_analyze(self) -> None:
        """Optimize database performance with VACUUM ANALYZE."""
        try:
            with self.engine.connect() as conn:
                self.logger.info("Running VACUUM ANALYZE on embeddings table...")
                conn.execute(text("VACUUM ANALYZE embeddings"))
                conn.commit()
                self.logger.info("VACUUM ANALYZE completed")

        except Exception as e:
            self.logger.error(f"Failed to run VACUUM ANALYZE: {e}")

    def create_collection_view(self, collection_name: str) -> None:
        """Create a view for a specific collection.

        Args:
            collection_name: Name of the collection
        """
        try:
            view_name = f"collection_{collection_name}"

            with self.engine.connect() as conn:
                # Drop view if exists
                conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))

                # Create view
                conn.execute(text(f"""
                    CREATE VIEW {view_name} AS
                    SELECT * FROM embeddings
                    WHERE collection_name = '{collection_name}'
                    ORDER BY created_at DESC
                """))

                conn.commit()
                self.logger.info(f"Created view '{view_name}' for collection '{collection_name}'")

        except Exception as e:
            self.logger.error(f"Failed to create collection view: {e}")
            raise