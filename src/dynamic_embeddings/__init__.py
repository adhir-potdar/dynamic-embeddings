"""Dynamic JSON Embeddings - Intelligent chunking for hierarchical JSON documents."""

from .core.dynamic_engine import DynamicChunkingEngine
from .engine.decision_engine import DecisionEngine, ChunkingStrategy
from .strategies.base_strategy import DocumentChunk, ChunkMetadata
from .config.analyzer_config import AnalyzerConfig
from .models.embedding_chunk import EmbeddingChunk
from .processors.document_processor import DocumentProcessor
from .processors.text_converter import ChunkTextConverter

__version__ = "1.0.0"
__author__ = "Adhir Potdar"
__email__ = "adhir.potdar@isanasystems.com"

__all__ = [
    'DynamicChunkingEngine',
    'DecisionEngine',
    'ChunkingStrategy',
    'DocumentChunk',
    'ChunkMetadata',
    'AnalyzerConfig',
    'EmbeddingChunk',
    'DocumentProcessor',
    'ChunkTextConverter'
]