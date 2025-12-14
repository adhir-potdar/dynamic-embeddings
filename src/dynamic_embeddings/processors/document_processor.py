"""Main processor orchestrating Phase 1 JSON-to-embeddings pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict

from ..core.dynamic_engine import DynamicChunkingEngine
from ..models.embedding_chunk import EmbeddingChunk
from .text_converter import ChunkTextConverter


class DocumentProcessor:
    """Main processor orchestrating Phase 1 JSON-to-embeddings pipeline."""

    def __init__(self, config_name: str = "default"):
        """Initialize processor with configuration.

        Args:
            config_name: Configuration to use for chunking strategy selection
        """
        self.engine = DynamicChunkingEngine(config_name=config_name)
        self.text_converter = ChunkTextConverter(strategy="contextual_description")
        self.logger = logging.getLogger(__name__)

        # Quality thresholds
        self.min_text_length = 10
        self.min_semantic_density = 0.1

    def process_file(self, file_path: Union[str, Path]) -> List[EmbeddingChunk]:
        """Process JSON file through complete Phase 1 pipeline.

        Args:
            file_path: Path to JSON file

        Returns:
            List of embedding-ready chunks with metadata
        """
        file_path = Path(file_path)

        try:
            # Load JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            self.logger.info(f"Processing file: {file_path.name}")

            return self.process_document(json_data, source_file=str(file_path))

        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            raise

    def process_document(self, json_data: Dict[str, Any], source_file: Optional[str] = None) -> List[EmbeddingChunk]:
        """Process JSON document through Phase 1 pipeline.

        Args:
            json_data: JSON document to process
            source_file: Optional source file path

        Returns:
            List of embedding-ready chunks with metadata
        """
        # Step 1: Use existing chunking engine
        chunks, decision_metadata = self.engine.process_document(json_data)

        strategy = decision_metadata.get('chosen_strategy', 'unknown')
        confidence = decision_metadata.get('confidence', 0.0)

        self.logger.info(f"Selected strategy: {strategy} (confidence: {confidence:.2f})")

        # Step 2: Convert chunks to embedding format
        embedding_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'path': chunk.metadata.source_path,
                'level': chunk.metadata.depth_level
            }
            chunk_data = chunk.content

            # Convert to text
            text = self.text_converter.convert_chunk_to_text(chunk_data, chunk_metadata)

            # Calculate quality metrics
            semantic_density = self.text_converter.calculate_semantic_density(chunk_data)

            # Create embedding chunk
            embedding_chunk = EmbeddingChunk(
                text=text,
                chunk_id=chunk.metadata.chunk_id,
                path=chunk.metadata.source_path or f'root.{i}',
                level=chunk.metadata.depth_level,
                content_type=self._detect_content_type(chunk_data),
                key_count=len(chunk_data) if isinstance(chunk_data, dict) else 0,
                value_types=self._get_value_types(chunk_data),
                strategy=strategy,
                confidence=confidence,
                semantic_density=semantic_density,
                source_file=source_file
            )

            # Quality validation
            if self._validate_chunk_quality(embedding_chunk):
                embedding_chunks.append(embedding_chunk)
            else:
                self.logger.warning(f"Chunk {i} failed quality validation, skipping")

        self.logger.info(f"Generated {len(embedding_chunks)} high-quality embedding chunks")

        return embedding_chunks

    def process_batch(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[EmbeddingChunk]]:
        """Process multiple files in batch.

        Args:
            file_paths: List of JSON file paths

        Returns:
            Dictionary mapping file paths to embedding chunks
        """
        results = {}

        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                results[str(file_path)] = chunks
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                results[str(file_path)] = []

        total_chunks = sum(len(chunks) for chunks in results.values())
        self.logger.info(f"Batch processing complete: {total_chunks} total chunks from {len(file_paths)} files")

        return results

    def _detect_content_type(self, chunk: Dict[str, Any]) -> str:
        """Detect primary content type of chunk."""
        if not chunk:
            return "empty"

        types = set()
        for value in chunk.values():
            if isinstance(value, str):
                types.add("text")
            elif isinstance(value, (int, float)):
                types.add("numeric")
            elif isinstance(value, (dict, list)):
                types.add("structured")

        if len(types) == 1:
            return list(types)[0]
        else:
            return "mixed"

    def _get_value_types(self, chunk: Dict[str, Any]) -> List[str]:
        """Get list of value types in chunk."""
        types = []
        for value in chunk.values():
            if isinstance(value, str):
                types.append("string")
            elif isinstance(value, int):
                types.append("integer")
            elif isinstance(value, float):
                types.append("float")
            elif isinstance(value, bool):
                types.append("boolean")
            elif isinstance(value, dict):
                types.append("object")
            elif isinstance(value, list):
                types.append("array")
            else:
                types.append("other")

        return list(set(types))  # Remove duplicates

    def _validate_chunk_quality(self, chunk: EmbeddingChunk) -> bool:
        """Validate chunk meets quality thresholds."""
        # Check minimum text length
        if chunk.text_length < self.min_text_length:
            return False

        # Check semantic density
        if chunk.semantic_density < self.min_semantic_density:
            return False

        # Check for meaningful content
        if not chunk.text.strip():
            return False

        return True

    def export_chunks(self, chunks: List[EmbeddingChunk], output_path: Union[str, Path], format: str = "json") -> None:
        """Export chunks to file for inspection or further processing.

        Args:
            chunks: List of embedding chunks
            output_path: Output file path
            format: Export format (json, jsonl, csv)
        """
        output_path = Path(output_path)

        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(chunk) for chunk in chunks], f, indent=2, ensure_ascii=False)

        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')

        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame([asdict(chunk) for chunk in chunks])
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported {len(chunks)} chunks to {output_path}")

    def get_processing_stats(self, chunks: List[EmbeddingChunk]) -> Dict[str, Any]:
        """Get processing statistics for chunks.

        Args:
            chunks: List of processed chunks

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}

        strategies = [chunk.strategy for chunk in chunks]
        content_types = [chunk.content_type for chunk in chunks]
        text_lengths = [chunk.text_length for chunk in chunks]
        semantic_densities = [chunk.semantic_density for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "avg_semantic_density": sum(semantic_densities) / len(semantic_densities),
            "strategy_distribution": {strategy: strategies.count(strategy) for strategy in set(strategies)},
            "content_type_distribution": {ctype: content_types.count(ctype) for ctype in set(content_types)},
            "quality_metrics": {
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths),
                "min_semantic_density": min(semantic_densities),
                "max_semantic_density": max(semantic_densities)
            }
        }

        return stats