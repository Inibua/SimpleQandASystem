"""
embedding_generator.py

FastEmbed-based embedding generation for dense, sparse, and rerank embeddings.
"""

import logging
from typing import List, Dict, Any
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from indexer.pdf_models import DocumentChunk
from indexer.indexer_config import IndexerConfig


class EmbeddingGenerator:
    """
    Generates embeddings using FastEmbed for efficient embedding computation.
    """

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize FastEmbed models
        self.dense_embedder = TextEmbedding(model_name=self.config.dense_embedding_model)
        self.sparse_embedder = SparseTextEmbedding(model_name=self.config.sparse_embedding_model)
        self.re_rank_embedder = LateInteractionTextEmbedding(model_name=self.config.rerank_embedding_model)
        self.logger.info("FastEmbed models initialized successfully")

    def generate_dense_embeddings(self, chunks: List[DocumentChunk]):
        """
        Generate dense embeddings for document chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of dense embedding vectors
        """
        texts = [chunk.chunk_text for chunk in chunks]

        # FastEmbed returns an iterator of embeddings
        embeddings = list(self.dense_embedder.embed(texts))
        self.logger.info(f"Generated dense embeddings for {len(chunks)} chunks")
        return embeddings

    def generate_sparse_embeddings(self, chunks: List[DocumentChunk]) -> List[Dict[int, float]]:
        """
        Generate sparse embeddings for document chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of sparse embeddings as dictionaries {index: score}
        """
        texts = [chunk.chunk_text for chunk in chunks]

        # FastEmbed sparse embeddings return (indices, values) tuples
        sparse_embeddings = list(self.sparse_embedder.embed(texts))
        # Convert to dictionary format for Qdrant
        sparse_vectors = []
        for sparse_embedding in sparse_embeddings:
            sparse_dict = dict(zip(sparse_embedding.indices.tolist(), sparse_embedding.values.tolist()))
            sparse_vectors.append(sparse_dict)
        self.logger.info(f"Generated sparse embeddings for {len(chunks)} chunks")
        return sparse_vectors

    def generate_rerank_embeddings(self, chunks: List[DocumentChunk]):
        """
        Generate embeddings suitable for reranking (cross-encoder style).
        Note: This might require a different approach as FastEmbed doesn't have direct rerank models.
        For now, we'll return the chunk texts for external reranking.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of chunk texts for reranking
        """
        texts = [chunk.chunk_text for chunk in chunks]
        # The backend will handle reranking with a separate model
        rerank_embeddings = list(self.re_rank_embedder.embed(texts))
        self.logger.info(f"Prepared {len(chunks)} chunks for reranking")
        return rerank_embeddings

    def generate_all_embeddings(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Generate all types of embeddings in one call.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary containing dense, sparse, and rerank embeddings
        """
        self.logger.info(f"Generating all embeddings for {len(chunks)} chunks")

        return {
            "dense_embeddings": self.generate_dense_embeddings(chunks),
            "sparse_embeddings": self.generate_sparse_embeddings(chunks),
            "rerank_embeddings": self.generate_rerank_embeddings(chunks),
            "chunks": chunks  # Keep reference to original chunks
        }
