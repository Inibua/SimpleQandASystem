"""
vector_store.py

Qdrant vector store operations for storing and managing embeddings.
"""

import logging
from typing import List, Dict, Any
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams, Distance, PointStruct, MultiVectorConfig, MultiVectorComparator,
                                  HnswConfigDiff, SparseVectorParams, Modifier, SparseVector)

from indexer.pdf_models import DocumentChunk
from indexer.indexer_config import IndexerConfig


class VectorStoreIndexer:
    """
    Handles all Qdrant operations for storing and retrieving embeddings.
    """

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.qdrant_client = self._init_qdrant_client()

    def _init_qdrant_client(self) -> QdrantClient:
        """Initialize Qdrant client and ensure collection exists."""
        client = QdrantClient(
            host=self.config.qdrant_host,
            grpc_port=self.config.qdrant_grpc_port,
            prefer_grpc=True,
            timeout=128
        )

        # Ensure collection exists with proper configuration
        self._ensure_collection(client)

        self.logger.info(f"Qdrant client initialized for collection: {self.config.collection_name}")
        return client

    def _ensure_collection(self, client: QdrantClient) -> None:
        """Ensure the Qdrant collection exists with proper vector configuration."""
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.config.collection_name not in collection_names:
            self._create_collection(client)
        else:
            # Verify collection has the right configuration
            self._verify_collection_config(client)

    def _create_collection(self, client: QdrantClient) -> None:
        """Create a new Qdrant collection with hybrid vector support."""
        # Note: We need to know the dense vector dimension
        # Since we're using FastEmbed, we'll assume a standard dimension or get it from config
        # For sentence-transformers/all-MiniLM-L6-v2, the dimension is 384

        dense_vector_size = 384  # Default for all-MiniLM-L6-v2
        rerank_vector_size = 128  # Default for colbertv2.0

        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dense_vector_size,
                    distance=Distance.COSINE
                ),
                "rerank": VectorParams(
                    size=rerank_vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=HnswConfigDiff(m=0)
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            }
        )
        self.logger.info(f"Created Qdrant collection: {self.config.collection_name}")

    def _verify_collection_config(self, client: QdrantClient) -> None:
        """Verify that the existing collection has the right configuration."""
        collection_info = client.get_collection(self.config.collection_name)
        # Basic verification - in production, you might want more detailed checks
        if not all([
            collection_info.config.params.vectors.get("dense"),
            collection_info.config.params.vectors.get("rerank"),
            collection_info.config.params.vectors.get("sparse")
        ]):
            self.logger.warning("Collection exists but missing dense vector configuration")

    def store_embeddings(self, embedding_results: Dict[str, Any]) -> bool:
        """
        Store embeddings and chunk data in Qdrant.

        Args:
            embedding_results: Dictionary containing embeddings and chunks from EmbeddingGenerator

        Returns:
            True if storage successful, False otherwise
        """
        chunks = embedding_results["chunks"]
        dense_embeddings = embedding_results["dense_embeddings"]
        sparse_embeddings = embedding_results["sparse_embeddings"]
        rerank_embeddings = embedding_results["rerank_embeddings"]

        if len(chunks) != len(dense_embeddings) or len(chunks) != len(sparse_embeddings) or len(chunks) != len(rerank_embeddings):
            self.logger.error("Mismatch between chunks and embeddings count")
            return False

        try:
            points = self._prepare_points(chunks, dense_embeddings, sparse_embeddings, rerank_embeddings)
            self._store_points(points)

            self.logger.info(f"Successfully stored {len(points)} points in Qdrant")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store embeddings in Qdrant: {e}")
            return False

    def _prepare_points(self, chunks: List[DocumentChunk],
                        dense_embeddings: List[List[float]],
                        sparse_embeddings: List[Dict[int, float]],
                        rerank_embeddings: List[List[float]]) -> List[PointStruct]:
        """Prepare Qdrant points with hybrid vectors and metadata."""
        points = []

        for i, chunk in enumerate(chunks):
            # Convert sparse dictionary to Qdrant's sparse vector format
            sparse_dict = sparse_embeddings[i]
            indices = list(sparse_dict.keys())
            values = list(sparse_dict.values())

            point = PointStruct(
                id=str(uuid4()),  # Use UUID for better scalability
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": SparseVector(indices=indices, values=values),
                    "rerank": rerank_embeddings[i]
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.chunk_text,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "section_heading": chunk.section_heading,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)

        return points

    def _store_points(self, points: List[PointStruct]) -> None:
        """Store points in Qdrant with error handling."""
        self.qdrant_client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True  # Wait for operation to complete
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection."""
        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.config.collection_name
            )
            return {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}