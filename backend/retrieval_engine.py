"""
retrieval_engine.py

Multi-step hybrid retrieval with dense/sparse embeddings and re-ranking.
"""

import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from backend.backend_config import BackendConfig


class RetrievalEngine:
    """
    Handles hybrid retrieval (dense + sparse) with re-ranking for document chunks.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Qdrant client
        self.qdrant_client = self._init_qdrant_client()

        # Initialize FastEmbed models for query processing
        self.dense_embedder = TextEmbedding(model_name=self.config.dense_embedding_model)
        self.sparse_embedder = SparseTextEmbedding(model_name=self.config.sparse_embedding_model)
        self.rerank_embedder = LateInteractionTextEmbedding(model_name=self.config.rerank_embedding_model)

        self.logger.info("Retrieval engine initialized successfully")

    def _init_qdrant_client(self) -> QdrantClient:
        """Initialize and validate Qdrant connection."""
        client = QdrantClient(
            host=self.config.qdrant_host,
            grpc_port=self.config.qdrant_grpc_port,
            prefer_grpc=True,
            timeout=128
        )

        # Verify collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.config.collection_name not in collection_names:
            raise ValueError(f"Collection '{self.config.collection_name}' not found in Qdrant")

        self.logger.info(f"Qdrant client connected to collection: {self.config.collection_name}")
        return client

    async def retrieve(self, query: str, conversation_context: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform multi-step hybrid retrieval with re-ranking.

        Args:
            query: Current user query
            conversation_context: Previous conversation turns for context

        Returns:
            List of ranked chunks with metadata and scores
        """
        self.logger.info(f"Starting retrieval for query: '{query}'")

        try:
            # Step 1: Build enhanced query with conversation context
            enhanced_query = self._build_enhanced_query(query, conversation_context)

            # Step 2: Generate query embeddings
            query_embeddings = self._generate_query_embeddings(enhanced_query)

            # Step 3: Perform hybrid search (dense + sparse)
            ranked_results = self._retrieve_hybrid(query_embeddings)

            # Step 4: Format results with traceable IDs and metadata
            final_results = self._format_results(ranked_results)

            self.logger.info(f"Retrieval completed: {len(final_results)} results")
            return final_results

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []

    def _retrieve_hybrid(self, query_embeddings):
        prefetch = [
            Prefetch(
                query=query_embeddings["dense"],
                using="dense",
                limit=self.config.top_dense
            ),
            Prefetch(
                query=SparseVector(**query_embeddings["sparse"].as_object()),
                using="sparse",
                limit=self.config.top_sparse
            )
        ]

        result = self.qdrant_client.query_points(
            collection_name=self.config.collection_name,
            prefetch=prefetch,
            query=query_embeddings["rerank"],
            using="rerank",
            with_payload=True,
            limit=self.config.top_rerank
        )

        return [
            {
                "id": hit.id,
                "payload": hit.payload
            }
            for hit in result.points
        ]

    def _build_enhanced_query(self, query: str, conversation_context: List[Dict] = None) -> str:
        """Build an enhanced query using conversation context."""
        if not conversation_context or len(conversation_context) == 0:
            return query

        # Extract relevant context from recent conversation turns
        context_parts = []
        for turn in conversation_context[-2:]:  # Last 2 turns
            # Focus on questions and key terms from answers
            context_parts.append(turn.get('question', ''))
            # You can add answer analysis here for more sophisticated context

        context_text = " ".join(context_parts)
        enhanced_query = f"{context_text} {query}".strip()

        self.logger.debug(f"Enhanced query: {enhanced_query}")
        return enhanced_query

    def _generate_query_embeddings(self, query: str) -> Dict[str, Any]:
        """Generate dense and sparse embeddings for the query."""
        # Generate dense embedding
        dense_vectors = next(self.dense_embedder.query_embed(query))
        sparse_vectors = next(self.sparse_embedder.query_embed(query))
        rerank_vector = next(self.rerank_embedder.query_embed(query))

        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors,
            "rerank": rerank_vector
        }

    def _format_results(self, ranked_results: List[Dict]) -> List[Dict[str, Any]]:
        """Format results with traceable IDs and complete metadata."""
        formatted_results = []

        for result in ranked_results:
            payload = result["payload"]
            formatted_result = {
                "chunk_id": payload["chunk_id"],
                "chunk_text": payload["chunk_text"],
                "document_name": payload["document_name"],
                "page_number": payload["page_number"],
                "section_heading": payload.get("section_heading", ""),
                "metadata": payload.get("metadata", {})
            }
            formatted_results.append(formatted_result)

        return formatted_results

    def health_check(self) -> bool:
        """Check if the retrieval engine is healthy."""
        try:
            # Check Qdrant connection
            self.qdrant_client.get_collections()

            # Check if models are loaded (simple check)
            if not all([self.dense_embedder, self.sparse_embedder, self.rerank_embedder]):
                return False

            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
