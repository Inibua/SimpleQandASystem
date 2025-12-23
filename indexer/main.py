"""
main.py

Main indexing pipeline with FastEmbed and separated embedding/store components.
"""

import logging
from pathlib import Path

from indexer.pdf_processor import PDFProcessor
from indexer.chunker import DocumentChunker
from indexer.embedder import EmbeddingGenerator
from indexer.vector_store import VectorStoreIndexer
from indexer.indexer_config import IndexerConfig


class IndexerPipeline:
    """Main indexing pipeline with FastEmbed optimization."""

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.pdf_processor = PDFProcessor(Path(config.data_directory))
        self.document_chunker = DocumentChunker(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_store_indexer = VectorStoreIndexer(config)

    def run(self) -> bool:
        """Run the complete indexing pipeline."""
        self.logger.info("Starting FastEmbed-based indexing pipeline")

        try:
            # Step 1: Process PDFs
            self.logger.info("Step 1: Processing PDFs")
            pdf_documents = self.pdf_processor.process_all_pdfs()
            if not pdf_documents:
                self.logger.error("No PDF documents processed")
                return False

            # Step 2: Chunk documents
            self.logger.info("Step 2: Chunking documents")
            chunks = self.document_chunker.chunk_multiple_documents(pdf_documents)
            if not chunks:
                self.logger.error("No chunks created")
                return False

            # Step 3: Generate embeddings
            self.logger.info("Step 3: Generating embeddings with FastEmbed")
            embedding_results = self.embedding_generator.generate_all_embeddings(chunks)

            # Step 4: Store embeddings in Qdrant
            self.logger.info("Step 4: Storing embeddings in Qdrant")
            success = self.vector_store_indexer.store_embeddings(embedding_results)
            if not success:
                self.logger.error("Embedding storage failed")
                return False

            # Get final statistics
            stats = self.vector_store_indexer.get_collection_stats()
            self.logger.info(f"Indexing completed successfully. Stats: {stats}")
            return True

        except Exception as e:
            self.logger.error(f"Indexing pipeline failed: {e}")
            return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config and run pipeline
    config = IndexerConfig()
    pipeline = IndexerPipeline(config)
    pipeline.run()
