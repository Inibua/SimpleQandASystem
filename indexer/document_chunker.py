"""
document_chunker.py

Document chunking implementation using Docling's HybridChunker.
"""

import logging
from typing import List
from pathlib import Path

from indexer.pdf_models import PDFDocument, DocumentChunk
from indexer.indexer_config import IndexerConfig


class DocumentChunker:
    """
    Document chunker using Docling's HybridChunker for logical-token hybrid chunking.
    """

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._init_docling_chunker()

    def _init_docling_chunker(self) -> None:
        """Initialize Docling's HybridChunker."""
        try:
            from docling.chunking import HybridChunker
            self.HybridChunker = HybridChunker
            self.docling_chunker_available = True
            self.logger.info("Docling HybridChunker initialized successfully")
        except ImportError as e:
            self.logger.error("Docling chunking not available. Please install: pip install 'docling[chunking]'")
            raise RuntimeError("Docling chunking library not available") from e

    def chunk_document(self, pdf_document: PDFDocument) -> List[DocumentChunk]:
        """
        Chunk a PDF document using Docling's HybridChunker.

        Args:
            pdf_document: The processed PDF document

        Returns:
            List of DocumentChunk objects ready for indexing
        """
        if not pdf_document.docling_document:
            self.logger.error(f"No Docling document available for chunking: {pdf_document.file_name}")
            return []

        self.logger.info(f"Chunking document: {pdf_document.file_name}")

        try:
            # Initialize HybridChunker with configuration
            chunker = self.HybridChunker()

            # Chunk the document using Docling's native chunker
            docling_chunks = list(chunker.chunk(pdf_document.docling_document))

            # Convert Docling chunks to our DocumentChunk model
            document_chunks = self._convert_docling_chunks(docling_chunks, pdf_document, chunker)

            self.logger.info(f"Created {len(document_chunks)} chunks for {pdf_document.file_name}")
            return document_chunks

        except Exception as e:
            self.logger.error(f"Chunking failed for {pdf_document.file_name}: {e}")
            return []

    def _convert_docling_chunks(self, docling_chunks: List, pdf_document: PDFDocument, chunker) -> List[DocumentChunk]:
        """
        Convert Docling chunks to our internal DocumentChunk format.

        Args:
            docling_chunks: List of chunks from Docling's HybridChunker
            pdf_document: The source PDF document
            chunker: The Docling chunker instance

        Returns:
            List of DocumentChunk objects
        """
        document_chunks = []

        for i, docling_chunk in enumerate(docling_chunks):
            try:
                # Get contextualized text (metadata-enriched)
                chunk_text = chunker.contextualize(docling_chunk)

                # Extract metadata from Docling chunk
                chunk_metadata = self._extract_chunk_metadata(docling_chunk)

                # Determine section heading
                section_heading = self._get_section_heading(chunk_metadata)

                # Determine page number
                page_number = self._get_page_number(chunk_metadata, pdf_document)

                # Create our DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=f"{Path(pdf_document.file_name).stem}_{i}",
                    chunk_text=chunk_text,
                    document_name=pdf_document.file_name,
                    page_number=page_number,
                    section_heading=section_heading,
                    metadata=chunk_metadata
                )

                document_chunks.append(chunk)

            except Exception as e:
                self.logger.warning(f"Failed to convert chunk {i} for {pdf_document.file_name}: {e}")
                continue

        return document_chunks

    def _extract_chunk_metadata(self, docling_chunk) -> dict:
        """Extract metadata from a Docling chunk."""
        metadata = {}

        try:
            # Docling chunks typically have metadata attributes
            if hasattr(docling_chunk, 'metadata'):
                metadata.update(getattr(docling_chunk, 'metadata', {}))

            # Additional metadata extraction can be added here
            if hasattr(docling_chunk, 'heading_level'):
                metadata['heading_level'] = getattr(docling_chunk, 'heading_level', 0)

        except Exception as e:
            self.logger.debug(f"Metadata extraction failed: {e}")

        return metadata

    def _get_section_heading(self, chunk_metadata: dict) -> str:
        """Extract section heading from chunk metadata."""
        # Try various possible metadata keys for section heading
        heading_keys = ['section_heading', 'heading', 'title', 'header']

        for key in heading_keys:
            if key in chunk_metadata and chunk_metadata[key]:
                return str(chunk_metadata[key])

        return ""

    def _get_page_number(self, chunk_metadata: dict) -> int:
        """Extract page number from chunk metadata, fallback to document-level info."""
        # Try to get page number from metadata
        page_keys = ['page_number', 'page', 'page_no']

        for key in page_keys:
            if key in chunk_metadata and chunk_metadata[key] is not None:
                try:
                    return int(chunk_metadata[key])
                except (ValueError, TypeError):
                    continue

        # Fallback: use first page or document-level information
        return 1

    def chunk_multiple_documents(self, pdf_documents: List[PDFDocument]) -> List[DocumentChunk]:
        """
        Chunk multiple PDF documents.

        Args:
            pdf_documents: List of processed PDF documents

        Returns:
            Combined list of chunks from all documents
        """
        all_chunks = []

        for doc in pdf_documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
