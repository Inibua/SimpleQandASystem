"""
pdf_processor.py

Docling-only PDF processor for superior PDF parsing.
"""

import logging
from pathlib import Path
from typing import List, Optional
from docling.document_converter import DocumentConverter
from indexer.indexer_config import PDFProcessorConfig
from indexer.pdf_models import PDFDocument, DocumentElement, ElementCategory


class PDFProcessor:
    """
    PDF processor using Docling for superior PDF parsing.
    """

    def __init__(self, data_directory: Path, config: Optional[PDFProcessorConfig] = None):
        self.data_directory = Path(data_directory)
        self.config = config or PDFProcessorConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize Docling
        self._init_docling()

        # Validate data directory
        if not self.data_directory.exists():
            raise ValueError(f"Data directory does not exist: {self.data_directory}")

    def _init_docling(self) -> None:
        """Initialize Docling library."""
        self.DocumentConverter = DocumentConverter
        self.docling_available = True
        self.logger.info("Docling library initialized successfully")

    def discover_pdfs(self) -> List[Path]:
        """Discover all PDF files in the data directory."""
        pdf_files = list(self.data_directory.glob("**/*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {self.data_directory}")
        return pdf_files

    def process_pdf(self, pdf_path: Path) -> Optional[PDFDocument]:
        """Process a single PDF file using Docling."""
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None

        self.logger.info(f"Processing PDF with Docling: {pdf_path.name}")

        try:
            # Initialize Docling converter with configuration
            converter = self.DocumentConverter(
                enable_table_structure=self.config.table_extraction,
                enable_image_caption=self.config.include_images,
            )

            # Convert PDF to document object
            doc_result = converter.convert(pdf_path)

            # Create our document structure
            pdf_doc = PDFDocument(
                file_path=pdf_path,
                file_name=pdf_path.name,
                total_pages=len(doc_result.pages) if hasattr(doc_result, 'pages') else 1,
                metadata={
                    "processing_library": "docling",
                    "file_size": pdf_path.stat().st_size
                }
            )

            # Extract elements from Docling document
            if hasattr(doc_result, 'blocks'):
                for block in doc_result.blocks:
                    element = self._convert_docling_block(block)
                    if element:
                        pdf_doc.add_element(element)

            self.logger.info(f"Docling processed {pdf_path.name}: {len(pdf_doc.elements)} elements")
            return pdf_doc

        except Exception as e:
            self.logger.error(f"Docling processing failed for {pdf_path}: {e}")
            return None

    def _convert_docling_block(self, block) -> Optional[DocumentElement]:
        """Convert Docling block to our DocumentElement format."""
        try:
            # Map Docling block types to our categories
            type_mapping = {
                'title': ElementCategory.TITLE,
                'heading': ElementCategory.SECTION_HEADER,
                'paragraph': ElementCategory.NARRATIVE_TEXT,
                'list_item': ElementCategory.LIST_ITEM,
                'table': ElementCategory.TABLE,
                'figure': ElementCategory.IMAGE,
            }

            block_type = getattr(block, 'type', 'unknown').lower()
            element_type = type_mapping.get(block_type, ElementCategory.UNCATEGORIZED_TEXT)

            # Extract text content
            text = getattr(block, 'text', '') or getattr(block, 'caption', '')
            if not text.strip():
                return None  # Skip empty elements

            # Extract page number
            page_num = getattr(block, 'page_number', 1)

            # Extract metadata
            metadata = {
                "source_type": block_type,
                "confidence": getattr(block, 'confidence', 1.0),
            }

            # Table data extraction
            table_data = None
            if element_type == ElementCategory.TABLE and hasattr(block, 'data'):
                table_data = getattr(block, 'data', [])

            element = DocumentElement(
                element_id=f"docling_{id(block)}",
                element_type=element_type,
                text=text.strip(),
                page_number=page_num,
                metadata=metadata,
                table_data=table_data
            )

            return element

        except Exception as e:
            self.logger.warning(f"Failed to convert Docling block: {e}")
            return None

    def process_all_pdfs(self) -> List[PDFDocument]:
        """Process all PDF files in the data directory."""
        pdf_files = self.discover_pdfs()
        processed_docs = []

        for pdf_path in pdf_files:
            document = self.process_pdf(pdf_path)
            if document:
                processed_docs.append(document)

        self.logger.info(f"Processed {len(processed_docs)} out of {len(pdf_files)} PDF files")
        return processed_docs
