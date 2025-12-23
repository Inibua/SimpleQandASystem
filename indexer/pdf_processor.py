"""
pdf_processor.py

Simplified PDF processor using Docling's markdown export.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from indexer.indexer_config import PDFProcessorConfig
from indexer.pdf_models import PDFDocument, DocumentElement, ContentType
from paths import ROOT_DIR


class PDFProcessor:
    """
    PDF processor using Docling's markdown export for clean, structured content.
    """

    def __init__(self, data_directory: Path, config: Optional[PDFProcessorConfig] = None):
        self.data_directory = Path(ROOT_DIR, data_directory)
        self.config = config or PDFProcessorConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize Docling
        self._init_docling()

        # Validate data directory
        if not self.data_directory.exists():
            raise ValueError(f"Data directory does not exist: {self.data_directory}")

    def _init_docling(self) -> None:
        """Initialize Docling library."""
        try:
            from docling.document_converter import DocumentConverter
            self.DocumentConverter = DocumentConverter
            self.docling_available = True
            self.logger.info("Docling library initialized successfully")
        except ImportError as e:
            self.logger.error("Docling not available. Please install: pip install docling")
            raise RuntimeError("Docling library not available") from e

    def discover_pdfs(self) -> List[Path]:
        """Discover all PDF files in the data directory."""
        pdf_files = list(self.data_directory.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {self.data_directory}")
        return pdf_files

    def process_pdf(self, pdf_path: Path) -> Optional[PDFDocument]:
        """Process a single PDF file using Docling's markdown export."""
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None

        self.logger.info(f"Processing PDF with Docling markdown export: {pdf_path.name}")

        try:
            # Initialize Docling converter
            converter = self.DocumentConverter()

            # Convert PDF to document object and export to markdown
            doc_result = converter.convert(str(pdf_path))
            markdown_content = doc_result.document.export_to_markdown()

            # Get page count from the document
            total_pages = len(doc_result.document.pages)

            # Create our document structure
            pdf_doc = PDFDocument(
                file_path=pdf_path,
                file_name=pdf_path.name,
                total_pages=total_pages,
                markdown_content=markdown_content,
                docling_document=doc_result.document,
                metadata={
                    "processing_library": "docling",
                    "file_size": pdf_path.stat().st_size,
                    "page_count": total_pages,
                    "markdown_length": len(markdown_content)
                }
            )

            # Parse markdown into structured elements
            self._parse_markdown_to_elements(markdown_content, pdf_doc)

            self.logger.info(f"Processed {pdf_path.name}: {len(pdf_doc.elements)} elements from markdown")
            return pdf_doc

        except Exception as e:
            self.logger.error(f"Docling markdown processing failed for {pdf_path}: {e}")
            return None

    def _parse_markdown_to_elements(self, markdown_content: str, pdf_doc: PDFDocument) -> None:
        """Parse markdown content into structured elements."""
        lines = markdown_content.split('\n')
        current_page = 1  # Start with page 1
        element_counter = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Detect page breaks (Docling might include page markers)
            if self._is_page_break(line):
                current_page += 1
                i += 1
                continue

            # Parse based on markdown syntax
            element = None

            # Headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2)
                element = DocumentElement(
                    element_id=f"heading_{element_counter}",
                    content_type=ContentType.SECTION_HEADING,
                    text=text,
                    page_number=current_page,
                    heading_level=level
                )

            # Lists
            elif re.match(r'^[-*]\s+.+', line):
                text = line[2:].strip()  # Remove list marker
                element = DocumentElement(
                    element_id=f"list_{element_counter}",
                    content_type=ContentType.LIST_ITEM,
                    text=text,
                    page_number=current_page
                )

            # Tables (simplified detection - markdown tables have | characters)
            elif '|' in line and self._looks_like_table_line(line, lines, i):
                table_text = self._extract_table(lines, i)
                if table_text:
                    element = DocumentElement(
                        element_id=f"table_{element_counter}",
                        content_type=ContentType.TABLE,
                        text=table_text,
                        page_number=current_page
                    )
                    # Skip table rows we've processed
                    i += table_text.count('\n')

            # Code blocks
            elif line.startswith('```'):
                code_text = self._extract_code_block(lines, i)
                if code_text:
                    element = DocumentElement(
                        element_id=f"code_{element_counter}",
                        content_type=ContentType.CODE_BLOCK,
                        text=code_text,
                        page_number=current_page
                    )
                    i += code_text.count('\n') + 2  # Skip code block markers

            # Regular paragraphs
            else:
                # Group consecutive non-empty lines as a paragraph
                paragraph_lines = []
                while i < len(lines) and lines[i].strip() and not self._is_special_line(lines[i]):
                    paragraph_lines.append(lines[i].strip())
                    i += 1

                if paragraph_lines:
                    text = ' '.join(paragraph_lines)
                    element = DocumentElement(
                        element_id=f"para_{element_counter}",
                        content_type=ContentType.PARAGRAPH,
                        text=text,
                        page_number=current_page
                    )
                    i -= 1  # Adjust for the outer loop increment

            if element:
                pdf_doc.add_element(element)
                element_counter += 1

            i += 1

    def _is_page_break(self, line: str) -> bool:
        """Detect page break markers in markdown."""
        return bool(re.match(r'^-+ Page \d+ -+$', line)) or 'page break' in line.lower()

    def _is_special_line(self, line: str) -> bool:
        """Check if a line is a special markdown element."""
        line = line.strip()
        return (
            re.match(r'^#{1,6}\s+.+', line) or  # Headings
            re.match(r'^[-*]\s+.+', line) or    # Lists
            '|' in line or                      # Potential table
            line.startswith('```')              # Code blocks
        )

    def _looks_like_table_line(self, line: str, lines: List[str], index: int) -> bool:
        """Check if a line is part of a markdown table."""
        # A table should have multiple rows with | characters
        if index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            # Check for table separator line (---|---)
            if re.match(r'^[\s|:-]+$', next_line):
                return True
        return line.count('|') >= 2  # At least two | characters suggests a table

    def _extract_table(self, lines: List[str], start_index: int) -> Optional[str]:
        """Extract a complete markdown table."""
        table_lines = []
        i = start_index

        # Add header row
        table_lines.append(lines[i].strip())
        i += 1

        # Add separator row if present
        if i < len(lines) and re.match(r'^[\s|:-]+$', lines[i].strip()):
            table_lines.append(lines[i].strip())
            i += 1

        # Add data rows until we hit a non-table line
        while i < len(lines) and '|' in lines[i] and not self._is_special_line(lines[i]):
            table_lines.append(lines[i].strip())
            i += 1

        return '\n'.join(table_lines) if table_lines else None

    def _extract_code_block(self, lines: List[str], start_index: int) -> Optional[str]:
        """Extract a code block."""
        if not lines[start_index].startswith('```'):
            return None

        code_lines = []
        i = start_index + 1

        while i < len(lines) and not lines[i].startswith('```'):
            code_lines.append(lines[i])
            i += 1

        return '\n'.join(code_lines) if code_lines else None

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
