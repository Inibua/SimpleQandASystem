"""
pdf_models.py

Data models for markdown-based PDF processing and chunking.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any


class ContentType(Enum):
    """Types of content extracted from markdown."""
    SECTION_HEADING = "section_heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE_BLOCK = "code_block"


@dataclass
class DocumentElement:
    """Represents a single element from markdown content."""
    element_id: str
    content_type: ContentType
    text: str
    page_number: int
    heading_level: int = 0  # For headings: 1 for #, 2 for ##, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Represents a chunk of content for indexing."""
    chunk_id: str
    chunk_text: str
    document_name: str
    page_number: int
    section_heading: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "metadata": self.metadata
        }


@dataclass
class PDFDocument:
    """Represents a processed PDF file with markdown elements."""
    file_path: Path
    file_name: str
    total_pages: int
    elements: List[DocumentElement] = field(default_factory=list)
    markdown_content: str = ""  # Store full markdown for reference
    docling_document: Any = None  # Store the original Docling document for chunking
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_element(self, element: DocumentElement) -> None:
        """Add an element to the document."""
        self.elements.append(element)

    def get_text_elements(self) -> List[DocumentElement]:
        """Get all text-related elements."""
        return [elem for elem in self.elements if elem.content_type != ContentType.TABLE]
