"""
pdf_models.py

Data models for Docling PDF processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class ElementCategory(Enum):
    """Categories from Docling element types."""
    TITLE = "Title"
    NARRATIVE_TEXT = "NarrativeText"
    SECTION_HEADER = "SectionHeader"
    LIST_ITEM = "ListItem"
    TABLE = "Table"
    IMAGE = "Image"
    UNCATEGORIZED_TEXT = "UncategorizedText"


@dataclass
class DocumentElement:
    """Represents a single element from Docling output."""
    element_id: str
    element_type: ElementCategory
    text: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For tables
    table_data: Optional[List[List[str]]] = None


@dataclass
class PDFDocument:
    """Represents a processed PDF file with structured elements."""
    file_path: Path
    file_name: str
    total_pages: int
    elements: List[DocumentElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_element(self, element: DocumentElement) -> None:
        """Add an element to the document."""
        self.elements.append(element)

    def get_text_elements(self) -> List[DocumentElement]:
        """Get all text-related elements."""
        text_types = [
            ElementCategory.NARRATIVE_TEXT,
            ElementCategory.TITLE,
            ElementCategory.SECTION_HEADER,
            ElementCategory.LIST_ITEM,
            ElementCategory.UNCATEGORIZED_TEXT
        ]
        return [elem for elem in self.elements if elem.element_type in text_types]

    def get_table_elements(self) -> List[DocumentElement]:
        """Get all table elements."""
        return [elem for elem in self.elements if elem.element_type == ElementCategory.TABLE]
