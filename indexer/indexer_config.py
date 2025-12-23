from typing import Mapping, Any
from dataclasses import dataclass
from common.base_config import BaseConfig


class IndexerConfig(BaseConfig):
    """Configuration for the indexing pipeline."""

    _FILENAME = "indexer_config.json"

    # required keys → default (None means “no sensible default, must be present”)
    _REQUIRED: Mapping[str, Any] = {
        #--- data source -------------------------------------------------
        "data_directory": "domaindata",          # e.g. "./data/pdfs"
        #--- chunking ----------------------------------------------------
        "chunk_size": 500,                # max tokens per chunk (token‑based split)
        "chunk_overlap": 50,               # tokens that overlap between consecutive chunks
        "logical_sectioning": True,      # try to split on headings before token split
        #--- embedding models --------------------------------------------
        "dense_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sparse_embedding_model": "Qdrant/bm25",
        "rerank_embedding_model": "colbert-ir/colbertv2.0",
        #--- retrieval parameters ----------------------------------------
        "top_dense": 10,
        "top_sparse": 10,
        "top_rerank": 5,
        # --- Qdrant settings --------------------------------------------
        "qdrant_host": "localhost",
        "qdrant_http_port": 6333,
        "qdrant_grpc_port": 6334,
        "collection_name": "simple_kb",
    }


@dataclass
class PDFProcessorConfig:
    """Configuration for Docling PDF processing."""

    def __init__(self,
                 use_ocr: bool = True,
                 ocr_language: str = 'eng',
                 table_extraction: bool = True,
                 include_images: bool = False):  # Images as metadata only
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.table_extraction = table_extraction
        self.include_images = include_images
