from typing import Mapping, Any

from common.base_config import BaseConfig


class IndexerConfig(BaseConfig):
    """Configuration for the indexing pipeline."""

    _FILENAME = "indexer_config.json"

    # required keys → default (None means “no sensible default, must be present”)
    _REQUIRED: Mapping[str, Any] = {
        #--- data source -------------------------------------------------
        "data_directory": "./domaindata",          # e.g. "./data/pdfs"
        #--- chunking ----------------------------------------------------
        "chunk_size": 500,                # max tokens per chunk (token‑based split)
        "chunk_overlap": 50,               # tokens that overlap between consecutive chunks
        "logical_sectioning": True,      # try to split on headings before token split
        #--- embedding models --------------------------------------------
        "dense_embedding_model": "all-MiniLM-L6-v2",
        "sparse_embedding_model": "bm25",
        "rerank_embedding_model": "colbert2.0",
        #--- retrieval parameters ----------------------------------------
        "top_dense": 10,
        "top_sparse": 10,
        "top_rerank": 5,
    }