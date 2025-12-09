"""Regulatory document ingestor for BioRAG.

This ingestor extends the base ingestion component to add regulatory-specific
metadata extraction for FDA/EMA approvals, indications, and regulatory sections.
"""

import logging
from pathlib import Path
from typing import Any

from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.bio_metadata_extractors import BioMetadataExtractor
from private_gpt.components.ingest.ingest_component import SimpleIngestComponent
from private_gpt.components.ingest.ingest_helper import IngestionHelper

logger = logging.getLogger(__name__)


class RegulatoryIngestor(SimpleIngestComponent):
    """Ingestor for regulatory documents with metadata extraction."""

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        self.metadata_extractor = BioMetadataExtractor()

    def _preprocess_documents(self, documents: list[Document]) -> list[Document]:
        """Preprocess documents to extract regulatory metadata."""
        for document in documents:
            # Extract regulatory-specific fields
            regulatory = self.metadata_extractor.extract_regulatory_section(
                document.text
            )

            # Add to document metadata
            if regulatory:
                document.metadata["regulatory_section"] = regulatory

            # Mark as regulatory document
            document.metadata["document_type"] = "regulatory"

        return documents

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        """Ingest a regulatory document with metadata extraction."""
        logger.info("Ingesting regulatory document: file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        documents = self._preprocess_documents(documents)
        logger.info(
            "Transformed file=%s into count=%s documents with regulatory metadata",
            file_name,
            len(documents),
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        """Bulk ingest regulatory documents."""
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            documents = self._preprocess_documents(documents)
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

