"""Mechanism of Action (MOA) document ingestor for BioRAG.

This ingestor extends the base ingestion component to add MOA-specific
metadata extraction for mechanism of action information.
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


class MOAIngestor(SimpleIngestComponent):
    """Ingestor for mechanism of action documents with metadata extraction."""

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
        """Preprocess documents to extract MOA metadata."""
        for document in documents:
            # Extract MOA-specific fields
            moa = self.metadata_extractor.extract_mechanism_of_action(document.text)
            protein = self.metadata_extractor.extract_protein_target(document.text)

            # Add to document metadata
            if moa:
                document.metadata["mechanism_of_action"] = moa
            if protein:
                document.metadata["protein_target"] = protein

            # Mark as MOA document
            document.metadata["document_type"] = "moa"

        return documents

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        """Ingest a MOA document with metadata extraction."""
        logger.info("Ingesting MOA document: file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        documents = self._preprocess_documents(documents)
        logger.info(
            "Transformed file=%s into count=%s documents with MOA metadata",
            file_name,
            len(documents),
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        """Bulk ingest MOA documents."""
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            documents = self._preprocess_documents(documents)
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

