"""Protein target document ingestor for BioRAG.

This ingestor extends the base ingestion component to add protein-specific
metadata extraction for protein targets and related information.
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


class ProteinIngestor(SimpleIngestComponent):
    """Ingestor for protein target documents with metadata extraction."""

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
        """Preprocess documents to extract protein metadata."""
        for document in documents:
            # Extract protein-specific fields
            protein = self.metadata_extractor.extract_protein_target(document.text)
            moa = self.metadata_extractor.extract_mechanism_of_action(document.text)

            # Add to document metadata
            if protein:
                document.metadata["protein_target"] = protein
            if moa:
                document.metadata["mechanism_of_action"] = moa

            # Mark as protein document
            document.metadata["document_type"] = "protein"

        return documents

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        """Ingest a protein document with metadata extraction."""
        logger.info("Ingesting protein document: file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        documents = self._preprocess_documents(documents)
        logger.info(
            "Transformed file=%s into count=%s documents with protein metadata",
            file_name,
            len(documents),
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        """Bulk ingest protein documents."""
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            documents = self._preprocess_documents(documents)
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

