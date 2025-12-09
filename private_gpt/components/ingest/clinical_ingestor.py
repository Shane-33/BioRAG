"""Clinical trial document ingestor for BioRAG.

This ingestor extends the base ingestion component to add clinical-specific
metadata extraction for eligibility criteria, dosing schemas, and safety notes.
"""

import logging
from pathlib import Path
from typing import Any

from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.bio_metadata_extractors import BioMetadataExtractor
from private_gpt.components.ingest.ingest_component import (
    BaseIngestComponentWithIndex,
    SimpleIngestComponent,
)
from private_gpt.components.ingest.ingest_helper import IngestionHelper

logger = logging.getLogger(__name__)


class ClinicalIngestor(SimpleIngestComponent):
    """Ingestor for clinical trial documents with metadata extraction."""

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
        """Preprocess documents to extract clinical metadata."""
        for document in documents:
            metadata = self.metadata_extractor.extract_all_metadata(document)
            # Extract clinical-specific fields
            eligibility = self.metadata_extractor.extract_eligibility_criteria(
                document.text
            )
            dose = self.metadata_extractor.extract_dose_schema(document.text)
            safety = self.metadata_extractor.extract_safety_notes(document.text)

            # Add to document metadata
            if eligibility:
                document.metadata["eligibility_criteria"] = eligibility
            if dose:
                document.metadata["dose_schema"] = dose
            if safety:
                document.metadata["safety_notes"] = safety

            # Mark as clinical document
            document.metadata["document_type"] = "clinical"

        return documents

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        """Ingest a clinical document with metadata extraction."""
        logger.info("Ingesting clinical document: file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        documents = self._preprocess_documents(documents)
        logger.info(
            "Transformed file=%s into count=%s documents with clinical metadata",
            file_name,
            len(documents),
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        """Bulk ingest clinical documents."""
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            documents = self._preprocess_documents(documents)
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

