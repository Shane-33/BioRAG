"""BioEmbedding component for BioRAG.

This embedding component wraps biotech-specific embedding models like
BioBERT, SciBERT, or PubMedBERT from HuggingFace.
"""

import logging
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)


class BioEmbedding(BaseEmbedding):
    """BioEmbedding using biotech-specific models from HuggingFace.

    Supports models like:
    - dmis-lab/biobert-v1.1
    - allenai/scibert_scivocab_uncased
    - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    """

    model_name: str = Field(
        default="dmis-lab/biobert-v1.1",
        description="Name of the HuggingFace model to use for bio embeddings",
    )
    cache_folder: str | None = Field(
        default=None, description="Cache folder for the model"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="If set to True, the code from the remote model will be trusted and executed.",
    )
    _embedding_model: Any = PrivateAttr(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "BioEmbedding"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the HuggingFace embedding model."""
        try:
            from llama_index.embeddings.huggingface import (  # type: ignore
                HuggingFaceEmbedding,
            )
        except ImportError as e:
            raise ImportError(
                "Local dependencies not found, install with `poetry install --extras embeddings-huggingface`"
            ) from e

        logger.info(
            "Initializing BioEmbedding with model_name=%s", self.model_name
        )
        self._embedding_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            cache_folder=self.cache_folder,
            trust_remote_code=self.trust_remote_code,
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._embedding_model._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._embedding_model._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings for multiple texts."""
        return self._embedding_model._get_text_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get query embedding asynchronously."""
        return await self._embedding_model._aget_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get text embedding asynchronously."""
        return await self._embedding_model._aget_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings asynchronously."""
        return await self._embedding_model._aget_text_embeddings(texts)

    def __call__(self, nodes: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the embedding model on nodes (used by LlamaIndex transformations)."""
        # Delegate to the wrapped model's __call__ method which properly handles nodes
        return self._embedding_model(nodes, *args, **kwargs)
