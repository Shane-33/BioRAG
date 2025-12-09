"""Biotech-specific prompt templates for BioRAG."""

from private_gpt.components.prompts.clinical_prompt import ClinicalPromptTemplate
from private_gpt.components.prompts.moa_prompt import MOAPromptTemplate
from private_gpt.components.prompts.protein_prompt import ProteinPromptTemplate
from private_gpt.components.prompts.regulatory_prompt import RegulatoryPromptTemplate

__all__ = [
    "ClinicalPromptTemplate",
    "MOAPromptTemplate",
    "ProteinPromptTemplate",
    "RegulatoryPromptTemplate",
]

