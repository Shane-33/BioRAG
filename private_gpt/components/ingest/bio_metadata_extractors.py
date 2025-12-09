"""Biotech-specific metadata extractors for BioRAG.

This module provides metadata extraction functions for biotech documents,
including clinical trials, mechanisms of action, protein targets, and regulatory information.
"""

import logging
import re
from typing import Any

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class BioMetadataExtractor:
    """Base class for extracting biotech-specific metadata from documents."""

    @staticmethod
    def extract_eligibility_criteria(text: str) -> str | None:
        """Extract eligibility criteria from clinical trial documents."""
        # Look for common eligibility criteria patterns
        patterns = [
            r"(?:inclusion|eligibility)\s+criteria[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"patients?\s+(?:must|should|eligible)\s+(?:to|for|with)\s+(.*?)(?:\.|$)",
            r"age\s+(?:range|between|of)\s+(\d+)\s*(?:-|to)\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def extract_dose_schema(text: str) -> str | None:
        """Extract dosing schema from clinical documents."""
        patterns = [
            r"dose[s]?\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:mg|μg|g|IU|units?)(?:\s*(?:per|/)\s*(?:day|week|month|kg|m2))?",
            r"dosing\s+(?:regimen|schedule|schema)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"(\d+(?:\.\d+)?)\s*(?:mg|μg|g)\s*(?:once|twice|daily|weekly|monthly)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def extract_safety_notes(text: str) -> str | None:
        """Extract safety information from documents."""
        patterns = [
            r"(?:adverse\s+event|side\s+effect|safety|toxicity)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"contraindication[s]?[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"warning[s]?[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def extract_protein_target(text: str) -> str | None:
        """Extract protein target information."""
        # Look for common protein naming patterns
        patterns = [
            r"(?:target[s]?|inhibits?|binds?\s+to|acts?\s+on)[:\s]+([A-Z0-9]+(?:-[A-Z0-9]+)*)",
            r"([A-Z0-9]+(?:-[A-Z0-9]+)*)\s+(?:receptor|protein|enzyme|kinase|pathway)",
            r"(?:PD-1|PD-L1|EGFR|HER2|VEGF|ALK|BRAF|KRAS|PI3K|mTOR)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
        return None

    @staticmethod
    def extract_mechanism_of_action(text: str) -> str | None:
        """Extract mechanism of action information."""
        patterns = [
            r"mechanism\s+of\s+action[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"(?:MOA|mode\s+of\s+action)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"(?:inhibits?|blocks?|activates?|modulates?)\s+(.*?)(?:pathway|receptor|enzyme|protein)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
        return None

    @staticmethod
    def extract_regulatory_section(text: str) -> str | None:
        """Extract regulatory information."""
        patterns = [
            r"(?:FDA|EMA|regulatory|approval|indication)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"(?:indication[s]?|approved\s+for)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
            r"(?:phase\s+[I|II|III|IV]|clinical\s+trial)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
        return None

    @classmethod
    def extract_all_metadata(cls, document: Document) -> dict[str, Any]:
        """Extract all biotech metadata from a document."""
        text = document.text
        metadata: dict[str, Any] = {}

        # Extract all metadata fields
        eligibility = cls.extract_eligibility_criteria(text)
        if eligibility:
            metadata["eligibility_criteria"] = eligibility

        dose = cls.extract_dose_schema(text)
        if dose:
            metadata["dose_schema"] = dose

        safety = cls.extract_safety_notes(text)
        if safety:
            metadata["safety_notes"] = safety

        protein = cls.extract_protein_target(text)
        if protein:
            metadata["protein_target"] = protein

        moa = cls.extract_mechanism_of_action(text)
        if moa:
            metadata["mechanism_of_action"] = moa

        regulatory = cls.extract_regulatory_section(text)
        if regulatory:
            metadata["regulatory_section"] = regulatory

        return metadata

