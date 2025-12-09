# BioRAG Overview

BioRAG is a biotech-focused extension to PrivateGPT that provides specialized RAG (Retrieval-Augmented Generation) capabilities for biotech and pharmaceutical documents.

## Features

### 1. Domain-Specific Ingestion Processors

BioRAG extends the existing ingestion pipeline with specialized processors for different types of biotech documents:

- **Clinical Ingestor** (`ClinicalIngestor`): Extracts metadata from clinical trial documents including:
  - Eligibility criteria
  - Dosing schemas
  - Safety notes and adverse events

- **MOA Ingestor** (`MOAIngestor`): Extracts mechanism of action information including:
  - Mechanism of action descriptions
  - Protein targets

- **Protein Ingestor** (`ProteinIngestor`): Extracts protein target information including:
  - Protein target names
  - Mechanism of action

- **Regulatory Ingestor** (`RegulatoryIngestor`): Extracts regulatory information including:
  - FDA/EMA approval information
  - Regulatory sections

### 2. Biotech Metadata Extractors

The `BioMetadataExtractor` class provides pattern-based extraction of biotech-specific metadata:

- `eligibility_criteria`: Patient eligibility requirements
- `dose_schema`: Dosing information and schedules
- `safety_notes`: Safety information and warnings
- `protein_target`: Protein target names and identifiers
- `mechanism_of_action`: Drug mechanism descriptions
- `regulatory_section`: Regulatory approval and indication information

### 3. BioEmbedding Component

BioRAG includes a specialized embedding component (`BioEmbedding`) that wraps biotech-specific embedding models:

- **BioBERT** (`dmis-lab/biobert-v1.1`)
- **SciBERT** (`allenai/scibert_scivocab_uncased`)
- **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)

To use BioEmbedding, set the embedding mode to `bio` in your `settings.yaml`:

```yaml
embedding:
  mode: bio
huggingface:
  embedding_hf_model_name: "dmis-lab/biobert-v1.1"
```

### 4. Domain-Specific Prompt Templates

BioRAG provides specialized prompt templates for each domain:

- **Clinical Prompt Template**: Optimized for clinical trial queries
- **MOA Prompt Template**: Optimized for mechanism of action queries
- **Protein Prompt Template**: Optimized for protein target queries
- **Regulatory Prompt Template**: Optimized for regulatory queries

### 5. Biotech Modes in API

The chat API now supports a `mode` parameter to select the appropriate biotech domain:

```python
POST /v1/chat/completions
{
  "messages": [...],
  "use_context": true,
  "mode": "clinical"  # Options: "clinical", "moa", "protein", "regulatory"
}
```

### 6. Gradio UI Extensions

The Gradio UI has been extended with:

- **Biotech Mode Dropdown**: Select from Clinical, MOA, Protein, or Regulatory modes
- **Metadata Display**: View extracted metadata from ingested documents

## Architecture

BioRAG follows the PrivateGPT extension pattern:

- **Extends, doesn't replace**: All BioRAG components extend existing PrivateGPT components
- **Backward compatible**: Existing functionality remains unchanged
- **Modular design**: Each component can be used independently

### Component Structure

```
private_gpt/
├── components/
│   ├── ingest/
│   │   ├── bio_metadata_extractors.py  # Metadata extraction
│   │   ├── clinical_ingestor.py        # Clinical document processor
│   │   ├── moa_ingestor.py              # MOA document processor
│   │   ├── protein_ingestor.py          # Protein document processor
│   │   └── regulatory_ingestor.py      # Regulatory document processor
│   ├── embedding/
│   │   └── bio_embedding.py            # BioEmbedding component
│   └── prompts/
│       ├── clinical_prompt.py          # Clinical prompt template
│       ├── moa_prompt.py                # MOA prompt template
│       ├── protein_prompt.py            # Protein prompt template
│       └── regulatory_prompt.py         # Regulatory prompt template
└── server/
    └── chat/
        ├── chat_service.py              # Extended with mode support
        └── chat_router.py               # Extended with mode parameter
```

## Usage

### Configuration

1. **Set embedding mode** in `settings.yaml`:
```yaml
embedding:
  mode: bio
huggingface:
  embedding_hf_model_name: "dmis-lab/biobert-v1.1"
```

2. **Use biotech modes** in API calls:
```python
# Clinical mode
response = chat_service.chat(
    messages=messages,
    use_context=True,
    mode="clinical"
)
```

### Ingestion

The ingestion process automatically extracts biotech metadata when using the specialized ingestors. Metadata is stored in the document's metadata dictionary and can be used for filtering and retrieval.

### Querying

When querying with a biotech mode, the system:
1. Uses the appropriate prompt template for the domain
2. Applies domain-specific reasoning instructions
3. Retrieves context using the BioEmbedding model (if configured)
4. Returns answers optimized for the selected domain

## Examples

See `docs/biotech_examples.md` for detailed examples of using BioRAG with different document types and query modes.

