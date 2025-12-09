# BioRAG Examples

This document provides practical examples of using BioRAG for different biotech use cases.

## Example 1: Clinical Trial Document Ingestion

### Ingesting a Clinical Trial Protocol

```python
from private_gpt.components.ingest.clinical_ingestor import ClinicalIngestor
from pathlib import Path

# Initialize the clinical ingestor
ingestor = ClinicalIngestor(
    storage_context=storage_context,
    embed_model=embedding_model,
    transformations=transformations
)

# Ingest a clinical trial protocol PDF
documents = ingestor.ingest(
    file_name="clinical_trial_protocol.pdf",
    file_data=Path("data/clinical_trial_protocol.pdf")
)

# The documents now have extracted metadata:
# - eligibility_criteria: Patient eligibility requirements
# - dose_schema: Dosing information
# - safety_notes: Safety information
```

### Querying Clinical Documents

```python
# Using the API
POST /v1/chat/completions
{
  "messages": [
    {
      "role": "user",
      "content": "What are the eligibility criteria for this trial?"
    }
  ],
  "use_context": true,
  "mode": "clinical"
}

# Using the Python service
from private_gpt.server.chat.chat_service import ChatService

completion = chat_service.chat(
    messages=[ChatMessage(content="What are the eligibility criteria?", role=MessageRole.USER)],
    use_context=True,
    mode="clinical"
)
```

## Example 2: Mechanism of Action (MOA) Documents

### Ingesting MOA Documents

```python
from private_gpt.components.ingest.moa_ingestor import MOAIngestor

ingestor = MOAIngestor(
    storage_context=storage_context,
    embed_model=embedding_model,
    transformations=transformations
)

# Ingest a drug mechanism document
documents = ingestor.ingest(
    file_name="drug_moa.pdf",
    file_data=Path("data/drug_moa.pdf")
)

# Extracted metadata:
# - mechanism_of_action: Description of how the drug works
# - protein_target: Target protein information
```

### Querying MOA Documents

```python
# Query about mechanism of action
POST /v1/chat/completions
{
  "messages": [
    {
      "role": "user",
      "content": "How does this drug inhibit its target protein?"
    }
  ],
  "use_context": true,
  "mode": "moa"
}
```

## Example 3: Protein Target Documents

### Ingesting Protein Target Documents

```python
from private_gpt.components.ingest.protein_ingestor import ProteinIngestor

ingestor = ProteinIngestor(
    storage_context=storage_context,
    embed_model=embedding_model,
    transformations=transformations
)

# Ingest a protein target document
documents = ingestor.ingest(
    file_name="protein_targets.pdf",
    file_data=Path("data/protein_targets.pdf")
)

# Extracted metadata:
# - protein_target: Protein names and identifiers
# - mechanism_of_action: How drugs interact with proteins
```

### Querying Protein Documents

```python
# Query about protein targets
POST /v1/chat/completions
{
  "messages": [
    {
      "role": "user",
      "content": "What proteins does this drug target?"
    }
  ],
  "use_context": true,
  "mode": "protein"
}
```

## Example 4: Regulatory Documents

### Ingesting Regulatory Documents

```python
from private_gpt.components.ingest.regulatory_ingestor import RegulatoryIngestor

ingestor = RegulatoryIngestor(
    storage_context=storage_context,
    embed_model=embedding_model,
    transformations=transformations
)

# Ingest a regulatory document
documents = ingestor.ingest(
    file_name="fda_approval.pdf",
    file_data=Path("data/fda_approval.pdf")
)

# Extracted metadata:
# - regulatory_section: FDA/EMA approval information
```

### Querying Regulatory Documents

```python
# Query about regulatory information
POST /v1/chat/completions
{
  "messages": [
    {
      "role": "user",
      "content": "What are the approved indications for this drug?"
    }
  ],
  "use_context": true,
  "mode": "regulatory"
}
```

## Example 5: Using BioEmbedding

### Configuration

```yaml
# settings.yaml
embedding:
  mode: bio
huggingface:
  embedding_hf_model_name: "dmis-lab/biobert-v1.1"
  trust_remote_code: false
```

### Using BioEmbedding in Code

```python
from private_gpt.components.embedding.bio_embedding import BioEmbedding

# Initialize BioEmbedding
bio_embedding = BioEmbedding(
    model_name="dmis-lab/biobert-v1.1",
    cache_folder="./models",
    trust_remote_code=False
)

# Get embeddings
query_embedding = bio_embedding.get_query_embedding("What is the mechanism of action?")
text_embeddings = bio_embedding.get_text_embeddings([
    "This drug inhibits EGFR",
    "The protein target is HER2"
])
```

## Example 6: Using the Gradio UI

1. **Start the server** with UI enabled:
```bash
python -m private_gpt.main
```

2. **Select Biotech Mode** from the dropdown:
   - Clinical: For clinical trial queries
   - MOA: For mechanism of action queries
   - Protein: For protein target queries
   - Regulatory: For regulatory queries

3. **Upload biotech documents** using the upload button

4. **Query** using the selected mode - the system will automatically:
   - Use the appropriate prompt template
   - Apply domain-specific reasoning
   - Display extracted metadata

## Example 7: Metadata Extraction

### Extracting Metadata Manually

```python
from private_gpt.components.ingest.bio_metadata_extractors import BioMetadataExtractor
from llama_index.core.schema import Document

extractor = BioMetadataExtractor()

# Create a document
doc = Document(text="""
This clinical trial evaluates Drug X at a dose of 100mg daily.
Eligibility criteria: Patients aged 18-65 with confirmed diagnosis.
Adverse events include nausea and fatigue.
""")

# Extract all metadata
metadata = extractor.extract_all_metadata(doc)

# Access specific fields
eligibility = extractor.extract_eligibility_criteria(doc.text)
dose = extractor.extract_dose_schema(doc.text)
safety = extractor.extract_safety_notes(doc.text)
```

## Example 8: Combining Multiple Modes

You can use different modes for different queries on the same document set:

```python
# Query about clinical aspects
clinical_response = chat_service.chat(
    messages=[ChatMessage(content="What are the dosing requirements?", role=MessageRole.USER)],
    use_context=True,
    mode="clinical"
)

# Query about mechanism
moa_response = chat_service.chat(
    messages=[ChatMessage(content="How does this drug work?", role=MessageRole.USER)],
    use_context=True,
    mode="moa"
)

# Query about regulatory status
regulatory_response = chat_service.chat(
    messages=[ChatMessage(content="Is this drug FDA approved?", role=MessageRole.USER)],
    use_context=True,
    mode="regulatory"
)
```

## Best Practices

1. **Use appropriate ingestors**: Match the ingestor to your document type for better metadata extraction
2. **Select the right mode**: Use the mode that matches your query domain
3. **Configure BioEmbedding**: Use BioEmbedding for better semantic understanding of biotech text
4. **Combine modes**: Use different modes for different aspects of the same document set
5. **Review metadata**: Check extracted metadata to ensure it's being captured correctly

## Troubleshooting

### Metadata Not Extracted

If metadata is not being extracted:
- Check that you're using the appropriate ingestor for your document type
- Verify that the document text contains the expected patterns
- Review the regex patterns in `BioMetadataExtractor` if needed

### Embedding Issues

If BioEmbedding fails:
- Ensure `embeddings-huggingface` extra is installed: `poetry install --extras embeddings-huggingface`
- Check that the model name is correct
- Verify internet connection for model download

### Mode Not Working

If biotech modes don't work:
- Verify that the mode parameter is being passed correctly
- Check that the prompt templates are imported correctly
- Ensure the chat service has been updated with mode support

