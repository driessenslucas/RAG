# Long RAG

## Overview

**Long RAG** is an advanced document question-answering system designed to leverage transformer models for processing and querying PDF documents. This project facilitates the ingestion of large volumes of text from PDFs, transforming them into a format suitable for effective querying.

## Features

- **PDF Processing**: Advanced PDF text extraction using the `marker` library
- **Document Chunking**: Intelligent text splitting for optimal processing
- **Vector Search**: FAISS-powered similarity search for quick and accurate retrieval
- **Conversation Memory**: Context-aware responses using conversation buffer
- **User Interface**: Clean, intuitive Gradio interface for document upload and querying
- **Async Support**: Asynchronous processing for better performance
- **Error Handling**: Comprehensive logging and error management

## Technical Stack

- **Language Models**: Meta Llama 3.2 3B Instruct
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Vector Store**: FAISS
- **PDF Processing**: marker
- **Frontend**: Gradio
- **Processing**: LangChain for document processing and chain management

## Installation

```bash
# Clone the repository
git clone https://github.com/driessenslucas/longRAG
cd longRAG

# Install dependencies
conda env create -f environment.yml
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the UI at `http://localhost:7860`

3. Upload PDF documents and initialize the system

4. Start asking questions about your documents

## Configuration

Key configurations are managed through the `TransformerConfig` class:

```python
class TransformerConfig:
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_NEW_TOKENS = 512
    TOP_P = 0.95
    CONTEXT_WINDOW = 3900
```

## Error Handling

The system includes comprehensive error handling and logging:
- File processing errors are caught and reported
- Model initialization failures are logged
- Query processing issues are handled gracefully

## API Reference

### Main Functions

#### `initialize_workflow_with_files(files)`
Initializes the QA system with uploaded PDF files.

#### `process_query(query: str)`
Processes a user query against the loaded documents.

#### `process_pdf(file_obj)`
Converts PDF files to processable text documents.

## Limitations

- PDF files must be text-searchable
- Large documents may require significant processing time
- Memory usage scales with document size
- Response quality depends on document clarity and question specificity

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.