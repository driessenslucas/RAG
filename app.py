import gradio as gr
import os
import tempfile
import logging
import asyncio
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from langchain.schema import Document as LangchainDocument

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for storing documents and QA chain
documents = []
qa_chain = None

class TransformerConfig:
    """Configuration class for transformer model settings."""
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_NEW_TOKENS = 512
    TOP_P = 0.95
    CONTEXT_WINDOW = 3900

def init_llm():
    """Initialize the transformer model with proper configuration."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=TransformerConfig.MODEL_ID,
            max_length=TransformerConfig.MAX_NEW_TOKENS,
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def process_pdf(file_obj) -> list[LangchainDocument]:
    """Process a PDF file and return its content as LangChain Document objects."""
    try:
        # Get the file path directly from the Gradio file object
        tmp_path = file_obj.name  # Gradio provides a file path for uploaded files

        # Load models from marker
        model_lst = load_all_models()

        # Convert PDF to text using marker
        full_text, images, out_meta = convert_single_pdf(tmp_path, model_lst)

        if not full_text.strip():
            raise Exception("No readable text found in PDF")

        # Create a LangChain Document
        metadata = {
            'title': out_meta.get('title', 'Untitled'),
            'author': out_meta.get('author', 'Unknown'),
            'source': tmp_path,
        }
        document = LangchainDocument(page_content=full_text, metadata=metadata)

        return [document]  # Return a list of documents
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise Exception(f"Error processing PDF: {str(e)}")

async def initialize_workflow_with_files(files) -> str:
    """Initialize the workflow with uploaded files."""
    global qa_chain
    if not files:
        return "❌ Please upload at least one PDF file."
    
    try:
        # Initialize LLM and embeddings
        llm = init_llm()
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # Process PDFs and prepare documents
        documents = []
        for file in files:
            docs = process_pdf(file)
            documents.extend(docs)
        
        if not documents:
            return "❌ No valid documents were processed."

        # Create text splitter and vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(split_docs, embeddings)

        # Initialize the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            memory=ConversationBufferMemory()
        )

        return "✅ Workflow initialized successfully!"
    except Exception as e:
        logger.error(f"Error in workflow initialization: {str(e)}")
        return f"❌ Error initializing workflow: {str(e)}"

async def process_query(query: str) -> str:
    """Process a query using the initialized QA chain."""
    global qa_chain
    if not qa_chain:
        return "Please initialize the workflow by uploading PDF files first!"
    
    try:
        result = await qa_chain.arun(query)
        return str(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Error processing query: {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📚 Document Q&A with Transformer Models
            Upload PDF documents and ask questions about their content.
            """
        )
        
        with gr.Row():
            with gr.Column():
                file_output = gr.File(
                    file_count="multiple",
                    label="Upload PDF Documents",
                    file_types=[".pdf"]
                )
                init_button = gr.Button("🚀 Initialize System", variant="primary")
        
        init_output = gr.Textbox(
            label="System Status",
            lines=8,
            show_copy_button=True
        )
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know about the documents?",
                    lines=3
                )
                query_button = gr.Button("🔍 Submit Question", variant="primary")
        
        response_output = gr.Textbox(
            label="Answer",
            lines=10,
            show_copy_button=True
        )
        
        # Event handlers
        init_button.click(
            fn=lambda x: asyncio.run(initialize_workflow_with_files(x)),
            inputs=[file_output],
            outputs=[init_output]
        )
        
        query_button.click(
            fn=lambda x: asyncio.run(process_query(x)),
            inputs=[query_input],
            outputs=[response_output]
        )
        
        gr.Markdown(
            """
            ### Notes:
            - The system uses transformer-based models for both embedding and question answering.
            - Make sure your PDFs are text-searchable for best results.
            - Large documents may take longer to process.
            """
        )
    
    return demo

def upload_documents(files):
    """Process uploaded files and add to documents list."""
    global documents
    if not files:
        return "❌ No files uploaded."
    
    for file in files:
        try:
            docs = process_pdf(file)
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")

    return "✅ Documents uploaded successfully!"

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
