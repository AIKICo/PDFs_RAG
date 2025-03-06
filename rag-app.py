import argparse
import hashlib
import json
import os
import sqlite3
from typing import List, Dict

# For PDF processing
import fitz  # PyMuPDF
# Persian language support
import hazm
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings  # Updated import for OllamaEmbeddings
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Define the database path
DB_PATH = "pdf_database.db"
VECTOR_STORE_PATH = "chroma_db"


class PDFProcessor:
    def __init__(self, model_name: str = "llama3", embeddings_model: str = "nomic-embed-text"):
        """
        Initialize the PDF processor with specified models.

        Args:
            model_name: Ollama model name for generating responses
            embeddings_model: Ollama model name for embeddings
        """
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self.db_conn = self._init_database()
        self.ollama_embeddings = OllamaEmbeddings(model=embeddings_model)

        # Initialize the Persian normalizer and tokenizer
        self.normalizer = hazm.Normalizer()

        # Persian-aware text splitter settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " ", ""],
            keep_separator=True,
        )

        # Create/load vector store
        self._init_vector_store()

    @staticmethod
    def _init_database() -> sqlite3.Connection:
        """Initialize SQLite database for tracking processed files."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            page_count INTEGER,
            processed_at TEXT,
            metadata TEXT
        )
        ''')

        conn.commit()
        return conn

    def _init_vector_store(self):
        """Initialize or load the vector store for document embeddings."""
        if os.path.exists(VECTOR_STORE_PATH):
            self.vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH,
                embedding_function=self.ollama_embeddings
            )
        else:
            # Create an empty vector store if it doesn't exist
            self.vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH,
                embedding_function=self.ollama_embeddings
            )

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of a file to uniquely identify it."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _is_file_processed(self, file_hash: str) -> bool:
        """Check if a file has already been processed based on its hash."""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT file_hash FROM processed_files WHERE file_hash = ?", (file_hash,))
        result = cursor.fetchone()
        return result is not None

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text with Persian language support."""
        # Normalize Persian text
        text = self.normalizer.normalize(text)
        return text

    def _extract_text_from_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF file with preprocessing for Persian text."""
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        # Preprocess each document's page content
        for doc in documents:
            doc.page_content = self._preprocess_text(doc.page_content)

            # Add the file path to metadata
            doc.metadata["source_file"] = file_path

        return documents

    def process_pdfs(self, pdf_paths: List[str]) -> None:
        """Process a list of PDF files and add them to the database if not already processed."""
        new_documents = []
        processed_count = 0
        skipped_count = 0

        print("Starting PDF processing...")

        for pdf_path in tqdm(pdf_paths, desc="Processing PDF files"):
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                continue

            file_hash = self._calculate_file_hash(pdf_path)

            if self._is_file_processed(file_hash):
                print(f"Skipping already processed file: {os.path.basename(pdf_path)}")
                skipped_count += 1
                continue

            try:
                # Extract metadata
                pdf_document = fitz.open(pdf_path)
                page_count = len(pdf_document)

                # Extract text from PDF
                documents = self._extract_text_from_pdf(pdf_path)

                # Split documents into chunks
                for doc in tqdm(documents, desc=f"Splitting {os.path.basename(pdf_path)}", leave=False):
                    chunks = self.text_splitter.split_documents([doc])
                    new_documents.extend(chunks)

                # Record the processed file in the database
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "INSERT INTO processed_files VALUES (?, ?, ?, ?, datetime('now'), ?)",
                    (
                        file_hash,
                        pdf_path,
                        os.path.basename(pdf_path),
                        page_count,
                        json.dumps({"language": "mixed", "contains_persian": True})
                    )
                )
                self.db_conn.commit()

                processed_count += 1

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

        # Add new documents to vector store if there are any
        if new_documents:
            print(f"Adding {len(new_documents)} new document chunks to vector store...")
            self.vector_store.add_documents(new_documents)
            print("Vector store updated and persisted.")

        print(f"Processing complete. {processed_count} files processed, {skipped_count} files skipped.")

    def query(self, question: str, top_k: int = 4) -> str:
        """Query the system with a question and get a response."""
        # Initialize Ollama for response generation
        llm = OllamaLLM(model=self.model_name)

        # Create a retrieval QA chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Execute the query
        print("Searching for relevant information...")
        # result = qa_chain({"query": question})
        result = qa_chain.invoke({"query": question})

        answer = result["result"]
        sources = result["source_documents"]

        # Format sources for display
        source_texts = []
        for i, doc in enumerate(sources, 1):
            source_file = doc.metadata.get("source_file", "Unknown source")
            page_num = doc.metadata.get("page", "Unknown page")
            source_texts.append(f"Source {i}: {os.path.basename(source_file)}, Page {page_num}")

        # Format and return the response
        formatted_answer = f"""Answer: {answer}\n\nReferences:\n"""
        formatted_answer += "\n".join(source_texts)

        return formatted_answer

    def list_processed_files(self) -> List[Dict]:
        """List all processed files in the database."""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT file_path, file_name, page_count, processed_at FROM processed_files")
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "file_path": row[0],
                "file_name": row[1],
                "page_count": row[2],
                "processed_at": row[3]
            })

        return result

    def close(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()


def main():
    parser = argparse.ArgumentParser(description="PDF RAG System with Persian language support")
    parser.add_argument("--process", action="store_true", help="Process PDF files")
    parser.add_argument("--query", type=str, help="Query the system")
    parser.add_argument("--list", action="store_true", help="List processed files")
    parser.add_argument("--pdf-dir", type=str, help="Directory containing PDF files to process")
    parser.add_argument("--pdf-files", nargs="+", help="Specific PDF files to process")
    parser.add_argument("--model", type=str, default="llama3.1", help="Ollama model for responses")
    parser.add_argument("--embeddings", type=str, default="nomic-embed-text", help="Ollama model for embeddings")

    args = parser.parse_args()

    processor = PDFProcessor(model_name=args.model, embeddings_model=args.embeddings)

    try:
        if args.process:
            pdf_files = []

            # Process specific PDF files
            if args.pdf_files:
                pdf_files.extend(args.pdf_files)

            # Process all PDFs in a directory
            if args.pdf_dir:
                if os.path.isdir(args.pdf_dir):
                    for file in os.listdir(args.pdf_dir):
                        if file.lower().endswith(".pdf"):
                            pdf_files.append(os.path.join(args.pdf_dir, file))
                else:
                    print(f"Directory not found: {args.pdf_dir}")

            if pdf_files:
                processor.process_pdfs(pdf_files)
            else:
                print("No PDF files specified for processing.")

        elif args.query:
            response = processor.query(args.query)
            print("\n" + "=" * 50)
            print(response)
            print("=" * 50)

        elif args.list:
            files = processor.list_processed_files()
            print("\nProcessed Files:")
            print("=" * 80)
            for i, file in enumerate(files, 1):
                print(f"{i}. {file['file_name']} ({file['page_count']} pages) - Processed at: {file['processed_at']}")
            print("=" * 80)

        else:
            print("Please specify an action: --process, --query, or --list")

    finally:
        processor.close()


if __name__ == "__main__":
    main()