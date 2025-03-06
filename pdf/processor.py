import hashlib
import os
from typing import List, Dict, Callable, Optional, Tuple

# For PDF processing
import fitz
# PyMuPDF
# Persian language support
import hazm
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import settings
from ..core.database import Database


class PDFProcessor:
    def __init__(self, model_name: str = settings.DEFAULT_LLM_MODEL,
                 embeddings_model: str = settings.DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the PDF processor with specified models.

        Args:
            model_name: Ollama model name for generating responses
            embeddings_model: Ollama model name for embeddings
        """
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self.db = Database(settings.DB_PATH)
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

    def _init_vector_store(self):
        """Initialize or load the vector store for document embeddings."""
        if os.path.exists(settings.VECTOR_STORE_PATH):
            self.vector_store = Chroma(
                persist_directory=settings.VECTOR_STORE_PATH,
                embedding_function=self.ollama_embeddings
            )
        else:
            # Create an empty vector store if it doesn't exist
            self.vector_store = Chroma(
                persist_directory=settings.VECTOR_STORE_PATH,
                embedding_function=self.ollama_embeddings
            )

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of a file to uniquely identify it."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

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

    def process_pdfs(self, pdf_paths: List[str], progress_callback: Optional[Callable] = None) -> Dict:
        """Process a list of PDF files and add them to the database if not already processed."""
        new_documents = []
        processed_count = 0
        skipped_count = 0
        errors = []

        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                errors.append(f"فایل پیدا نشد: {pdf_path}")
                continue

            file_hash = self.calculate_file_hash(pdf_path)

            if self.db.is_file_processed(file_hash):
                skipped_count += 1
                if progress_callback:
                    progress_callback(i + 1, len(pdf_paths),
                                      f"رد کردن فایل از قبل پردازش شده: {os.path.basename(pdf_path)}")
                continue

            try:
                if progress_callback:
                    progress_callback(i + 1, len(pdf_paths), f"در حال پردازش: {os.path.basename(pdf_path)}")

                # Extract metadata
                pdf_document = fitz.open(pdf_path)
                page_count = len(pdf_document)

                # Extract text from PDF
                documents = self._extract_text_from_pdf(pdf_path)

                # Split documents into chunks
                for j, doc in enumerate(documents):
                    if progress_callback:
                        sub_progress = f"تقسیم صفحه {j + 1}/{len(documents)} از {os.path.basename(pdf_path)}"
                        progress_callback(i + 1, len(pdf_paths), sub_progress)

                    chunks = self.text_splitter.split_documents([doc])
                    new_documents.extend(chunks)

                # Record the processed file in the database
                self.db.add_processed_file(
                    file_hash,
                    pdf_path,
                    os.path.basename(pdf_path),
                    page_count,
                    {"language": "mixed", "contains_persian": True}
                )

                processed_count += 1

            except Exception as e:
                error_msg = f"خطا در پردازش {pdf_path}: {e}"
                errors.append(error_msg)

        # Add new documents to vector store if there are any
        if new_documents:
            if progress_callback:
                progress_callback(len(pdf_paths), len(pdf_paths),
                                  f"اضافه کردن {len(new_documents)} قطعه سند به پایگاه داده برداری...")

            self.vector_store.add_documents(new_documents)

        result = {
            "processed": processed_count,
            "skipped": skipped_count,
            "errors": errors,
            "new_chunks": len(new_documents)
        }

        return result

    def query(self, question: str, top_k: int = 4, progress_callback: Optional[Callable] = None) -> Tuple[
        str, List[Dict]]:
        """
        Query the system with a question and get a response.

        Returns:
            Tuple containing (answer, sources)
        """
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
        if progress_callback:
            progress_callback(0.3, "در حال جستجو برای اطلاعات مرتبط...")

        result = qa_chain.invoke({"query": question})

        if progress_callback:
            progress_callback(0.7, "در حال تولید پاسخ...")

        answer = result["result"]
        source_docs = result["source_documents"]

        # Format sources for display and API
        sources = []
        for i, doc in enumerate(source_docs, 1):
            source_file = doc.metadata.get("source_file", "منبع ناشناخته")
            page_num = doc.metadata.get("page", "صفحه ناشناخته")
            sources.append({
                "id": i,
                "file": os.path.basename(source_file),
                "page": page_num,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })

        if progress_callback:
            progress_callback(1.0, "تکمیل شد")

        return answer, sources

    def list_processed_files(self) -> List[Dict]:
        """List all processed files in the database."""
        return self.db.get_processed_files()

    def close(self):
        """Close database connection."""
        self.db.close()