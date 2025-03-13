import hashlib
import json
import os
import sqlite3
from typing import List, Dict

# Document processing
from docling.document_converter import DocumentConverter
# Use Chroma for vector store
from langchain_chroma import Chroma
from langchain_core.document_loaders import BaseLoader
# Persian support
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import PromptTemplate
# Huggingface embeddings instead of Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from TextProcessing.FastPersianNormalizer import FastPersianNormalizer

# Define constants
DB_PATH = "pdf_database.db"
VECTOR_STORE_PATH = "chroma_db"


class DoclingLoader(BaseLoader):
    """Processes documents using Docling with Persian language support"""

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()
        self.normalizer = FastPersianNormalizer()

    def load(self) -> List[LCDocument]:
        """Load all documents at once"""
        docs = []
        for source in self._file_paths:
            try:
                dl_doc = self._converter.convert(source).document
                text = dl_doc.export_to_markdown()
                normalized_text = self.normalizer.normalize(text)

                metadata = {
                    "source": source,
                    "filename": os.path.basename(source),
                    "filetype": os.path.splitext(source)[1][1:],
                }

                docs.append(LCDocument(page_content=normalized_text, metadata=metadata))
            except Exception as e:
                print(f"Error processing {source}: {e}")
        return docs


class DocumentProcess:
    def __init__(self, model_name: str = "gemma3",
                 embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize with LLM and embedding models

        Args:
            model_name: Ollama model name for response generation
            embeddings_model: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self.db_conn = self._init_database()

        # Use HuggingFace embeddings instead of Ollama
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            cache_folder="./hf_cache"  # Cache embeddings locally
        )

        # Persian-aware text splitter with simplified settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " ", ""],
        )

        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=self.hf_embeddings
        )

        # RAG prompt template
        self.prompt = PromptTemplate.from_template("""
        متن زمینه در زیر آمده است.
        ---------------------
        {context}
        ---------------------
        با توجه به اطلاعات زمینه و بدون استفاده از دانش قبلی، به پرسش پاسخ دهید.
        پرسش: {question}
        پاسخ:
        """)

    @staticmethod
    def _init_database() -> sqlite3.Connection:
        """Initialize SQLite database for tracking processed files"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
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

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_documents(self, file_paths: List[str], progress_callback=None) -> Dict:
        """Process documents and add to vector store if not already processed"""
        new_documents = []
        processed_count = skipped_count = 0
        errors = []

        for i, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                errors.append(f"فایل پیدا نشد: {file_path}")
                continue

            file_hash = self._calculate_file_hash(file_path)

            # Check if already processed
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT 1 FROM processed_files WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone():
                skipped_count += 1
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), f"رد کردن فایل تکراری: {os.path.basename(file_path)}")
                continue

            try:
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), f"در حال پردازش: {os.path.basename(file_path)}")

                # Load and process document
                documents = DoclingLoader(file_path).load()
                page_count = len(documents)

                # Split into chunks
                chunks = self.text_splitter.split_documents(documents)
                new_documents.extend(chunks)

                # Record in database
                self.db_conn.execute(
                    "INSERT INTO processed_files VALUES (?, ?, ?, ?, datetime('now'), ?)",
                    (
                        file_hash,
                        file_path,
                        os.path.basename(file_path),
                        page_count,
                        json.dumps({"language": "mixed", "contains_persian": True})
                    )
                )
                self.db_conn.commit()
                processed_count += 1

            except Exception as e:
                errors.append(f"خطا در پردازش {file_path}: {e}")

        # Add documents to vector store if any new ones
        if new_documents:
            if progress_callback:
                progress_callback(len(file_paths), len(file_paths),
                                  f"افزودن {len(new_documents)} قطعه به پایگاه داده برداری...")
            self.vector_store.add_documents(new_documents)

        return {
            "processed": processed_count,
            "skipped": skipped_count,
            "errors": errors,
            "new_chunks": len(new_documents)
        }

    def query(self, question: str, top_k: int = 4, progress_callback=None) -> str:
        """Optimized query function with error handling"""
        try:
            # Progress reporting
            if progress_callback:
                progress_callback(0.3, "در حال جستجوی اطلاعات مرتبط...")

            # Create retriever with limited top_k
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

            # Get source documents
            source_docs = retriever.invoke(question)

            # Format context
            context = "\n\n".join(doc.page_content for doc in source_docs)

            if progress_callback:
                progress_callback(0.7, "در حال تولید پاسخ...")

            # Generate response with explicit base_url
            llm = OllamaLLM(
                model=self.model_name,
                base_url="http://localhost:11434",  # Explicitly set base URL
                temperature=0.1  # Lower temperature for more focused responses
            )
            answer = llm.invoke(self.prompt.format(context=context, question=question))

            # Format sources
            sources = "\n".join(
                f"منبع {i}: {os.path.basename(doc.metadata.get('source', 'منبع ناشناخته'))}"
                for i, doc in enumerate(source_docs, 1)
            )

            if progress_callback:
                progress_callback(1.0, "تکمیل شد")

            return f"پاسخ: {answer}\n\nمنابع:\n{sources}"

        except Exception as e:
            error_message = f"خطا در پردازش پرسش: {str(e)}"
            print(error_message)
            return f"متأسفانه خطایی رخ داد: {error_message}\n\nلطفاً اطمینان حاصل کنید که سرویس Ollama در حال اجراست و مدل {self.model_name} نصب شده است."
    def list_processed_files(self) -> List[Dict]:
        """List all processed files"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT file_path, file_name, page_count, processed_at FROM processed_files")
        return [
            {
                "file_path": row[0],
                "file_name": row[1],
                "page_count": row[2],
                "processed_at": row[3]
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        """Close database connection"""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()