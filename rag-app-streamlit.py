import hashlib
import json
import os
import sqlite3
import tempfile
from typing import List, Dict

# For PDF processing
import fitz  # PyMuPDF
# Persian language support
import hazm
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings  # Updated import for OllamaEmbeddings
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the database path
DB_PATH = "pdf_database.db"
VECTOR_STORE_PATH = "chroma_db"

# Streamlit configuration
st.set_page_config(
    page_title="سیستم RAG فایل‌های PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set RTL direction for the entire app
st.markdown("""
<style>
    .stApp {
        direction: rtl;
    }
    div[data-testid="stForm"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stMarkdown"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stText"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stTable"] th, div[data-testid="stTable"] td {
        text-align: right;
    }
    div[data-testid="stFileUploader"] > label {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label {
        direction: rtl;
        text-align: right;
    }
    button[kind="primary"], button[kind="secondary"] {
        float: right;
    }
    div[data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    section[data-testid="stSidebar"] > div {
        direction: rtl;
    }
    div[data-testid="stExpander"] {
        direction: rtl;
    }
    div[data-testid="stSlider"] > div {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)


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

    def process_pdfs(self, pdf_paths: List[str], progress_callback=None) -> Dict:
        """Process a list of PDF files and add them to the database if not already processed."""
        new_documents = []
        processed_count = 0
        skipped_count = 0
        errors = []

        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                errors.append(f"فایل پیدا نشد: {pdf_path}")
                continue

            file_hash = self._calculate_file_hash(pdf_path)

            if self._is_file_processed(file_hash):
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

    def query(self, question: str, top_k: int = 4, progress_callback=None) -> str:
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
        if progress_callback:
            progress_callback(0.3, "در حال جستجو برای اطلاعات مرتبط...")

        result = qa_chain.invoke({"query": question})

        if progress_callback:
            progress_callback(0.7, "در حال تولید پاسخ...")

        answer = result["result"]
        sources = result["source_documents"]

        # Format sources for display
        source_texts = []
        for i, doc in enumerate(sources, 1):
            source_file = doc.metadata.get("source_file", "منبع ناشناخته")
            page_num = doc.metadata.get("page", "صفحه ناشناخته")
            source_texts.append(f"منبع {i}: {os.path.basename(source_file)}، صفحه {page_num}")

        # Format and return the response
        formatted_answer = f"""پاسخ: {answer}\n\nمنابع:\n"""
        formatted_answer += "\n".join(source_texts)

        if progress_callback:
            progress_callback(1.0, "تکمیل شد")

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


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path."""
    try:
        # Create a temporary file path
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        st.error(f"خطا در ذخیره فایل: {e}")
        return None


def main():
    # Sidebar for settings and navigation
    st.sidebar.title("سیستم RAG فایل‌های PDF")
    st.sidebar.subheader("با پشتیبانی از زبان فارسی")

    # Navigation
    page = st.sidebar.radio("صفحات", ["پردازش فایل‌ها", "پرسش و پاسخ", "فایل‌های پردازش شده"])

    # Model settings
    st.sidebar.subheader("تنظیمات مدل")
    model_name = st.sidebar.text_input("مدل LLM (Ollama)", value="llama3.1")
    embeddings_model = st.sidebar.text_input("مدل Embeddings", value="nomic-embed-text")

    # Initialize processor
    processor = PDFProcessor(model_name=model_name, embeddings_model=embeddings_model)

    try:
        # Process PDF files page
        if page == "پردازش فایل‌ها":
            st.title("پردازش فایل‌های PDF")
            st.write("فایل‌های PDF خود را آپلود کنید تا پردازش شوند.")

            uploaded_files = st.file_uploader("فایل‌های PDF را انتخاب کنید",
                                              type="pdf",
                                              accept_multiple_files=True)

            col1, col2 = st.columns([4, 1])
            with col2:
                process_button = st.button("پردازش فایل‌ها", type="primary")

            if uploaded_files and process_button:
                pdf_paths = []

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Save uploaded files to temp location
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        pdf_paths.append(file_path)

                if pdf_paths:
                    # Define callback for progress updates
                    def update_progress(current, total, message=""):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)

                    # Process PDFs with progress updates
                    result = processor.process_pdfs(pdf_paths, update_progress)

                    # Display results
                    progress_bar.progress(1.0)
                    status_text.text("پردازش کامل شد!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("فایل‌های پردازش شده", result["processed"])
                    with col2:
                        st.metric("فایل‌های رد شده", result["skipped"])
                    with col3:
                        st.metric("تکه‌های متنی جدید", result["new_chunks"])

                    if result["errors"]:
                        st.error("خطاها:")
                        for error in result["errors"]:
                            st.write(f"- {error}")

                    st.success(f"{result['processed']} فایل با موفقیت پردازش شد و به پایگاه داده اضافه گردید.")

        # Query page
        elif page == "پرسش و پاسخ":
            st.title("پرسش و پاسخ از اسناد")

            # Check if there are documents in the database
            files = processor.list_processed_files()
            if not files:
                st.warning("هیچ فایلی در پایگاه داده موجود نیست. لطفا ابتدا فایل‌ها را پردازش کنید.")
            else:
                st.write(f"{len(files)} فایل در پایگاه داده موجود است.")

                # Query input
                query = st.text_area("پرسش خود را وارد کنید:", height=100)
                top_k = st.slider("تعداد منابع برای بازیابی:", min_value=1, max_value=10, value=4)

                col1, col2 = st.columns([4, 1])
                with col2:
                    submit_button = st.button("ارسال پرسش", type="primary")

                if submit_button:
                    if not query:
                        st.warning("لطفا یک پرسش وارد کنید.")
                    else:
                        # Show spinners and progress
                        progress_placeholder = st.empty()
                        progress_bar = st.progress(0)

                        def update_query_progress(progress, message):
                            progress_bar.progress(progress)
                            progress_placeholder.text(message)

                        # Execute query
                        with st.spinner("در حال پردازش پرسش..."):
                            response = processor.query(query, top_k, update_query_progress)

                        # Display response
                        st.subheader("پاسخ:")

                        # Split answer and references
                        parts = response.split("\n\nمنابع:\n")
                        answer = parts[0].replace("پاسخ: ", "")
                        references = parts[1] if len(parts) > 1 else ""

                        st.markdown(f"""<div style="direction: rtl; text-align: right;">{answer}</div>""",
                                    unsafe_allow_html=True)

                        # Display references in an expander
                        with st.expander("منابع"):
                            for line in references.split("\n"):
                                st.markdown(f"""<div style="direction: rtl; text-align: right;">{line}</div>""",
                                            unsafe_allow_html=True)

        # List processed files page
        elif page == "فایل‌های پردازش شده":
            st.title("فایل‌های پردازش شده")

            files = processor.list_processed_files()
            if not files:
                st.info("هیچ فایلی پردازش نشده است.")
            else:
                st.write(f"{len(files)} فایل در پایگاه داده موجود است.")

                # Create a table to display files
                data = []
                for i, file in enumerate(files, 1):
                    data.append({
                        "شماره": i,
                        "نام فایل": file["file_name"],
                        "تعداد صفحات": file["page_count"],
                        "تاریخ پردازش": file["processed_at"]
                    })

                st.table(data)

    finally:
        processor.close()


if __name__ == "__main__":
    main()