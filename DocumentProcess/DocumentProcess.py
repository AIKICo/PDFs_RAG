import hashlib
import json
import os
import sqlite3
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from FileProcessing.EnhancedDoclingLoader import EnhancedDoclingLoader

DB_PATH = "document_database.db"
VECTOR_STORE_PATH = "chroma_db"
MEMORY_FILE_PATH = "memory.json"


class DocumentProcess:
    def __init__(self, model_name: str = "gemma3",
                 embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self.db_conn = self._init_database()

        # تنظیمات حافظه گفتگو - استفاده از روش جدید
        self.message_history = FileChatMessageHistory(MEMORY_FILE_PATH)

        # تنظیمات تعبیه‌ها
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            cache_folder="./hf_cache"
        )

        # تنظیمات تقسیم‌کننده متن
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " ", ""],
        )

        # تنظیمات پایگاه داده برداری
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=self.hf_embeddings
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "شما یک دستیار هوش مصنوعی هستید که بر اساس اطلاعات زمینه و تاریخچه گفتگو پاسخ می‌دهید. "
             "در پاسخ‌های خود دقت کنید و فقط بر اساس داده‌های ارائه‌شده نتیجه‌گیری کنید. "
             "اگر اطلاعات کافی ندارید، بگویید که نمی‌توانید پاسخ دهید. "
             "پاسخ ها به فارسی باشد و اگر پاسخ تو انگلیسی بود ابتدا ترجمه شود و این ترجمه شامل پاسخ های مرتبط با کدنویس نباشد"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "متن زمینه:\n{context}\n\nپرسش:\n{question}\n\nپاسخ خود را ارائه دهید.")
        ])

    @staticmethod
    def _init_database() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                file_name TEXT,
                file_type TEXT,
                page_count INTEGER,
                processed_at TEXT,
                metadata TEXT
            )
            ''')
        conn.commit()
        return conn

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_file_stats(self, file_path: str) -> Dict:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        file_type = os.path.splitext(file_path)[1][1:].lower()
        return {
            "file_size": round(file_size, 2),
            "file_type": file_type,
            "file_name": os.path.basename(file_path)
        }

    def process_documents(self, file_paths: List[str], progress_callback=None) -> Dict:
        new_documents = []
        processed_count = skipped_count = 0
        errors = []

        for i, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                errors.append(f"فایل پیدا نشد: {file_path}")
                continue

            file_hash = self._calculate_file_hash(file_path)
            file_stats = self.get_file_stats(file_path)

            cursor = self.db_conn.cursor()
            cursor.execute("SELECT 1 FROM processed_files WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone():
                skipped_count += 1
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), f"رد کردن فایل تکراری: {file_stats['file_name']}")
                continue

            try:
                if progress_callback:
                    progress_callback(i + 1, len(file_paths),
                                      f"در حال پردازش: {file_stats['file_name']} ({file_stats['file_size']} MB)")

                documents = EnhancedDoclingLoader(file_path).load()

                if not documents:
                    errors.append(f"فایل پشتیبانی نمی‌شود یا خالی است: {file_path}")
                    continue

                page_count = len(documents)
                chunks = self.text_splitter.split_documents(documents)
                new_documents.extend(chunks)

                self.db_conn.execute(
                    "INSERT INTO processed_files VALUES (?, ?, ?, ?, ?, datetime('now'), ?)",
                    (
                        file_hash,
                        file_path,
                        file_stats['file_name'],
                        file_stats['file_type'],
                        page_count,
                        json.dumps(
                            {"language": "mixed", "contains_persian": True, "file_size_mb": file_stats['file_size']})
                    )
                )
                self.db_conn.commit()
                processed_count += 1

            except Exception as e:
                errors.append(f"خطا در پردازش {file_path}: {str(e)}")

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
        try:
            if progress_callback:
                progress_callback(0.3, "در حال جستجوی اطلاعات مرتبط...")

            # بازیابی اسناد مرتبط
            retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": 5})
            source_docs = retriever.invoke(question)

            if not source_docs:
                return "اطلاعات مرتبطی با پرسش شما یافت نشد."

            context = "\n\n".join(doc.page_content for doc in source_docs)

            if progress_callback:
                progress_callback(0.7, "در حال تولید پاسخ...")

            # تنظیم مدل ChatOllama
            llm = ChatOllama(
                model=self.model_name,
                base_url="http://localhost:11434",
                temperature=0.7
            )

            # ایجاد chain با حافظه
            chain = self.prompt | llm
            chain_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: self.message_history,
                input_messages_key="question",
                history_messages_key="history",
            )

            # تولید پاسخ با استفاده از زنجیره با حافظه
            answer = chain_with_history.invoke(
                {"context": context, "question": question},
                config={"configurable": {"session_id": "default"}}
            ).content

            # تهیه منابع
            sources = "\n".join(
                f"منبع {i}: {os.path.basename(doc.metadata.get('source', 'منبع ناشناخته'))} (نوع: {doc.metadata.get('filetype', 'نامشخص')})"
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
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT file_path, file_name, file_type, page_count, processed_at, metadata FROM processed_files")
        return [
            {
                "file_path": row[0],
                "file_name": row[1],
                "file_type": row[2],
                "page_count": row[3],
                "processed_at": row[4],
                "metadata": json.loads(row[5]) if row[5] else {}
            }
            for row in cursor.fetchall()
        ]

    def remove_document(self, file_hash: str) -> bool:
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT file_path FROM processed_files WHERE file_hash = ?", (file_hash,))
            result = cursor.fetchone()

            if not result:
                return False

            file_path = result[0]

            # حذف اسناد مرتبط از پایگاه داده برداری
            # جستجو و حذف تمام قطعه‌های مرتبط با این فایل در Chroma
            self.vector_store.delete(
                where={"source": file_path}
            )

            # حذف از پایگاه داده SQLite
            cursor.execute("DELETE FROM processed_files WHERE file_hash = ?", (file_hash,))
            self.db_conn.commit()

            return True
        except Exception as e:
            print(f"Error removing document: {e}")
            return False

    def close(self):
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()

    def clearChatHitsory(self):
        self.message_history.clear()
