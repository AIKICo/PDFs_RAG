import hashlib
import json
import os
import sqlite3
from typing import List, Dict, Any

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
             "متن‌های زمینه بر اساس اهمیت و ارتباط آنها با پرسش رتبه‌بندی شده‌اند. "
             "هر قطعه متن با علامت‌های # نشان‌گذاری شده است. تعداد بیشتر # نشانه اهمیت بیشتر آن قطعه است. "
             "در پاسخ‌های خود بر محتوای متن‌های با اهمیت بیشتر تمرکز کنید. "
             "اگر اطلاعات کافی ندارید، بگویید که نمی‌توانید پاسخ دهید. "
             "پاسخ ها به فارسی باشد و اگر پاسخ تو انگلیسی بود ابتدا ترجمه شود و این ترجمه شامل پاسخ های مرتبط با کدنویسی نباشد"),
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

    @staticmethod
    def get_file_stats(file_path: str) -> Dict:
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

    def _summarize_long_documents(self, docs, max_length=500):
        """خلاصه‌سازی اسناد طولانی‌تر از حد مشخص"""
        summarized_docs = []

        for doc in docs:
            if len(doc.page_content) > max_length:
                # استفاده از LLM برای خلاصه‌سازی
                llm = ChatOllama(model=self.model_name, base_url="http://localhost:11434")
                summarize_prompt = ChatPromptTemplate.from_messages([
                    ("system", "متن زیر را به طور خلاصه و با حفظ نکات مهم و مرتبط آن خلاصه کنید."),
                    ("human", "{text}")
                ])

                chain = summarize_prompt | llm
                summary = chain.invoke({"text": doc.page_content}).content

                # ایجاد یک نسخه جدید از سند با محتوای خلاصه شده
                from copy import deepcopy
                new_doc = deepcopy(doc)
                new_doc.page_content = summary
                new_doc.metadata["summarized"] = True
                summarized_docs.append(new_doc)
            else:
                summarized_docs.append(doc)

        return summarized_docs

    @staticmethod
    def _rerank_documents(query, docs, alpha=0.3, beta=0.2, gamma=0.5):
        """
        امتیازدهی مجدد به اسناد با استفاده از معیارهای چندگانه:
        - امتیاز شباهت برداری (vector similarity)
        - طول مناسب قطعه (chunk length appropriateness)
        - کیفیت متن (مانند نسبت کلمات کلیدی در متن)

        alpha: ضریب اهمیت طول مناسب
        beta: ضریب اهمیت کیفیت متن
        gamma: ضریب اهمیت امتیاز اولیه شباهت برداری
        """
        # استخراج کلمات کلیدی از پرسش
        query_keywords = set([word.strip() for word in query.lower().split() if len(word.strip()) > 2])

        reranked_docs = []
        for doc in docs:
            # امتیاز اولیه شباهت برداری (فرض می‌کنیم در metadata وجود دارد یا از قبل محاسبه شده)
            vector_score = doc.metadata.get('score', 0.5)  # اگر امتیاز نباشد، از 0.5 استفاده می‌کنیم

            # محاسبه امتیاز طول مناسب (قطعات خیلی کوتاه یا خیلی بلند امتیاز کمتری می‌گیرند)
            text_length = len(doc.page_content)
            length_score = 1.0 - abs((text_length - 500) / 1000)  # 500 کاراکتر طول ایده‌آل در این مثال
            length_score = max(0.1, min(1.0, length_score))  # محدود کردن بین 0.1 و 1.0

            # محاسبه امتیاز کیفیت متن (نسبت کلمات کلیدی موجود در متن)
            doc_content_lower = doc.page_content.lower()
            keyword_matches = sum(1 for keyword in query_keywords if keyword in doc_content_lower)

            # نسبت تطابق کلمات کلیدی (مقدار بین 0 و 1)
            quality_score = keyword_matches / len(query_keywords) if query_keywords else 0.5

            # محاسبه امتیاز نهایی با ترکیب سه معیار
            final_score = (alpha * length_score) + (beta * quality_score) + (gamma * vector_score)

            # اضافه کردن امتیاز به متادیتا برای استفاده بعدی
            doc.metadata['rerank_score'] = final_score
            reranked_docs.append((doc, final_score))

        # مرتب‌سازی بر اساس امتیاز نهایی (نزولی)
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # بازگرداندن فقط اسناد (بدون امتیاز)
        return [doc for doc, _ in reranked_docs]

    @staticmethod
    def _weighted_context_merge(docs):
        """ترکیب قطعات متن با وزن‌دهی بر اساس امتیاز آنها"""
        context_parts = []

        for i, doc in enumerate(docs):
            # استفاده از امتیاز محاسبه شده در مرحله امتیازدهی مجدد
            weight = doc.metadata.get('rerank_score', 1.0 - (i * 0.1))  # اگر امتیاز نباشد، از ترتیب استفاده می‌کنیم

            # افزودن اطلاعات اهمیت نسبی هر قطعه
            importance_marker = "#" * int(weight * 5)  # تعداد # نشان‌دهنده اهمیت است

            context_part = f"{importance_marker} متن منبع {i + 1} ({os.path.basename(doc.metadata.get('source', 'منبع ناشناخته'))}):\n{doc.page_content}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def query(self, question: str, top_k: int = 4, progress_callback=None) -> str | dict[
        str, list[dict[str, Any]] | Any]:
        try:
            if progress_callback:
                progress_callback(0.2, "در حال جستجوی اطلاعات مرتبط...")
            # بازیابی اسناد مرتبط با تعداد بیشتر برای امتیازدهی مجدد
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k * 2,  # بازیابی تعداد بیشتری برای انتخاب بهتر
                    "fetch_k": top_k * 3,  # تعداد بیشتری برای محاسبه MMR
                    "lambda_mult": 0.7  # تأکید بیشتر بر ارتباط نسبت به تنوع
                }
            )

            # بازیابی اسناد و ذخیره امتیازها
            initial_docs = retriever.invoke(question)
            if not initial_docs:
                return "اطلاعات مرتبطی با پرسش شما یافت نشد."

            # اطمینان از اینکه امتیاز اولیه در متادیتا ذخیره شده است
            # در این مرحله ممکن است امتیاز اولیه در متادیتا وجود داشته باشد یا نه
            # برای اطمینان، یک مقدار پیش‌فرض اضافه می‌کنیم اگر وجود نداشت
            for i, doc in enumerate(initial_docs):
                if 'score' not in doc.metadata:
                    # اگر امتیازی وجود ندارد، یک امتیاز بر اساس ترتیب دریافت اختصاص می‌دهیم
                    # (دقیق نیست اما بهتر از صفر است)
                    doc.metadata['score'] = 1.0 - (i * 0.1)

            if progress_callback:
                progress_callback(0.4, "در حال امتیازدهی به اسناد بازیابی شده...")

            # امتیازدهی مجدد به اسناد برای تعیین اهمیت آنها
            reranked_docs = self._rerank_documents(question, initial_docs)

            # انتخاب بهترین اسناد
            top_docs = reranked_docs[:top_k]

            if progress_callback:
                progress_callback(0.5, "در حال آماده‌سازی متن زمینه...")

            # خلاصه‌سازی اسناد طولانی (اختیاری)
            # processed_docs = self._summarize_long_documents(top_docs)
            processed_docs = top_docs  # برای سادگی، این مرحله را فعلاً نادیده می‌گیریم

            # ترکیب متن‌ها با وزن‌دهی و نشان‌گذاری اهمیت آنها
            context = self._weighted_context_merge(processed_docs)
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

            # تهیه منابع با نمایش امتیاز ارتباط
            if progress_callback:
                progress_callback(0.9, "در حال تهیه منابع...")

            sources = []
            for doc in top_docs:
                # استفاده از rerank_score به جای score برای نمایش امتیاز ارتباط
                source_info = {
                    "title": doc.metadata.get("title", "بدون عنوان"),
                    "score": round(doc.metadata.get("rerank_score", 0.0), 2),  # استفاده از rerank_score با دو رقم اعشار
                    "source": doc.metadata.get("source", "نامشخص")
                }
                sources.append(source_info)

            # افزودن منابع به پاسخ
            final_answer = {
                "answer": answer,
                "sources": sources
            }

            if progress_callback:
                progress_callback(1.0, "پاسخ آماده است.")

            return final_answer

        except Exception as e:
            print(f"خطا در فرآیند پرسش و پاسخ: {str(e)}")
            return "متأسفانه خطایی در پردازش پرسش شما رخ داد. لطفاً دوباره تلاش کنید."

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
