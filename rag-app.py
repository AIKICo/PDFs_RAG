import argparse
import concurrent.futures
import logging
import os
from typing import Dict, Any
import unicodedata

# برای ساخت زنجیره پرسش و پاسخ
from langchain.chains import RetrievalQA
# برای تقسیم متن ها
from langchain.text_splitter import RecursiveCharacterTextSplitter
# استفاده از Chroma به عنوان پایگاه داده برداری
from langchain_chroma import Chroma
# برای کار با PDF ها
from langchain_community.document_loaders import PyPDFLoader
# برای ساخت embedding
from langchain_ollama import OllamaEmbeddings  # برای LLM - استفاده از پکیج جدید
from langchain_ollama import OllamaLLM
# برای بخش interactive CLI
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress
from chromadb.config import Settings


class RAGSystem:
    def __init__(self,
                 model_name: str = "llama2",
                 persist_directory: str = "db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,  # افزایش همپوشانی برای بهبود پشتیبانی از فارسی
                 max_workers: int = 4):  # تعداد worker برای پردازش موازی
        """
        مقداردهی اولیه سیستم RAG

        Args:
            model_name: نام مدل Ollama برای استفاده
            persist_directory: مسیر ذخیره‌سازی پایگاه داده
            chunk_size: اندازه هر چانک متن
            chunk_overlap: میزان همپوشانی چانک‌ها
            max_workers: تعداد پردازنده‌های همزمان برای پردازش PDF
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.console = Console()

        # تنظیم logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("rag_system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RAGSystem")

        # ساخت embedding با استفاده از Ollama
        try:
            self.embeddings = OllamaEmbeddings(model=model_name)
        except Exception as e:
            self.logger.error(f"خطا در راه‌اندازی مدل embedding: {str(e)}")
            self.console.print(f"[red]خطا در راه‌اندازی مدل embedding: {str(e)}[/red]")
            raise

        # بررسی وجود پایگاه داده
        if os.path.exists(persist_directory):
            self.console.print(f"[green]پایگاه داده در {persist_directory} یافت شد. بارگذاری...[/green]")
            try:
                self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            except Exception as e:
                self.logger.error(f"خطا در بارگذاری پایگاه داده: {str(e)}")
                self.console.print(f"[red]خطا در بارگذاری پایگاه داده: {str(e)}[/red]")
                self.db = None
        else:
            self.console.print(f"[yellow]پایگاه داده‌ای یافت نشد. لطفاً از دستور 'index' استفاده کنید.[/yellow]")
            self.db = None

        # تنظیم LLM - استفاده از کلاس جدید OllamaLLM
        try:
            self.llm = OllamaLLM(
                model=model_name,
                temperature=0.1,  # کاهش دمای مدل برای پاسخ‌های دقیق‌تر
                num_predict=512,  # تعداد توکن‌های خروجی
            )
        except Exception as e:
            self.logger.error(f"خطا در راه‌اندازی مدل LLM: {str(e)}")
            self.console.print(f"[red]خطا در راه‌اندازی مدل LLM: {str(e)}[/red]")
            raise

        # تنظیم text splitter با تنظیمات بهتر برای متون فارسی
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", "؟", "!", "،", ";", ":", " ", ""],  # افزودن جداکننده‌های فارسی
        )

        # ساخت زنجیره پرسش و پاسخ اگر پایگاه داده موجود باشد
        if self.db is not None:
            self.qa_chain = self._create_qa_chain()
        else:
            self.qa_chain = None

    def _create_qa_chain(self) -> RetrievalQA:
        """ساخت زنجیره پرسش و پاسخ"""
        try:
            # افزایش تعداد نتایج بازیابی شده برای پوشش بهتر محتوا
            retriever = self.db.as_retriever(search_kwargs={"k": 8})
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"خطا در ساخت زنجیره پرسش و پاسخ: {str(e)}")
            self.console.print(f"[red]خطا در ساخت زنجیره پرسش و پاسخ: {str(e)}[/red]")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing invalid Unicode characters and normalizing

        Args:
            text: Input text to clean

        Returns:
            Cleaned text string
        """
        # Remove surrogate characters
        text = ''.join(char for char in text if not '\ud800' <= char <= '\udfff')

        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)

        # Replace any remaining problematic characters with space
        text = ''.join(char if ord(char) < 0x10000 else ' ' for char in text)

        return text

    def _process_pdf_file(self, pdf_file: str) -> list:
        """
        Process a PDF file and convert it to chunks with text cleaning
        """
        try:
            self.logger.info(f"شروع پردازش {pdf_file}")
            loader = PyPDFLoader(pdf_file, extract_images=False)
            documents = loader.load()

            if not documents:
                self.logger.warning(f"سند {pdf_file} خالی است یا قابل پردازش نیست.")
                return []

            # Clean text in documents
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)

            chunks = self.text_splitter.split_documents(documents)
            self.logger.info(f"{len(chunks)} چانک از {pdf_file} استخراج شد.")
            return chunks

        except Exception as e:
            self.logger.error(f"خطا در پردازش {pdf_file}: {str(e)}")
            return []

    def index_pdfs(self, pdf_directory: str, batch_size: int = 1000) -> None:
        """
        ایندکس کردن فایل‌های PDF داخل دایرکتوری مشخص شده

        Args:
            pdf_directory: مسیر دایرکتوری حاوی فایل‌های PDF
            batch_size: تعداد چانک‌ها در هر batch برای مدیریت بهتر حافظه
        """
        self.console.print(f"[blue]شروع پردازش فایل‌های PDF در {pdf_directory}...[/blue]")
        self.logger.info(f"شروع پردازش فایل‌های PDF در {pdf_directory}")

        # بررسی وجود دایرکتوری
        if not os.path.exists(pdf_directory):
            self.console.print(f"[red]دایرکتوری {pdf_directory} یافت نشد![/red]")
            self.logger.error(f"دایرکتوری {pdf_directory} یافت نشد")
            return

        # جمع‌آوری همه فایل‌های PDF
        pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory)
                     if f.lower().endswith('.pdf')]

        if not pdf_files:
            self.console.print(f"[red]هیچ فایل PDF در {pdf_directory} یافت نشد![/red]")
            self.logger.error(f"هیچ فایل PDF در {pdf_directory} یافت نشد")
            return

        self.console.print(f"[blue]تعداد {len(pdf_files)} فایل PDF یافت شد.[/blue]")
        self.logger.info(f"تعداد {len(pdf_files)} فایل PDF یافت شد")

        # پردازش موازی فایل‌های PDF
        all_chunks = []

        with Progress() as progress:
            task = progress.add_task("[cyan]پردازش PDF ها...", total=len(pdf_files))

            # استفاده از ThreadPoolExecutor برای پردازش موازی
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {executor.submit(self._process_pdf_file, pdf_file): pdf_file for pdf_file in pdf_files}

                for future in concurrent.futures.as_completed(future_to_pdf):
                    pdf_file = future_to_pdf[future]
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                        progress.update(task, advance=1)
                    except Exception as e:
                        self.logger.error(f"خطا در پردازش {pdf_file}: {str(e)}")

        # ساخت و ذخیره پایگاه داده برداری
        if all_chunks:
            self.console.print(f"[blue]ساخت پایگاه داده برداری با {len(all_chunks)} چانک...[/blue]")
            self.logger.info(f"ساخت پایگاه داده برداری با {len(all_chunks)} چانک")

            # پردازش batch-by-batch برای مدیریت بهتر حافظه
            try:
                # اگر پایگاه داده موجود نباشد، آن را ایجاد می‌کنیم
                if not os.path.exists(self.persist_directory):
                    # ابتدا با یک batch کوچک پایگاه داده را ایجاد می‌کنیم
                    first_batch = all_chunks[:min(batch_size, len(all_chunks))]
                    self.console.print(f"[blue]ایجاد پایگاه داده اولیه با {len(first_batch)} چانک...[/blue]")

                    self.db = Chroma.from_documents(
                        documents=first_batch,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    self.db = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings,
                        collection_name="documents",
                        client_settings=Settings(
                            persist_directory=self.persist_directory,
                            is_persistent=True,
                            anonymized_telemetry=False
                        )
                    )

                    # اگر چانک بیشتری وجود دارد، آنها را به صورت batch اضافه می‌کنیم
                    remaining_chunks = all_chunks[min(batch_size, len(all_chunks)):]
                else:
                    # اگر پایگاه داده موجود باشد، همه چانک‌ها را batch-by-batch اضافه می‌کنیم
                    self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                    remaining_chunks = all_chunks

                # افزودن چانک‌های باقیمانده به صورت batch
                with Progress() as progress:
                    # محاسبه تعداد batch‌ها
                    total_batches = (len(remaining_chunks) + batch_size - 1) // batch_size
                    task = progress.add_task("[green]افزودن به پایگاه داده...", total=total_batches)

                    for i in range(0, len(remaining_chunks), batch_size):
                        batch = remaining_chunks[i:i + batch_size]
                        self.db.add_documents(documents=batch)
                        self.db = Chroma(
                            persist_directory=self.persist_directory,
                            embedding_function=self.embeddings,
                            collection_name="documents",
                            client_settings=Settings(
                                persist_directory=self.persist_directory,
                                is_persistent=True,
                                anonymized_telemetry=False
                            )
                        )
                        progress.update(task, advance=1)
                        self.logger.info(f"Batch {i // batch_size + 1}/{total_batches} اضافه شد")

                self.console.print(f"[green]پایگاه داده با موفقیت در {self.persist_directory} ذخیره شد.[/green]")
                self.logger.info(f"پایگاه داده با موفقیت در {self.persist_directory} ذخیره شد")

                # بروزرسانی زنجیره پرسش و پاسخ
                self.qa_chain = self._create_qa_chain()

            except Exception as e:
                self.logger.error(f"خطا در ساخت پایگاه داده: {str(e)}")
                self.console.print(f"[red]خطا در ساخت پایگاه داده: {str(e)}[/red]")
        else:
            self.console.print("[red]هیچ چانکی برای ایندکس کردن یافت نشد![/red]")
            self.logger.error("هیچ چانکی برای ایندکس کردن یافت نشد")

    def ask(self, query: str) -> Dict[str, Any]:
        """
        پاسخ به پرسش با استفاده از پایگاه داده

        Args:
            query: پرسش کاربر

        Returns:
            Dict: نتیجه پرسش و پاسخ
        """
        if self.db is None or self.qa_chain is None:
            self.console.print("[red]پایگاه داده هنوز ایجاد نشده است. لطفاً ابتدا از دستور 'index' استفاده کنید.[/red]")
            return {"result": "پایگاه داده یافت نشد"}

        self.console.print(f"[blue]پاسخ به پرسش: {query}[/blue]")
        self.logger.info(f"پاسخ به پرسش: {query}")

        try:
            result = self.qa_chain({"query": query})
            return result
        except Exception as e:
            self.logger.error(f"خطا در پاسخ به پرسش: {str(e)}")
            self.console.print(f"[red]خطا در پاسخ به پرسش: {str(e)}[/red]")
            return {"result": f"خطا: {str(e)}"}

    def interactive(self) -> None:
        """اجرای حالت تعاملی برای پرسش و پاسخ"""
        if self.db is None or self.qa_chain is None:
            self.console.print("[red]پایگاه داده هنوز ایجاد نشده است. لطفاً ابتدا از دستور 'index' استفاده کنید.[/red]")
            return

        self.console.print("[green]حالت تعاملی. برای خروج 'exit' یا 'quit' یا 'خروج' را وارد کنید.[/green]")

        while True:
            query = input("\nپرسش خود را وارد کنید: ")

            if query.lower() in ['exit', 'quit', 'خروج']:
                break

            if not query.strip():
                continue

            result = self.ask(query)

            # نمایش پاسخ
            self.console.print("\n[bold blue]پاسخ:[/bold blue]")
            self.console.print(Markdown(result['result']))

            # نمایش منابع
            self.console.print("\n[bold green]منابع:[/bold green]")
            for i, doc in enumerate(result['source_documents']):
                self.console.print(
                    f"[bold cyan]منبع {i + 1}:[/bold cyan] {doc.metadata.get('source', 'نامشخص')}, صفحه {doc.metadata.get('page', 'نامشخص')}")
                self.console.print(f"[dim]{doc.page_content[:150]}...[/dim]\n")


def main():
    """تابع اصلی برنامه"""
    parser = argparse.ArgumentParser(description="سیستم RAG برای فایل‌های PDF با استفاده از Ollama")

    subparsers = parser.add_subparsers(dest="command", help="دستور مورد نظر")

    # دستور index
    index_parser = subparsers.add_parser("index", help="ایندکس کردن فایل‌های PDF")
    index_parser.add_argument("pdf_dir", help="مسیر دایرکتوری حاوی فایل‌های PDF")
    index_parser.add_argument("--model", default="llama2", help="نام مدل Ollama (پیش‌فرض: llama2)")
    index_parser.add_argument("--db-dir", default="db", help="مسیر ذخیره‌سازی پایگاه داده (پیش‌فرض: db)")
    index_parser.add_argument("--chunk-size", type=int, default=1000, help="اندازه چانک (پیش‌فرض: 1000)")
    index_parser.add_argument("--chunk-overlap", type=int, default=200, help="همپوشانی چانک (پیش‌فرض: 200)")
    index_parser.add_argument("--batch-size", type=int, default=1000, help="تعداد چانک در هر batch (پیش‌فرض: 1000)")
    index_parser.add_argument("--workers", type=int, default=4, help="تعداد پردازنده‌های همزمان (پیش‌فرض: 4)")

    # دستور ask
    ask_parser = subparsers.add_parser("ask", help="پرسش از سیستم")
    ask_parser.add_argument("query", help="پرسش مورد نظر")
    ask_parser.add_argument("--model", default="llama2", help="نام مدل Ollama (پیش‌فرض: llama2)")
    ask_parser.add_argument("--db-dir", default="db", help="مسیر پایگاه داده (پیش‌فرض: db)")

    # دستور interactive
    interactive_parser = subparsers.add_parser("interactive", help="حالت تعاملی برای پرسش و پاسخ")
    interactive_parser.add_argument("--model", default="llama2", help="نام مدل Ollama (پیش‌فرض: llama2)")
    interactive_parser.add_argument("--db-dir", default="db", help="مسیر پایگاه داده (پیش‌فرض: db)")

    args = parser.parse_args()

    if args.command == "index":
        rag_system = RAGSystem(
            model_name=args.model,
            persist_directory=args.db_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_workers=args.workers
        )
        rag_system.index_pdfs(args.pdf_dir, batch_size=args.batch_size)

    elif args.command == "ask":
        rag_system = RAGSystem(model_name=args.model, persist_directory=args.db_dir)
        result = rag_system.ask(args.query)

        console = Console()
        console.print("\n[bold blue]پاسخ:[/bold blue]")
        console.print(Markdown(result['result']))

        console.print("\n[bold green]منابع:[/bold green]")
        for i, doc in enumerate(result['source_documents']):
            console.print(
                f"[bold cyan]منبع {i + 1}:[/bold cyan] {doc.metadata.get('source', 'نامشخص')}, صفحه {doc.metadata.get('page', 'نامشخص')}")
            console.print(f"[dim]{doc.page_content[:150]}...[/dim]\n")

    elif args.command == "interactive":
        rag_system = RAGSystem(model_name=args.model, persist_directory=args.db_dir)
        rag_system.interactive()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
