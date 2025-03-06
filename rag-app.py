import os
import argparse
from typing import List, Dict, Any

# برای کار با PDF ها
from langchain_community.document_loaders import PyPDFLoader
# برای تقسیم متن ها
from langchain.text_splitter import RecursiveCharacterTextSplitter
# برای ساخت embedding
from langchain_community.embeddings import OllamaEmbeddings
# استفاده از Chroma به عنوان پایگاه داده برداری
from langchain_community.vectorstores import Chroma
# برای LLM 
from langchain_community.llms import Ollama
# برای ساخت زنجیره پرسش و پاسخ
from langchain.chains import RetrievalQA
# برای بخش interactive CLI
from rich.console import Console
from rich.markdown import Markdown

class RAGSystem:
    def __init__(self, 
                 model_name: str = "llama2", 
                 persist_directory: str = "db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100):
        """
        مقداردهی اولیه سیستم RAG
        
        Args:
            model_name: نام مدل Ollama برای استفاده
            persist_directory: مسیر ذخیره‌سازی پایگاه داده
            chunk_size: اندازه هر چانک متن
            chunk_overlap: میزان همپوشانی چانک‌ها
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.console = Console()
        
        # ساخت embedding با استفاده از Ollama
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # بررسی وجود پایگاه داده
        if os.path.exists(persist_directory):
            self.console.print(f"[green]پایگاه داده در {persist_directory} یافت شد. بارگذاری...[/green]")
            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            self.console.print(f"[yellow]پایگاه داده‌ای یافت نشد. لطفاً از دستور 'index' استفاده کنید.[/yellow]")
            self.db = None
            
        # تنظیم LLM
        self.llm = Ollama(model=model_name)
        
        # تنظیم text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # ساخت زنجیره پرسش و پاسخ اگر پایگاه داده موجود باشد
        if self.db is not None:
            self.qa_chain = self._create_qa_chain()
            
    def _create_qa_chain(self) -> RetrievalQA:
        """ساخت زنجیره پرسش و پاسخ"""
        retriever = self.db.as_retriever(search_kwargs={"k": 5})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def index_pdfs(self, pdf_directory: str) -> None:
        """
        ایندکس کردن فایل‌های PDF داخل دایرکتوری مشخص شده
        
        Args:
            pdf_directory: مسیر دایرکتوری حاوی فایل‌های PDF
        """
        self.console.print(f"[blue]شروع پردازش فایل‌های PDF در {pdf_directory}...[/blue]")
        
        # بررسی وجود دایرکتوری
        if not os.path.exists(pdf_directory):
            self.console.print(f"[red]دایرکتوری {pdf_directory} یافت نشد![/red]")
            return
        
        # جمع‌آوری همه فایل‌های PDF
        pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            self.console.print(f"[red]هیچ فایل PDF در {pdf_directory} یافت نشد![/red]")
            return
        
        all_chunks = []
        
        # پردازش هر فایل PDF
        for pdf_file in pdf_files:
            self.console.print(f"[blue]پردازش {pdf_file}...[/blue]")
            
            try:
                # بارگذاری PDF
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                
                # تقسیم به چانک‌های کوچکتر
                chunks = self.text_splitter.split_documents(documents)
                
                self.console.print(f"[green]{len(chunks)} چانک از {pdf_file} استخراج شد.[/green]")
                all_chunks.extend(chunks)
                
            except Exception as e:
                self.console.print(f"[red]خطا در پردازش {pdf_file}: {str(e)}[/red]")
        
        # ساخت و ذخیره پایگاه داده برداری
        if all_chunks:
            self.console.print(f"[blue]ساخت پایگاه داده برداری با {len(all_chunks)} چانک...[/blue]")
            self.db = Chroma.from_documents(
                documents=all_chunks, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.db.persist()
            self.console.print(f"[green]پایگاه داده با موفقیت در {self.persist_directory} ذخیره شد.[/green]")
            
            # بروزرسانی زنجیره پرسش و پاسخ
            self.qa_chain = self._create_qa_chain()
        else:
            self.console.print("[red]هیچ چانکی برای ایندکس کردن یافت نشد![/red]")
    
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
        
        try:
            result = self.qa_chain({"query": query})
            return result
        except Exception as e:
            self.console.print(f"[red]خطا در پاسخ به پرسش: {str(e)}[/red]")
            return {"result": f"خطا: {str(e)}"}
    
    def interactive(self) -> None:
        """اجرای حالت تعاملی برای پرسش و پاسخ"""
        if self.db is None or self.qa_chain is None:
            self.console.print("[red]پایگاه داده هنوز ایجاد نشده است. لطفاً ابتدا از دستور 'index' استفاده کنید.[/red]")
            return
        
        self.console.print("[green]حالت تعاملی. برای خروج 'exit' یا 'quit' را وارد کنید.[/green]")
        
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
                self.console.print(f"[bold cyan]منبع {i+1}:[/bold cyan] {doc.metadata.get('source', 'نامشخص')}, صفحه {doc.metadata.get('page', 'نامشخص')}")
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
    index_parser.add_argument("--chunk-overlap", type=int, default=100, help="همپوشانی چانک (پیش‌فرض: 100)")
    
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
            chunk_overlap=args.chunk_overlap
        )
        rag_system.index_pdfs(args.pdf_dir)
        
    elif args.command == "ask":
        rag_system = RAGSystem(model_name=args.model, persist_directory=args.db_dir)
        result = rag_system.ask(args.query)
        
        console = Console()
        console.print("\n[bold blue]پاسخ:[/bold blue]")
        console.print(Markdown(result['result']))
        
        console.print("\n[bold green]منابع:[/bold green]")
        for i, doc in enumerate(result['source_documents']):
            console.print(f"[bold cyan]منبع {i+1}:[/bold cyan] {doc.metadata.get('source', 'نامشخص')}, صفحه {doc.metadata.get('page', 'نامشخص')}")
            console.print(f"[dim]{doc.page_content[:150]}...[/dim]\n")
            
    elif args.command == "interactive":
        rag_system = RAGSystem(model_name=args.model, persist_directory=args.db_dir)
        rag_system.interactive()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
