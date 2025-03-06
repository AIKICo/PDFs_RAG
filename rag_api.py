import os
import sqlite3
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pydantic import BaseModel

# Configuration
DB_PATH = "pdf_database.db"
VECTOR_STORE_PATH = "chroma_db"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_EMBEDDINGS_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 4

app = FastAPI(
    title="RAG PDF Question Answering API",
    description="API for querying PDF documents using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for request and response
class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    llm_model: str = DEFAULT_LLM_MODEL
    embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    execution_time: float


class StatusResponse(BaseModel):
    status: str
    file_count: int
    chunk_count: Optional[int] = None
    message: str = ""


class RAGProcessor:
    def __init__(self, llm_model: str = DEFAULT_LLM_MODEL, embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL):
        """
        Initialize the RAG processor with specified models.

        Args:
            llm_model: Ollama model name for generating responses
            embeddings_model: Ollama model name for embeddings
        """
        self.llm_model = llm_model
        self.embeddings_model = embeddings_model
        self.db_conn = self._init_database()

        # Initialize vector store if it exists
        if os.path.exists(VECTOR_STORE_PATH):
            self.ollama_embeddings = OllamaEmbeddings(model=embeddings_model)
            self.vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH,
                embedding_function=self.ollama_embeddings
            )
        else:
            self.vector_store = None

    @staticmethod
    def _init_database() -> sqlite3.Connection:
        """Initialize SQLite database connection."""
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database file {DB_PATH} not found. Please run the PDF processor first.")

        return sqlite3.connect(DB_PATH)

    def query(self, question: str, top_k: int = 4) -> Dict:
        """Query the system with a question and get a response."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized or empty. Please process PDF files first.")

        # Initialize Ollama for response generation
        llm = OllamaLLM(model=self.llm_model)

        # Create a retrieval QA chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Execute the query
        start_time = time.time()
        result = qa_chain.invoke({"query": question})
        execution_time = time.time() - start_time

        answer = result["result"]
        sources = result["source_documents"]

        # Format sources for display
        source_details = []
        for doc in sources:
            source_file = doc.metadata.get("source_file", "Unknown source")
            page_num = doc.metadata.get("page", "Unknown page")

            source_details.append({
                "filename": os.path.basename(source_file),
                "filepath": source_file,
                "page": page_num,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })

        # Format and return the response
        response = {
            "answer": answer,
            "sources": source_details,
            "execution_time": execution_time
        }

        return response

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

    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the vector store."""
        if not self.vector_store:
            return 0
        return len(self.vector_store.get()["ids"])

    def close(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()


# Global processor instance
_processor = None


# Dependency to get processor instance
def get_processor():
    global _processor
    if _processor is None:
        _processor = RAGProcessor()
    return _processor


@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest, default_processor: RAGProcessor = Depends(get_processor)):
    """
    Query the knowledge base with a question.

    Returns the answer and relevant sources.
    """
    try:
        # Create processor with specified models if different from default
        if (query_request.llm_model != DEFAULT_LLM_MODEL or
                query_request.embeddings_model != DEFAULT_EMBEDDINGS_MODEL):
            processor = RAGProcessor(
                llm_model=query_request.llm_model,
                embeddings_model=query_request.embeddings_model
            )
            custom_processor = True
        else:
            processor = default_processor
            custom_processor = False

        result = processor.query(
            question=query_request.question,
            top_k=query_request.top_k
        )

        # Close if we created a new processor instance
        if custom_processor:
            processor.close()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status(processor: RAGProcessor = Depends(get_processor)):
    """Get system status including number of processed files and vector store status."""
    try:
        files = processor.list_processed_files()
        chunk_count = processor.get_chunk_count()

        if not files:
            return {
                "status": "not_ready",
                "file_count": 0,
                "chunk_count": 0,
                "message": "No processed files found. Please process PDF files first."
            }

        if chunk_count == 0:
            return {
                "status": "partial",
                "file_count": len(files),
                "chunk_count": 0,
                "message": "Files processed but vector store is empty or not properly initialized."
            }

        return {
            "status": "ready",
            "file_count": len(files),
            "chunk_count": chunk_count,
            "message": "System is ready to answer queries."
        }
    except Exception as e:
        return {
            "status": "error",
            "file_count": 0,
            "message": f"Error checking system status: {str(e)}"
        }


@app.get("/files")
async def list_files(processor: RAGProcessor = Depends(get_processor)):
    """List all processed files in the database."""
    try:
        files = processor.list_processed_files()
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Proper shutdown
@app.on_event("shutdown")
def shutdown_event():
    global _processor
    if _processor is not None:
        _processor.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)