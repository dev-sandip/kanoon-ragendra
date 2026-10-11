import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

from pathlib import Path

from fastapi.responses import FileResponse


# Global instances
store = None
rag_search = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, rag_search

    store = FaissVectorStore(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
    )

    # Load existing vector store or build a new one.
    if (
        os.path.isfile(store.faiss_path)
        and os.path.isfile(store.metadata_path)
    ):
        print("[INFO] Loading existing vector store...")
        store.load()
    else:
        print("[INFO] Vector store not found. Building a new one...")

        docs = load_all_documents("data")

        if not docs:
            raise RuntimeError(
                "No documents found in the data directory."
            )

        store.build_from_documents(docs)

    rag_search = RAGSearch()

    print("[INFO] RAG API is ready.")

    yield

    # Add cleanup logic here if your components require it.


app = FastAPI(
    title="RAG Search API",
    description="Search documents and generate summaries using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


# Request schema
class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question to search for",
        examples=[
            "How is the head of election commission selected in Nepal?"
        ],
    )
    top_k: int = Field(default=3, ge=1, le=20)


# Response schema
class SearchResponse(BaseModel):
    query: str
    top_k: int
    summary: str

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "static" / "index.html"


@app.get("/ui", include_in_schema=False)
def serve_ui():
    return FileResponse(HTML_FILE)
@app.get("/")
def home():
    return {
        "message": "RAG Search API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store_loaded": store is not None,
        "rag_search_initialized": rag_search is not None,
    }


@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    if rag_search is None:
        raise HTTPException(
            status_code=503,
            detail="RAG search is not initialized.",
        )

    try:
        summary = rag_search.search_and_summarize(
            request.query,
            top_k=request.top_k,
        )

        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            summary=str(summary),
        )

    except Exception as exc:
        # Log the full exception in production.
        print(f"[ERROR] Search failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching documents.",
        ) from exc