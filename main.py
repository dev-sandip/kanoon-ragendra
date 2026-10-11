import os

from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch


if __name__ == "__main__":
    store = FaissVectorStore(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
    )

    # Load the existing vector store or build it on the first run.
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
            raise ValueError("No documents found in the data directory.")

        store.build_from_documents(docs)

    # Initialize the RAG search pipeline.
    rag_search = RAGSearch()

    query = "How is the head of election commission selected in Nepal?"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3,
    )

    print("\nQuery:", query)
    print("\nSummary:", summary)