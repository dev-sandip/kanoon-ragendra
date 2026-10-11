import os
import pickle
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        # Use an absolute path to avoid working-directory issues.
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(embedding_model)

        self.faiss_path = os.path.join(
            self.persist_dir, "faiss.index"
        )
        self.metadata_path = os.path.join(
            self.persist_dir, "metadata.pkl"
        )

        print(f"[INFO] Embedding model: {embedding_model}")
        print(f"[INFO] Persistence directory: {self.persist_dir}")

    def build_from_documents(self, documents: List[Any]):
        if not documents:
            raise ValueError("No documents provided to build the index.")

        print(f"[INFO] Processing {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)

        if not chunks:
            raise ValueError("Document processing produced no chunks.")

        embeddings = np.asarray(
            emb_pipe.embed_chunks(chunks), dtype=np.float32
        )

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Invalid embedding dimensions or chunk count.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        # Normalize vectors so cosine similarity can be used with
        # an inner-product FAISS index.
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = [
            {"text": chunk.page_content}
            for chunk in chunks
        ]

        self.save()

        print(
            f"[INFO] Built index with {self.index.ntotal} vectors."
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None,
    ):
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        if metadatas is not None and len(metadatas) != len(embeddings):
            raise ValueError("Number of metadata entries must match vectors.")

        # Use the same similarity metric as build_from_documents().
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        elif self.index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimension does not match index.")

        self.index.add(embeddings)

        if metadatas is not None:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {len(embeddings)} vectors.")

    def save(self):
        if self.index is None:
            raise ValueError("Cannot save: index has not been built.")

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index and metadata counts do not match.")

        faiss.write_index(self.index, self.faiss_path)

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Index saved: {self.faiss_path}")
        print(f"[INFO] Metadata saved: {self.metadata_path}")

    def load(self):
        if not os.path.isfile(self.faiss_path):
            raise FileNotFoundError(
                f"FAISS index not found: {self.faiss_path}. "
                "Build the index first."
            )

        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}."
            )

        index = faiss.read_index(self.faiss_path)

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        if index.ntotal != len(metadata):
            raise ValueError(
                f"Index contains {index.ntotal} vectors, but "
                f"metadata contains {len(metadata)} entries."
            )

        self.index = index
        self.metadata = metadata

        print(f"[INFO] Loaded {self.index.ntotal} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index is not loaded. Build or load it first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be a 1D or 2D array.")

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} "
                f"does not match index dimension {self.index.d}."
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        faiss.normalize_L2(query_embedding)

        k = min(top_k, self.index.ntotal)

        if k == 0:
            return []

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue

            results.append(
                {
                    "index": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[int(idx)],
                }
            )

        return results

    def query(self, query_text: str, top_k: int = 5):
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        print(f"[INFO] Query: {query_text}")

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
        ).astype(np.float32)

        return self.search(query_embedding, top_k=top_k)