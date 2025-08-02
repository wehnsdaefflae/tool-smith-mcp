"""Vector store for indexing and searching tool descriptions."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .cache import SimpleCache, embedding_cache_key

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for tool descriptions using ChromaDB and sentence transformers."""

    def __init__(
        self,
        db_path: Path,
        collection_name: str = "tool_descriptions",
        model_name: str = "all-MiniLM-L6-v2",
        cache: Optional[SimpleCache] = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection to use
            model_name: Sentence transformer model name
            cache: Optional cache for embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.cache = cache

        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Tool descriptions for semantic search"},
        )

        logger.info(f"Initialized vector store at {db_path}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with optional caching."""
        if self.cache:
            cache_key = embedding_cache_key(text)
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Generate embedding
        embedding = self.encoder.encode(text).tolist()

        # Cache if available
        if self.cache:
            cache_key = embedding_cache_key(text)
            self.cache.set(cache_key, embedding)

        return embedding

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the vector store.

        Args:
            doc_id: Unique identifier for the document
            content: Text content to index
            metadata: Optional metadata to store with the document
        """
        try:
            # Generate embedding with caching
            embedding = self._get_embedding(content)

            # Add to collection
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}],
            )

            logger.debug(f"Added document {doc_id} to vector store")

        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Search for similar documents.

        Args:
            query: Search query
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of tuples: (doc_id, similarity_score, content, metadata)
        """
        try:
            # Generate query embedding with caching
            query_embedding = self._get_embedding(query)

            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            # Process results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Calculate similarity score (ChromaDB returns distances)
                    distance = (
                        results["distances"][0][i] if results["distances"] else 0.0
                    )
                    similarity = 1.0 - distance  # Convert distance to similarity

                    if similarity >= min_similarity:
                        content = (
                            results["documents"][0][i] if results["documents"] else ""
                        )
                        metadata = (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        )

                        search_results.append((doc_id, similarity, content, metadata))

            # Sort by similarity (highest first)
            search_results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(
                f"Found {len(search_results)} results for query: {query[:50]}..."
            )
            return search_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if document was deleted, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    async def get_document(self, doc_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get a specific document by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Tuple of (content, metadata) if found, None otherwise
        """
        try:
            results = self.collection.get(ids=[doc_id])

            if results["ids"] and results["ids"][0]:
                content = results["documents"][0] if results["documents"] else ""
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                return (content, metadata)

            return None

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None

    async def list_documents(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """List all documents in the vector store.

        Returns:
            List of tuples: (doc_id, content, metadata)
        """
        try:
            results = self.collection.get()

            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    content = results["documents"][i] if results["documents"] else ""
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    documents.append((doc_id, content, metadata))

            return documents

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    async def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Tool descriptions for semantic search"},
            )
            logger.info("Cleared vector store")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
