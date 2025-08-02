"""Tests for the VectorStore class."""

import tempfile
from pathlib import Path

import pytest

from tool_smith_mcp.utils.vector_store import VectorStore


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def vector_store(temp_db_path: Path) -> VectorStore:
    """Create a VectorStore instance for testing."""
    return VectorStore(temp_db_path)


@pytest.mark.asyncio
async def test_add_and_search_document(vector_store: VectorStore) -> None:
    """Test adding and searching documents."""
    # Add a document
    doc_id = "test_tool"
    content = "Calculate mathematical expressions"
    metadata = {"function": "calculate", "type": "math"}

    await vector_store.add_document(doc_id, content, metadata)

    # Search for similar content
    results = await vector_store.search("math calculation", top_k=1)

    assert len(results) == 1
    assert results[0][0] == doc_id
    assert results[0][2] == content
    assert results[0][3] == metadata


@pytest.mark.asyncio
async def test_search_similarity_threshold(vector_store: VectorStore) -> None:
    """Test search with similarity threshold."""
    # Add documents
    await vector_store.add_document("math_tool", "Calculate mathematical expressions")
    await vector_store.add_document("text_tool", "Format and manipulate text")

    # Search with high similarity threshold
    results = await vector_store.search("text formatting", min_similarity=0.8)

    # Should find the text tool
    assert len(results) >= 1
    assert any(result[0] == "text_tool" for result in results)


@pytest.mark.asyncio
async def test_get_document(vector_store: VectorStore) -> None:
    """Test getting a specific document."""
    doc_id = "test_doc"
    content = "Test document content"
    metadata = {"test": "data"}

    await vector_store.add_document(doc_id, content, metadata)

    result = await vector_store.get_document(doc_id)
    assert result is not None
    assert result[0] == content
    assert result[1] == metadata

    # Test non-existent document
    result = await vector_store.get_document("non_existent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_document(vector_store: VectorStore) -> None:
    """Test deleting a document."""
    doc_id = "to_delete"
    await vector_store.add_document(doc_id, "Content to delete")

    # Verify document exists
    result = await vector_store.get_document(doc_id)
    assert result is not None

    # Delete document
    success = await vector_store.delete_document(doc_id)
    assert success is True

    # Verify document is gone
    result = await vector_store.get_document(doc_id)
    assert result is None


@pytest.mark.asyncio
async def test_list_documents(vector_store: VectorStore) -> None:
    """Test listing all documents."""
    # Add multiple documents
    docs = [
        ("doc1", "First document", {"type": "first"}),
        ("doc2", "Second document", {"type": "second"}),
    ]

    for doc_id, content, metadata in docs:
        await vector_store.add_document(doc_id, content, metadata)

    # List documents
    documents = await vector_store.list_documents()

    assert len(documents) == 2
    doc_ids = [doc[0] for doc in documents]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


@pytest.mark.asyncio
async def test_clear_vector_store(vector_store: VectorStore) -> None:
    """Test clearing the vector store."""
    # Add documents
    await vector_store.add_document("doc1", "Content 1")
    await vector_store.add_document("doc2", "Content 2")

    # Verify documents exist
    documents = await vector_store.list_documents()
    assert len(documents) == 2

    # Clear store
    await vector_store.clear()

    # Verify store is empty
    documents = await vector_store.list_documents()
    assert len(documents) == 0
