"""Isolated unit tests for VectorStore without MCP server dependencies.

These tests demonstrate how to test the VectorStore module independently
of the MCP server and other components.
"""

import pytest
from pathlib import Path
from tool_smith_mcp.utils.vector_store import VectorStore


class TestVectorStoreIsolated:
    """Test VectorStore functionality in isolation."""

    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, temp_vector_db_path: Path):
        """Test that VectorStore initializes correctly."""
        vector_store = VectorStore(
            db_path=temp_vector_db_path,
            collection_name="test_collection",
            model_name="all-MiniLM-L6-v2"
        )
        
        # Should initialize without errors
        await vector_store.initialize()
        
        # Verify properties
        assert vector_store.db_path == temp_vector_db_path
        assert vector_store.collection_name == "test_collection"
        assert vector_store.model_name == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_add_and_search_documents(self, temp_vector_db_path: Path):
        """Test adding documents and searching for them."""
        vector_store = VectorStore(
            db_path=temp_vector_db_path,
            collection_name="test_docs"
        )
        await vector_store.initialize()
        
        # Add some test documents
        await vector_store.add_document(
            doc_id="math_tool",
            content="Calculate mathematical expressions and equations",
            metadata={"type": "math", "category": "calculation"}
        )
        
        await vector_store.add_document(
            doc_id="text_tool", 
            content="Format and manipulate text strings",
            metadata={"type": "text", "category": "formatting"}
        )
        
        # Search for math-related content
        results = await vector_store.search(
            query="calculate numbers and math",
            top_k=2
        )
        
        # Should find the math tool with high similarity
        assert len(results) > 0
        assert results[0][0] == "math_tool"  # doc_id
        assert results[0][1] > 0.5  # similarity score should be reasonable

    @pytest.mark.asyncio
    async def test_search_empty_database(self, temp_vector_db_path: Path):
        """Test searching in an empty database."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Search in empty database
        results = await vector_store.search("any query")
        
        # Should return empty results
        assert results == []

    @pytest.mark.asyncio
    async def test_document_metadata_retrieval(self, temp_vector_db_path: Path):
        """Test that document metadata is properly stored and retrieved."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Add document with metadata
        metadata = {
            "author": "test",
            "version": "1.0",
            "tags": ["utility", "helper"]
        }
        
        await vector_store.add_document(
            doc_id="test_doc",
            content="Test document for metadata",
            metadata=metadata
        )
        
        # Search and verify metadata is preserved
        results = await vector_store.search("test document")
        assert len(results) > 0
        
        # The metadata should be in the result
        result_metadata = results[0][3]  # metadata is the 4th element
        assert result_metadata["author"] == "test"
        assert result_metadata["version"] == "1.0"
        assert "utility" in result_metadata["tags"]

    @pytest.mark.asyncio
    async def test_update_document(self, temp_vector_db_path: Path):
        """Test updating an existing document."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Add initial document
        await vector_store.add_document(
            doc_id="update_test",
            content="Original content",
            metadata={"version": 1}
        )
        
        # Update the document
        await vector_store.add_document(
            doc_id="update_test",
            content="Updated content with new information",
            metadata={"version": 2}
        )
        
        # Search should find the updated version
        results = await vector_store.search("updated content")
        assert len(results) > 0
        assert results[0][0] == "update_test"
        assert results[0][3]["version"] == 2

    @pytest.mark.asyncio
    async def test_delete_document(self, temp_vector_db_path: Path):
        """Test deleting documents from the vector store."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Add documents
        await vector_store.add_document(
            doc_id="keep_doc",
            content="Document to keep",
            metadata={"keep": True}
        )
        
        await vector_store.add_document(
            doc_id="delete_doc",
            content="Document to delete",
            metadata={"delete": True}
        )
        
        # Verify both documents exist
        results = await vector_store.search("document")
        assert len(results) == 2
        
        # Delete one document
        await vector_store.delete_document("delete_doc")
        
        # Verify only one document remains
        results = await vector_store.search("document")
        assert len(results) == 1
        assert results[0][0] == "keep_doc"

    @pytest.mark.asyncio
    async def test_similarity_scoring(self, temp_vector_db_path: Path):
        """Test that similarity scores make sense."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Add documents with varying similarity to our test query
        await vector_store.add_document(
            doc_id="exact_match",
            content="calculate mathematical expressions",
            metadata={"relevance": "high"}
        )
        
        await vector_store.add_document(
            doc_id="partial_match",
            content="perform calculations with numbers",
            metadata={"relevance": "medium"}
        )
        
        await vector_store.add_document(
            doc_id="no_match",
            content="format text and strings",
            metadata={"relevance": "low"}
        )
        
        # Search with math-related query
        results = await vector_store.search(
            query="calculate mathematical expressions",
            top_k=3
        )
        
        # Results should be ordered by similarity
        assert len(results) == 3
        
        # First result should be exact match with highest score
        assert results[0][0] == "exact_match"
        assert results[0][1] > results[1][1]  # Higher similarity than second
        
        # Last result should have lowest similarity
        assert results[2][1] < results[1][1]

    @pytest.mark.asyncio
    async def test_top_k_limiting(self, temp_vector_db_path: Path):
        """Test that top_k parameter limits results correctly."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Add multiple documents
        for i in range(5):
            await vector_store.add_document(
                doc_id=f"doc_{i}",
                content=f"Document number {i} with some text content",
                metadata={"index": i}
            )
        
        # Search with different top_k values
        results_all = await vector_store.search("document", top_k=10)
        results_limited = await vector_store.search("document", top_k=2)
        
        # Should respect top_k limit
        assert len(results_all) == 5  # All documents
        assert len(results_limited) == 2  # Limited to 2
        
        # Limited results should be the top 2 from the full results
        assert results_limited[0][0] == results_all[0][0]
        assert results_limited[1][0] == results_all[1][0]

    def test_vector_store_configuration(self, temp_vector_db_path: Path):
        """Test VectorStore with different configuration options."""
        # Test with custom model name
        vector_store = VectorStore(
            db_path=temp_vector_db_path,
            collection_name="custom_collection",
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        assert vector_store.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert vector_store.collection_name == "custom_collection"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_vector_db_path: Path):
        """Test that concurrent operations don't interfere with each other."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Perform concurrent add operations
        import asyncio
        
        async def add_doc(doc_id: str, content: str) -> None:
            await vector_store.add_document(
                doc_id=doc_id,
                content=content,
                metadata={"concurrent": True}
            )
        
        # Add multiple documents concurrently
        tasks = [
            add_doc(f"concurrent_{i}", f"Concurrent document {i}")
            for i in range(3)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all documents were added
        results = await vector_store.search("concurrent document")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_vector_db_path: Path):
        """Test error handling in VectorStore operations."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Test deleting non-existent document
        # Should not raise an error (graceful handling)
        await vector_store.delete_document("non_existent_doc")
        
        # Test adding document with None content
        with pytest.raises((ValueError, TypeError)):
            await vector_store.add_document(
                doc_id="invalid_doc",
                content=None,
                metadata={}
            )

    @pytest.mark.asyncio 
    async def test_persistence_across_sessions(self, temp_vector_db_path: Path):
        """Test that data persists across VectorStore sessions."""
        # First session - add data
        vector_store1 = VectorStore(db_path=temp_vector_db_path)
        await vector_store1.initialize()
        
        await vector_store1.add_document(
            doc_id="persistent_doc",
            content="This document should persist",
            metadata={"persistent": True}
        )
        
        # Second session - should see the data
        vector_store2 = VectorStore(db_path=temp_vector_db_path)  
        await vector_store2.initialize()
        
        results = await vector_store2.search("persistent document")
        assert len(results) > 0
        assert results[0][0] == "persistent_doc"
        assert results[0][3]["persistent"] is True


# Integration test with real embeddings
class TestVectorStoreIntegration:
    """Integration tests that use real embeddings."""
    
    @pytest.mark.asyncio
    async def test_real_embedding_similarity(self, temp_vector_db_path: Path):
        """Test with real embeddings to verify semantic similarity works."""
        vector_store = VectorStore(db_path=temp_vector_db_path)
        await vector_store.initialize()
        
        # Add semantically related documents
        await vector_store.add_document(
            doc_id="math_calc",
            content="Perform mathematical calculations and arithmetic operations",
            metadata={"domain": "mathematics"}
        )
        
        await vector_store.add_document(
            doc_id="text_format",
            content="Format text strings and manipulate character data",
            metadata={"domain": "text_processing"}
        )
        
        await vector_store.add_document(
            doc_id="file_ops",
            content="Read and write files to disk storage",
            metadata={"domain": "file_system"}
        )
        
        # Search for math-related query
        results = await vector_store.search("calculate numbers and do math")
        
        # Should find math_calc with highest similarity
        assert len(results) > 0
        assert results[0][0] == "math_calc"
        assert results[0][3]["domain"] == "mathematics"
        
        # Search for text-related query  
        results = await vector_store.search("process and format strings")
        
        # Should find text_format with highest similarity
        assert len(results) > 0
        assert results[0][0] == "text_format"
        assert results[0][3]["domain"] == "text_processing"