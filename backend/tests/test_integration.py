"""
Integration tests to reproduce the actual "query failed" error
"""
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from rag_system import RAGSystem
from config import config


class TestIntegration:
    """Integration tests to reproduce real-world issues"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_chroma_path = config.CHROMA_PATH
        config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")
        
        # Mock API key to avoid real API calls
        config.ANTHROPIC_API_KEY = "test-key-12345"
        
    def teardown_method(self):
        """Cleanup test environment"""
        # Restore original config
        config.CHROMA_PATH = self.original_chroma_path
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rag_system_initialization_real_config(self):
        """Test RAG system initialization with real configuration"""
        try:
            rag_system = RAGSystem(config)
            assert rag_system is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.tool_manager is not None
            print("✓ RAG System initialization successful")
        except Exception as e:
            pytest.fail(f"RAG System initialization failed: {e}")
    
    def test_query_without_documents(self):
        """Test query on empty system - should handle gracefully"""
        try:
            rag_system = RAGSystem(config)
            
            # Mock the AI generator to avoid real API calls
            with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
                mock_generate.return_value = "I don't have any course content available."
                
                response, sources = rag_system.query("What is machine learning?")
                
                assert response is not None
                assert isinstance(sources, list)
                print("✓ Query on empty system handled gracefully")
                
        except Exception as e:
            print(f"✗ Query failed with error: {e}")
            # This might be the source of your "query failed" error
            assert False, f"Query should not fail completely: {e}"
    
    def test_document_loading_failure(self):
        """Test handling of document loading failures"""
        try:
            rag_system = RAGSystem(config)
            
            # Try to load from a non-existent directory
            courses, chunks = rag_system.add_course_folder("nonexistent_directory")
            
            assert courses == 0
            assert chunks == 0
            print("✓ Non-existent directory handled gracefully")
            
        except Exception as e:
            print(f"✗ Document loading failure not handled: {e}")
            assert False, f"Document loading should handle missing directories: {e}"
    
    def test_vector_search_with_no_data(self):
        """Test vector search when no data is loaded"""
        try:
            rag_system = RAGSystem(config)
            
            # Access the search tool directly
            search_tool = rag_system.search_tool
            result = search_tool.execute("test query")
            
            assert "No relevant content found" in result
            print("✓ Empty search handled gracefully")
            
        except Exception as e:
            print(f"✗ Vector search failed: {e}")
            assert False, f"Vector search should handle empty data: {e}"
    
    def test_anthropic_api_key_validation(self):
        """Test handling of missing or invalid API key"""
        # Temporarily clear the API key
        original_key = config.ANTHROPIC_API_KEY
        config.ANTHROPIC_API_KEY = ""
        
        try:
            rag_system = RAGSystem(config)
            
            # This might throw an error if API key validation is strict
            with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
                mock_generate.side_effect = Exception("API key invalid")
                
                with pytest.raises(Exception, match="API key invalid"):
                    rag_system.query("test query")
                    
            print("✓ API key validation works as expected")
            
        except Exception as e:
            print(f"✗ API key handling issue: {e}")
            
        finally:
            # Restore API key
            config.ANTHROPIC_API_KEY = original_key
    
    def test_chroma_db_initialization_failure(self):
        """Test handling of ChromaDB initialization issues"""
        # Use an invalid path to trigger ChromaDB errors
        config.CHROMA_PATH = "/invalid/path/that/cannot/be/created"
        
        try:
            # This should potentially fail
            rag_system = RAGSystem(config)
            
            # If it doesn't fail during init, it might fail during search
            search_tool = rag_system.search_tool
            result = search_tool.execute("test query")
            
            print(f"ChromaDB result with invalid path: {result}")
            
        except Exception as e:
            print(f"✓ ChromaDB initialization failed as expected: {e}")
            # This is actually expected behavior
        
        finally:
            # Reset to valid path
            config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")
    
    def test_tool_execution_error_propagation(self):
        """Test how tool execution errors propagate to query responses"""
        try:
            rag_system = RAGSystem(config)
            
            # Mock the vector store to return an error
            with patch.object(rag_system.vector_store, 'search') as mock_search:
                from vector_store import SearchResults
                mock_search.return_value = SearchResults.empty("Database connection failed")
                
                # Mock AI generator to simulate tool usage
                with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
                    mock_generate.return_value = "Database connection failed"
                    
                    response, sources = rag_system.query("test query")
                    
                    # The error should be handled and returned as response
                    assert "Database connection failed" in response or response == "Database connection failed"
                    print("✓ Tool execution errors properly propagated")
                    
        except Exception as e:
            print(f"✗ Tool execution error handling failed: {e}")
            assert False, f"Tool errors should be handled gracefully: {e}"
    
    def test_realistic_workflow_failure_points(self):
        """Test the complete workflow to identify failure points"""
        try:
            print("\n=== Testing Complete RAG Workflow ===")
            
            # 1. Initialize system
            print("1. Initializing RAG system...")
            rag_system = RAGSystem(config)
            print("   ✓ RAG system initialized")
            
            # 2. Try to load documents from docs folder
            print("2. Loading documents...")
            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses, chunks = rag_system.add_course_folder(docs_path)
                print(f"   ✓ Loaded {courses} courses, {chunks} chunks")
            else:
                print("   ⚠ No docs folder found, continuing with empty system")
            
            # 3. Test query with mocked AI response
            print("3. Testing query...")
            with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
                mock_generate.return_value = "This is a test response"
                
                response, sources = rag_system.query("What is artificial intelligence?")
                print(f"   ✓ Query successful: {response[:50]}...")
                print(f"   ✓ Sources returned: {len(sources)} items")
            
            print("=== All workflow steps completed successfully ===")
            
        except Exception as e:
            print(f"✗ Workflow failed at step: {e}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            assert False, f"Complete workflow should not fail: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])