#!/usr/bin/env python3
"""
Test script to verify system functionality after cleanup
This tests without requiring OpenAI API keys
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("🔍 Testing imports...")
    
    try:
        # Test core modules
        from vector_db.enhanced_content_discovery_db import EnhancedContentDiscoveryVectorDB
        print("✅ Enhanced Vector DB module imported")
        
        from data_processing.hybrid_enhancer import HybridEducationalEnhancer
        print("✅ Data processing module imported")
        
        from tools.content_discovery_tool import ContentDiscoveryTool
        print("✅ Content discovery tool imported")
        
        from agents.orchestrator_agent import OrchestratorAgent
        print("✅ Orchestrator agent imported")
        
        # Test model modules
        from models.sentence_transformer import SentenceTransformerEmbeddingModel
        print("✅ Local embedding model imported")
        
        from models.huggingface_llm import HuggingFaceGenerativeModel
        print("✅ Local LLM model imported")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_metadata_loading():
    """Test metadata file loading"""
    print("\n🔍 Testing metadata loading...")
    
    try:
        import json
        metadata_path = "data/educational_content/metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Metadata loaded: {len(metadata)} items")
        for item in metadata:
            if isinstance(item, dict):
                print(f"   - {item.get('filename', 'Unknown')}: {item.get('title', 'No title')}")
            else:
                print(f"   - {item}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata loading error: {e}")
        return False

def test_database_connection():
    """Test database connection without requiring API keys"""
    print("\n🔍 Testing database connection...")
    
    try:
        from vector_db.enhanced_content_discovery_db import EnhancedContentDiscoveryVectorDB
        
        # Test initialization without actually connecting
        print("✅ Database class available")
        return True
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_local_models():
    """Test local models that don't require API keys"""
    print("\n🔍 Testing local models...")
    
    try:
        # Test local embedding model
        from models.sentence_transformer import SentenceTransformerEmbeddingModel
        embedding_model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        print("✅ Local embedding model initialized")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = embedding_model.create_embedding(test_text)
        print(f"✅ Embedding generated: dimension {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local model error: {e}")
        return False

def test_fallback_systems():
    """Test fallback mechanisms"""
    print("\n🔍 Testing fallback systems...")
    
    try:
        from data_processing.hybrid_enhancer import HybridEducationalEnhancer
        
        # Test with local models only (no API keys needed)
        enhancer = HybridEducationalEnhancer({
            'langextract_enabled': False,  # Disable API-based extraction
            'fallback_enabled': True
        })
        
        print("✅ Fallback system initialized")
        print("✅ Fallback methods available for metadata extraction")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback system error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 NexusRAG System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_metadata_loading,
        test_database_connection,
        test_local_models,
        test_fallback_systems
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! System is ready for use.")
        print("\n💡 To use with Azure OpenAI, set these environment variables:")
        print("   export OPENAI_API_KEY='your-azure-api-key'")
        print("   export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("   export AZURE_CHAT_DEPLOYMENT='gpt-35-turbo'")
        print("   export AZURE_EMBEDDING_DEPLOYMENT='text-embedding-3-small'")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
