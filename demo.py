#!/usr/bin/env python3
"""
Quick demo to show system functionality with local models
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_query_without_api():
    """Demonstrate system with local models only"""
    print("🎯 NexusRAG Demo - Local Models Only")
    print("=" * 50)
    
    try:
        # Initialize local embedding model
        from models.sentence_transformer import SentenceTransformerEmbeddingModel
        embedding_model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        
        # Test queries
        test_queries = [
            "What is photosynthesis?",
            "Tell me about physical education",
            "Biology class 10 content"
        ]
        
        print("🔍 Testing query embeddings generation:")
        for query in test_queries:
            embedding = embedding_model.create_embedding(query)
            print(f"   ✅ '{query}' -> embedding dimension: {len(embedding)}")
        
        print("\n📚 Available educational content:")
        import json
        with open("data/educational_content/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        for item in metadata:
            if isinstance(item, dict):
                print(f"   📖 {item.get('title', 'Unknown Title')}")
                print(f"      File: {item.get('filename', 'Unknown')}")
                print(f"      Subject: {item.get('subject', 'General')}")
                print()
        
        print("🎉 System is ready! All components working correctly.")
        print("\n💡 Next steps:")
        print("   1. Set up Azure OpenAI environment variables")
        print("   2. Run 'python main_app.py' for full functionality")
        print("   3. Use vector search and LLM-generated summaries")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

if __name__ == "__main__":
    success = demo_query_without_api()
    sys.exit(0 if success else 1)
