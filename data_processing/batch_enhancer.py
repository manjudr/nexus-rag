"""
Batch Educational Metadata Enhancement
Pre-computes educational metadata and stores in database
"""
import json
import time
from typing import List, Dict
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.hybrid_enhancer import HybridEducationalEnhancer
from vector_db.milvus_db import MilvusDB

class BatchEducationalEnhancer:
    """Pre-computes educational metadata for all content"""
    
    def __init__(self, use_langextract: bool = True, api_key: str = None):
        self.enhancer = HybridEducationalEnhancer(use_langextract, api_key)
        self.db = MilvusDB()
        
    def enhance_all_content(self, collection_name: str = "educational_content"):
        """Pre-enhance all content in the database"""
        print("🔄 Starting batch educational enhancement...")
        
        # Get all chunks without educational metadata
        results = self.db.search_collection(
            collection_name=collection_name,
            query_vector=[0.0] * 1536,  # Dummy vector to get all results
            top_k=10000,  # Get all chunks
            filter_expr="educational_metadata == ''"  # Only unenhanced chunks
        )
        
        enhanced_count = 0
        start_time = time.time()
        
        for result in results:
            chunk_id = result.id
            content = result.entity.get('content', '')
            source_file = result.entity.get('source_file', '')
            
            if content and not result.entity.get('educational_metadata'):
                print(f"📚 Enhancing chunk {chunk_id} from {source_file}...")
                
                # Enhance with educational metadata
                enhanced = self.enhancer.enhance_educational_content(content, source_file)
                
                # Update database with enhanced metadata
                self.db.update_educational_metadata(
                    chunk_id=chunk_id,
                    educational_metadata=enhanced.to_dict()
                )
                
                enhanced_count += 1
                
                # Progress update every 10 chunks
                if enhanced_count % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / enhanced_count
                    print(f"✅ Enhanced {enhanced_count} chunks (avg: {avg_time:.2f}s per chunk)")
        
        total_time = time.time() - start_time
        print(f"🎉 Batch enhancement complete!")
        print(f"📊 Enhanced {enhanced_count} chunks in {total_time:.1f} seconds")
        print(f"⚡ Average: {total_time/enhanced_count:.2f}s per chunk")
        
    def enhance_single_file(self, file_path: str, collection_name: str = "educational_content"):
        """Enhance all chunks from a specific file"""
        print(f"🔄 Enhancing chunks from {file_path}...")
        
        # Get chunks from specific file
        filter_expr = f'source_file == "{Path(file_path).name}"'
        results = self.db.search_collection(
            collection_name=collection_name,
            query_vector=[0.0] * 1536,
            top_k=1000,
            filter_expr=filter_expr
        )
        
        enhanced_count = 0
        for result in results:
            content = result.entity.get('content', '')
            if content:
                enhanced = self.enhancer.enhance_educational_content(content, file_path)
                self.db.update_educational_metadata(
                    chunk_id=result.id,
                    educational_metadata=enhanced.to_dict()
                )
                enhanced_count += 1
                
        print(f"✅ Enhanced {enhanced_count} chunks from {file_path}")
