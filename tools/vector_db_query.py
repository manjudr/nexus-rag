from .base import BaseTool
from vector_db.base import BaseVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel
from rank_bm25 import BM25Okapi
from vector_db.milvus_db import MilvusVectorDB

class VectorDBQueryTool(BaseTool):
    def __init__(self, db: BaseVectorDB, embedding_model: EmbeddingModel, llm: GenerativeModel, name: str, description: str, top_k: int = 5):
        self.db = db
        self.embedding_model = embedding_model
        self.llm = llm
        self.name = name
        self.description = description
        self.top_k = top_k

    def _initialize_bm25(self):
        if isinstance(self.db, MilvusVectorDB):
            print(f"Tool ({self.name}): Initializing BM25...")
            documents_data = self.db.get_all_documents()
            if not documents_data:
                return None, None
            # Extract text content for BM25, handling both old and new formats
            documents = []
            for doc in documents_data:
                if isinstance(doc, tuple):
                    documents.append(doc[0])  # Text is first element
                else:
                    documents.append(doc)
            tokenized_corpus = [doc.split(" ") for doc in documents]
            return BM25Okapi(tokenized_corpus), documents_data
        return None, None

    def _create_response_prompt(self, query: str, context_chunks: list) -> str:
        """Create a prompt for the LLM to generate a response based on retrieved context"""
        # Limit context to avoid overwhelming the model
        context = "\n\n".join(context_chunks[:3])  # Use only top 3 chunks
        
        prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {query}

Answer the question using the information provided above. Be specific and detailed."""
        return prompt

    def run(self, query: str):
        """Runs the tool to query the vector DB and generate a response."""
        bm25, documents_data = self._initialize_bm25()

        print(f"Tool ({self.name}): Creating embedding for query: '{query}'")
        query_embedding = self.embedding_model.create_embedding(query)
        
        print(f"Tool ({self.name}): Querying Vector DB...")
        vector_results = self.db.search(query_embedding, top_k=self.top_k)

        # Extract text content from results (handle both old and new format)
        context_chunks = []
        for result in vector_results:
            if isinstance(result, tuple):
                context_chunks.append(result[0])  # Text is first element
            else:
                context_chunks.append(result)
        
        if bm25 and documents_data:
            print(f"Tool ({self.name}): Performing BM25 search...")
            tokenized_query = query.split(" ")
            # Get text documents for BM25
            text_documents = []
            for doc in documents_data:
                if isinstance(doc, tuple):
                    text_documents.append(doc[0])
                else:
                    text_documents.append(doc)
            
            bm25_results = bm25.get_top_n(tokenized_query, text_documents, n=self.top_k)
            
            # Combine and deduplicate results
            combined_results = list(set(context_chunks + bm25_results))
            context_chunks = combined_results

        if not context_chunks:
            return f"I couldn't find any relevant information in the {self.name.lower()} to answer your question."

        # Generate response using LLM
        print(f"Tool ({self.name}): Generating response...")
        prompt = self._create_response_prompt(query, context_chunks)
        response = self.llm.generate(prompt)
        
        return response
