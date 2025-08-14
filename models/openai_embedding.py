import os
from openai import OpenAI
from .embedding import EmbeddingModel

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        # Check if we're using Azure
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")
        
        if azure_endpoint and azure_deployment:
            # Azure OpenAI with specific embedding deployment and API version
            base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{azure_deployment}/"
            self.client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                default_query={"api-version": "2024-02-15-preview"}
            )
            # For Azure, use the deployment name as the model name
            self.model_name = azure_deployment
            print(f"🔗 Using Azure OpenAI embedding: {base_url} (model: {self.model_name})")
        else:
            # Regular OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            print("🔗 Using regular OpenAI embedding")
        self._dimension = None

    def create_embedding(self, text):
        import time
        
        # Handle both single text and list of texts
        if isinstance(text, list):
            # Process list of texts in batches to respect rate limits
            batch_size = 50  # Process 50 texts at a time
            all_embeddings = []
            
            print(f"🔄 Processing {len(text)} texts in batches of {batch_size}...")
            
            for i in range(0, len(text), batch_size):
                batch = text[i:i + batch_size]
                processed_batch = [t.replace("\n", " ") for t in batch]
                
                try:
                    response = self.client.embeddings.create(input=processed_batch, model=self.model_name)
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    if self._dimension is None and batch_embeddings:
                        self._dimension = len(batch_embeddings[0])
                    
                    print(f"   ✅ Processed batch {i//batch_size + 1}/{(len(text) + batch_size - 1)//batch_size}")
                    
                    # Add small delay between batches to respect rate limits
                    if i + batch_size < len(text):  # Don't delay after last batch
                        time.sleep(1)  # 1 second delay between batches
                        
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        print(f"   ⏳ Rate limit hit, waiting 10 seconds...")
                        time.sleep(10)
                        # Retry the same batch
                        response = self.client.embeddings.create(input=processed_batch, model=self.model_name)
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        print(f"   ✅ Retry successful for batch {i//batch_size + 1}")
                    else:
                        raise e
            
            return all_embeddings
        else:
            # Process single text
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(input=[text], model=self.model_name)
            embedding = response.data[0].embedding
            if self._dimension is None:
                self._dimension = len(embedding)
            return embedding

    def get_embedding_dimension(self):
        if self._dimension is None:
            # Create a dummy embedding to determine the dimension
            self.create_embedding("test")
        return self._dimension
