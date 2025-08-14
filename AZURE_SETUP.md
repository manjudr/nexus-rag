# Azure OpenAI Setup Guide (Simplified!)

## What You Need
1. **Azure OpenAI Endpoint** - Example: `https://your-resource.openai.azure.com/`
2. **Azure OpenAI API Key** - 32-character key from Azure
3. **Deployment Names** - Names of your deployed models (e.g., `gpt-35-turbo`, `text-embedding-3-small`)

## Super Simple Setup (Uses OpenAI-Compatible Format)

### 1. Set Environment Variables
```bash
# Your Azure API key
export OPENAI_API_KEY="your-azure-api-key-here"

# For embedding model (replace 'text-embedding-3-small' with your deployment name)
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/text-embedding-3-small/"

# Alternative: You can also set different base URLs for different models if needed
```

### 2. Update Configuration
In `main_config.py`, change:
```python
MODEL_PROVIDER = "azure"  # Change from "local" to "azure"
```

Update the model names to match your Azure deployment names:
```python
"azure": {
    "embedding": "text-embedding-3-small",  # Your embedding deployment name
    "llm": "gpt-35-turbo",  # Your chat deployment name
}
```

### 3. Test the Setup
```bash
# Test status
python main_app.py --status

# Test with a query
python main_app.py --query "test azure connection" --json
```

## Why This is Better
✅ **No separate Azure classes needed** - uses existing OpenAI code  
✅ **Same API format** - just different endpoint  
✅ **Simpler setup** - only 2 environment variables  
✅ **Standard OpenAI SDK** - no Azure-specific code  

## Example Setup
If your Azure resource is `my-openai-resource` and you deployed:
- GPT-3.5-turbo as `gpt-35-turbo`  
- Embeddings as `text-embedding-3-small`

```bash
export OPENAI_API_KEY="a1b2c3d4e5f67890..."
export OPENAI_BASE_URL="https://my-openai-resource.openai.azure.com/openai/deployments/text-embedding-3-small/"
```
