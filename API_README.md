# NexusRAG API Documentation

## Overview

The NexusRAG API provides RESTful endpoints for intelligent educational content discovery and summarization. It accepts queries in natural language and returns structured JSON responses with relevant content recommendations.

## Features

- 🔍 **Natural Language Queries**: Ask questions in plain English
- 📚 **Educational Content Discovery**: Find relevant content from PDF documents
- 🤖 **AI-Powered Summarization**: Get intelligent summaries tailored to your query
- 🔤 **Typo Tolerance**: Handles spelling mistakes automatically
- 📊 **Structured Responses**: Returns JSON with metadata and relevance scores
- 🌐 **CORS Enabled**: Ready for frontend integration

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn
```

### 2. Start the API Server

```bash
# Simple API (recommended for testing)
python simple_api.py

# Full API (requires complete setup)
python api.py
```

The server will start on `http://localhost:8000`

### 3. Test the API

```bash
# Check status
curl http://localhost:8000/status

# Simple query
curl "http://localhost:8000/query?q=What%20is%20coronavirus?"

# Advanced query (POST)
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "photosynthesis process", "max_results": 3}'
```

## API Endpoints

### GET `/`
Returns basic API information.

**Response:**
```json
{
  "message": "NexusRAG Content Discovery API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### GET `/status`
Returns system status and health check.

**Response:**
```json
{
  "status": "ready",
  "version": "1.0.0",
  "timestamp": "2025-08-19T11:30:00"
}
```

### POST `/query`
Main endpoint for content discovery queries.

**Request Body:**
```json
{
  "query": "What is photosynthesis?",
  "max_results": 5
}
```

**Response:**
```json
{
  "query": "What is photosynthesis?",
  "status": "success",
  "total_results": 3,
  "processing_time_ms": 1250,
  "agent_used": "orchestrator",
  "recommendations": [
    {
      "filename": "plant_biology.pdf",
      "title": "Plant Photosynthesis Process",
      "course": "Biology Studies",
      "page_number": 12,
      "keywords": ["photosynthesis", "chlorophyll", "light", "carbon dioxide"],
      "summary": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
      "relevance_score": 0.95
    }
  ],
  "message": "Query processed successfully",
  "timestamp": "2025-08-19T11:30:15"
}
```

### GET `/query`
GET version of the query endpoint for simple requests.

**Parameters:**
- `q` (required): The search query
- `max_results` (optional): Maximum number of results (default: 5)

**Example:**
```
GET /query?q=coronavirus&max_results=3
```

## Frontend Integration

### JavaScript Example

```javascript
// Simple query
async function searchContent(query) {
    const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            max_results: 5
        })
    });
    
    const data = await response.json();
    return data;
}

// Usage
searchContent("What is coronavirus?")
    .then(data => {
        console.log(`Found ${data.total_results} results:`);
        data.recommendations.forEach(rec => {
            console.log(`- ${rec.title} (${rec.relevance_score})`);
        });
    });
```

### Frontend Demo

Open `frontend_demo.html` in your browser for a complete working example with a user interface.

## Response Format

### Success Response

```json
{
  "query": "user query",
  "status": "success",
  "total_results": 3,
  "processing_time_ms": 1200,
  "agent_used": "orchestrator",
  "recommendations": [
    {
      "filename": "document.pdf",
      "title": "Content Title",
      "course": "Course Name",
      "page_number": 5,
      "keywords": ["keyword1", "keyword2"],
      "summary": "Relevant content summary...",
      "relevance_score": 0.85
    }
  ],
  "message": "Query processed successfully",
  "timestamp": "2025-08-19T11:30:00"
}
```

### Error Response

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

## Testing

### Automated Testing

```bash
# Install requests library
pip install requests

# Run test suite
python test_api.py
```

### Manual Testing

1. **Basic functionality:**
   ```bash
   curl "http://localhost:8000/query?q=photosynthesis"
   ```

2. **Typo handling:**
   ```bash
   curl "http://localhost:8000/query?q=coronovirus"
   ```

3. **Complex queries:**
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "How do plants convert sunlight to energy?", "max_results": 3}'
   ```

## Error Handling

The API handles various error scenarios:

- **400 Bad Request**: Empty or invalid query
- **500 Internal Server Error**: System or processing errors
- **503 Service Unavailable**: System not initialized

## Performance

- **Response Time**: Typically 500-2000ms depending on query complexity
- **Concurrent Requests**: Supports multiple simultaneous queries
- **Caching**: Results may be cached for repeated queries

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for Azure OpenAI integration
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION`: API version (default: 2023-05-15)

### Model Configuration

Edit `main_config.py` to configure:
- Model providers (Azure OpenAI, local models)
- Database paths and collections
- Agent configurations
- Processing parameters

## Security

- **CORS**: Enabled for all origins (configure for production)
- **Rate Limiting**: Not implemented (add as needed)
- **Authentication**: Not implemented (add as needed)

## Deployment

### Local Development
```bash
python simple_api.py
```

### Production (Docker)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production (Direct)
```bash
uvicorn simple_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Troubleshooting

### Common Issues

1. **"Address already in use"**
   ```bash
   # Find and kill existing processes
   pkill -f "api.py"
   lsof -ti:8000 | xargs kill -9
   ```

2. **"Module not found"**
   ```bash
   # Ensure you're in the correct directory
   cd /path/to/NexusRAG
   # Use virtual environment
   source .venv/bin/activate
   ```

3. **"OpenAI API key not set"**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # Or edit main_config.py
   ```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## Support

For issues and questions:
1. Check the logs in the terminal where the API is running
2. Test with the provided `test_api.py` script
3. Verify your configuration in `main_config.py`
