from fastmcp import FastMCP
import json
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(name="MCP Server")

def _query_content_internal(query: str) -> str:
    """Internal function to query the content database via API"""
    try:
        # Make HTTP request to local API endpoint
        url = "http://localhost:8000/query"
        params = {"q": query}
        
        logger.info(f"Making request to {url} with query: {query}")
        
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Return the response content
        return response.text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making HTTP request: {e}")
        return json.dumps({
            "error": f"Failed to connect to API endpoint: {str(e)}",
            "status": "error"
        })
    except Exception as e:
        logger.error(f"Error in query_content: {e}")
        return json.dumps({
            "error": f"Query processing failed: {str(e)}",
            "status": "error"
        })

@mcp.tool
def query_content(query: str) -> str:
    """Query the content database via API endpoint"""
    return _query_content_internal(query)

if __name__ == "__main__":
    mcp.run()