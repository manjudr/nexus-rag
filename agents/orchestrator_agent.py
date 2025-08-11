from agents.base import BaseAgent
from models.llm import GenerativeModel
import re

class OrchestratorAgent(BaseAgent):
    def __init__(self, llm: GenerativeModel, tools: list):
        self.llm = llm
        self.tools = tools

    def _create_prompt(self, query: str):
        """Create a prompt for the LLM to choose the best tool"""
        tools_description = ""
        for tool in self.tools:
            tools_description += f"- {tool.name}: {tool.description}\n"
        
        prompt = f"""You are an intelligent router for a RAG system. Choose the most appropriate data source.

Available data sources:
{tools_description}

User Query: "{query}"

ROUTING RULES:
- If the query mentions video, transcript, interview, or spoken content, choose "Video Transcripts"
- For all other queries (learning, studying, documents, PDFs, educational content), choose "Content Discovery"

Respond with ONLY the exact tool name.

Tool name:"""
        return prompt

    def _simple_router(self, query: str):
        """Simple keyword-based routing as fallback"""
        query_lower = query.lower()
        
        # Video-related keywords
        video_keywords = ['video', 'transcript', 'interview', 'speech', 'conversation', 'spoken']
        if any(keyword in query_lower for keyword in video_keywords):
            return next((tool for tool in self.tools if "video" in tool.name.lower()), None)
        
        # Default to Content Discovery for all other queries (learning, educational content, etc.)
        return next((tool for tool in self.tools if "content discovery" in tool.name.lower()), None)

    def run(self, query: str):
        """Route query to the most appropriate tool"""
        print("🎯 Orchestrator: Analyzing query and choosing best data source...")
        
        # Debug: Show available tools
        print(f"🔍 Available tools: {[tool.name for tool in self.tools]}")
        
        # For now, skip LLM routing and use keyword-based routing directly
        # This is more reliable with local models
        print("🎯 Using keyword-based routing for reliability")
        chosen_tool = self._simple_router(query)
        
        print(f"✅ Orchestrator: Selected '{chosen_tool.name}' for this query")
        
        # Run the selected tool
        return chosen_tool.run(query)
