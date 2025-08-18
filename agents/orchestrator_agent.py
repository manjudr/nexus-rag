from agents.base import BaseAgent
from models.llm import GenerativeModel
import re

class OrchestratorAgent(BaseAgent):
    def __init__(self, llm: GenerativeModel, agents: list):
        self.llm = llm
        self.agents = agents  # Now routes to agents, not tools

    def _create_prompt(self, query: str):
        """Create a prompt for the LLM to choose the best agent"""
        agents_description = ""
        for agent in self.agents:
            # Get agent name from class name if no name attribute
            agent_name = getattr(agent, 'name', agent.__class__.__name__)
            agent_desc = getattr(agent, 'description', f"Handles queries for {agent_name}")
            agents_description += f"- {agent_name}: {agent_desc}\n"
        
        prompt = f"""You are an intelligent router for a RAG system. Choose the most appropriate specialized agent.

Available agents:
{agents_description}

User Query: "{query}"

ROUTING RULES:
- If the query mentions video, transcript, interview, or spoken content, choose "Video Transcript Agent"
- For educational content, learning, studying, documents, PDFs, choose "PDF Content Agent"  
- For general questions not fitting other categories, choose "General Query Agent"

Respond with ONLY the exact agent name.

Agent name:"""
        return prompt

    def _simple_router(self, query: str):
        """Simple keyword-based routing as fallback"""
        query_lower = query.lower()
        
        # Video-related keywords
        video_keywords = ['video', 'transcript', 'interview', 'speech', 'conversation', 'spoken']
        if any(keyword in query_lower for keyword in video_keywords):
            # Find video transcript agent
            for agent in self.agents:
                if "video" in agent.__class__.__name__.lower() or "transcript" in agent.__class__.__name__.lower():
                    return agent
        
        # Educational/PDF content keywords (default for most learning queries)
        educational_keywords = ['learn', 'study', 'education', 'pdf', 'document', 'teaching', 'course']
        if any(keyword in query_lower for keyword in educational_keywords):
            # Find PDF content agent
            for agent in self.agents:
                if "pdf" in agent.__class__.__name__.lower() or "content" in agent.__class__.__name__.lower():
                    return agent
        
        # Default to PDF Content Agent for educational system
        for agent in self.agents:
            if "pdf" in agent.__class__.__name__.lower():
                return agent
                
        # Final fallback to general agent
        for agent in self.agents:
            if "general" in agent.__class__.__name__.lower():
                return agent
                
        # If no specific agent found, return first agent
        return self.agents[0] if self.agents else None

    def run(self, query: str):
        """Route query to the most appropriate specialized agent"""
        print("🎯 Orchestrator: Analyzing query and choosing best specialized agent...")
        
        # Use keyword-based routing for reliability
        chosen_agent = self._simple_router(query)
        
        if chosen_agent:
            agent_name = chosen_agent.__class__.__name__
            print(f"✅ Using {agent_name} for this query")
            
            # Run the selected agent
            result = chosen_agent.run(query)
            
            # Check if PDF agent returned low relevance error
            if (agent_name == "PDFContentAgent" and 
                isinstance(result, str) and 
                ("No relevant content found" in result or "low_relevance" in result)):
                
                print("🔄 PDF content not relevant, falling back to General Query Agent")
                
                # Find and use General Query Agent as fallback
                general_agent = None
                for agent in self.agents:
                    if "general" in agent.__class__.__name__.lower():
                        general_agent = agent
                        break
                
                if general_agent:
                    return general_agent.run(query)
                else:
                    return result
            
            return result
        else:
            agent_names = [agent.__class__.__name__ for agent in self.agents]
            return {
                "error": "No suitable agent found for this query",
                "available_agents": agent_names,
                "query": query
            }
