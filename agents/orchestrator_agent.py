from .base import BaseAgent
import re

class OrchestratorAgent(BaseAgent):
    def __init__(self, pdf_agent: BaseAgent, image_agent: BaseAgent):
        self.pdf_agent = pdf_agent
        self.image_agent = image_agent

    def run(self, query: str):
        """
        Analyzes the user's query and routes it to the appropriate agent(s).
        This is a simple rule-based router. A more advanced version could use an LLM to decide.
        """
        print("Orchestrator: Analyzing query...")

        # Check if the query contains an image path and a text command
        image_path_match = re.search(r'[\w/\\-]+\.(?:png|jpg|jpeg)', query, re.I)
        
        if image_path_match:
            image_path = image_path_match.group(0)
            # Isolate the text command from the image path
            command = query.replace(image_path, "").strip()

            print(f"Orchestrator: Detected image task. Path: '{image_path}', Command: '{command}'")
            
            # --- Agent Collaboration ---
            # 1. Get information from the image using the ImageAgent
            image_content = self.image_agent.run(image_path)
            
            # 2. Pass the output to the PDF/Text agent for further processing
            print("Orchestrator: Passing image content to PDF agent for processing...")
            # We combine the original command with the new context from the image
            new_query = f"{command}: '{image_content}'"
            final_answer = self.pdf_agent.run(new_query)
            return final_answer
        else:
            # If it's not an image task, assume it's a PDF/text task
            print("Orchestrator: Detected PDF/Text task. Routing to PDF agent.")
            return self.pdf_agent.run(query)
