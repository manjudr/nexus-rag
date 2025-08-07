from .base import BaseAgent
from tools.image_analyzer import ImageAnalyzerTool

class ImageAgent(BaseAgent):
    def __init__(self, tool: ImageAnalyzerTool):
        self.tool = tool

    def run(self, query: str):
        """
        Runs the agent. For this simple agent, the query is the path to the image.
        """
        print("Agent: Received image path. Using tool to analyze.")
        analysis_result = self.tool.run(query)
        return analysis_result
