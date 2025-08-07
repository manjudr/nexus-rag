from .base import BaseAgent
from models.llm import GenerativeModel
from tools.base import BaseTool

class GenerativeAgent(BaseAgent):
    def __init__(self, llm: GenerativeModel, tool: BaseTool):
        self.llm = llm
        self.tool = tool

    def _answer_question(self, question, contexts):
        context_text = "\n\n".join(contexts)
        input_text = f"question: {question} context: {context_text}"
        result = self.llm.generate(input_text)
        return result

    def run(self, question: str):
        """Runs the agent to answer a question."""
        print("Agent: Retrieving context with tool...")
        retrieved_chunks = self.tool.run(question)
        
        print("Agent: Generating final answer...")
        answer = self._answer_question(question, retrieved_chunks)
        return answer
