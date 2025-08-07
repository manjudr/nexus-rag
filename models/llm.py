from abc import ABC, abstractmethod

class GenerativeModel(ABC):
    @abstractmethod
    def generate(self, prompt: str):
        pass
