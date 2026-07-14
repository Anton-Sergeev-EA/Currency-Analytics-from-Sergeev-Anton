import importlib


class RAGService:
    def __init__(self):
        module = importlib.import_module('src.infrastructure.rag.generation.generator')
        self.generator = module.RAGGenerator()

    def ask(self, question: str) -> dict:
        return self.generator.generate_response(question)
    