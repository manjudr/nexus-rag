class TextParser:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_txt(self, file_path: str) -> list[str]:
        """Parse text file into chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple chunking strategy
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
