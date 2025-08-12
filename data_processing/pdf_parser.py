from pypdf import PdfReader

class PDFParser:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str) -> tuple[list[str], list[dict]]:
        """Parse PDF and return list of text chunks with metadata."""
        import os
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        # Create metadata for each chunk
        filename = os.path.basename(pdf_path)
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "filename": filename,
                "page": i + 1,  # Approximate page number
                "source_path": pdf_path
            })
        
        return chunks, metadata

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
        return full_text

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            start += self.chunk_size - self.chunk_overlap
            
        return chunks

# Keep the original functions for backward compatibility
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n\n"
    return full_text

def chunk_text(text, max_length):
    paras = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paras:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
