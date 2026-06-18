"""
Document RAG Ingestor for Long Term Memory
===========================================

Reads PDF, DOCX, and TXT files, chunks them, computes their semantic
embeddings, and directly stores them in LongTermMemory.

Prerequisites:
    pip install PyMuPDF python-docx sentence-transformers
"""

import os
from typing import List
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF (fitz) not installed. PDF ingestion disabled.")

try:
    import docx
except ImportError:
    docx = None
    print("Warning: python-docx not installed. DOCX ingestion disabled.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("Warning: sentence-transformers not installed. Embedding disabled.")

from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.knowledge.extractor import chunk_text
from eris.computation.activations import BVec

class DocumentIngestor:
    def __init__(self, memory_system: MemorySystem, embedding_model: str = "all-MiniLM-L6-v2"):
        self.memory = memory_system
        self.encoder = None
        if SentenceTransformer:
            print(f"Loading embedding model: {embedding_model}...")
            self.encoder = SentenceTransformer(embedding_model)

    def ingest_directory(self, directory_path: str):
        """Ingest all supported documents in a directory."""
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} not found.")
            return
        
        count = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                filepath = os.path.join(root, file)
                if file.lower().endswith('.pdf'):
                    self.ingest_pdf(filepath)
                    count += 1
                elif file.lower().endswith('.docx'):
                    self.ingest_docx(filepath)
                    count += 1
                elif file.lower().endswith('.txt'):
                    self.ingest_txt(filepath)
                    count += 1
        print(f"Ingested {count} documents into Long Term Memory.")

    def ingest_pdf(self, filepath: str):
        if not fitz:
            print("PyMuPDF required for PDF ingestion.")
            return
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        self._process_and_store(text, os.path.basename(filepath))

    def ingest_docx(self, filepath: str):
        if not docx:
            print("python-docx required for DOCX ingestion.")
            return
        doc = docx.Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])
        self._process_and_store(text, os.path.basename(filepath))

    def ingest_txt(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        self._process_and_store(text, os.path.basename(filepath))

    def _process_and_store(self, text: str, title: str):
        chunks = chunk_text(text, max_chars=1500)
        print(f"Ingesting '{title}' ({len(chunks)} chunks)...")
        
        for i, chunk in enumerate(chunks):
            chunk_title = f"[{title} Chunk {i+1}/{len(chunks)}]"
            content = f"{chunk_title}\n{chunk}"
            
            # Create empty BVec or compute if field is available
            # For simple ingestion, we use a neutral BVec. The field physics
            # will align it when recalled into the active FRACTAL workspace.
            bvec = BVec(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            
            embedding = None
            if self.encoder:
                embedding = self.encoder.encode(content)
            
            record = MemoryRecord(
                text=content,
                bvec=bvec,
                embedding=embedding,
                source="document_rag",
                metadata={"filename": title, "chunk": i}
            )
            # Store directly in Long Term Memory
            self.memory.ltm.store(record)
