import os
import sys
import time
import pathlib
import requests
from typing import List

# Configure library path
DOC_DIR = pathlib.Path.home() / "Documents" / "Eris_Library"
API_URL = "http://localhost:8001/ingest"
DELAY_SEC = 0.1

try:
    import docx
    import PyPDF2
except ModuleNotFoundError:
    print("Dependencies missing. Run: pip install python-docx PyPDF2 requests")
    sys.exit(1)

def extract_chunks(file_path: pathlib.Path) -> List[str]:
    """Extract text chunks from DOCX or PDF files."""
    chunks = []
    if file_path.suffix.lower() == ".docx":
        try:
            doc = docx.Document(str(file_path))
            for para in doc.paragraphs:
                text = para.text.strip()
                if len(text) > 20:
                    chunks.append(text)
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            
    elif file_path.suffix.lower() == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # Split by double newlines for rough paragraphing
                        paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
                        chunks.extend(paras)
        except Exception as e:
            print(f"Error reading PDF {file_path.name}: {e}")
            
    return chunks

def main():
    print("===================================================")
    print("Eris v4 - FRACTAL PDE Library Ingestor")
    print("===================================================")
    
    if not DOC_DIR.exists():
        print(f"[!] Directory not found: {DOC_DIR}")
        print("Creating directory. Please drop your files there and run again.")
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        return

    files = list(DOC_DIR.glob("*.docx")) + list(DOC_DIR.glob("*.pdf"))
    if not files:
        print(f"No .docx or .pdf files found in {DOC_DIR}")
        return

    print(f"Found {len(files)} files to ingest. Routing to {API_URL}\n")

    total_chunks = 0
    for path in files:
        print(f"Processing: {path.name}...")
        chunks = extract_chunks(path)
        
        for chunk in chunks:
            try:
                res = requests.post(API_URL, json={"text": chunk, "title": path.name})
                if res.status_code == 200:
                    total_chunks += 1
                time.sleep(DELAY_SEC)
            except requests.exceptions.ConnectionError:
                print("[!] Failed to connect to Eris server. Is FastAPI running on port 8000?")
                sys.exit(1)
                
        print(f"  → Added {len(chunks)} conceptual seeds to the PDE.\n")

    print(f"Finished. {total_chunks} total knowledge seeds ingested.")

if __name__ == "__main__":
    main()
