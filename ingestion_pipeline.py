import os
import re
import uuid
import io
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

# Disable ChromaDB telemetry to suppress warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Disable HuggingFace tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress macOS malloc stack logging warnings
os.environ["MallocStackLogging"] = "0"

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer # No longer directly used
# from chromadb import PersistentClient # No longer directly used
import PyPDF2
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma # Use LangChain's Chroma

# ─────────────────────────────────────────────────────────────────────────────
# 0. Config & helpers
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

SHEET_URL = os.getenv("MIT_RISKS_SHEET_URL")
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", "./papers"))
# CHROMA_PERSIST_DIR is used by LangChain's Chroma wrapper directly
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vector_store"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en")

# Ensure the persist directory exists
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

if not SHEET_URL:
    raise ValueError("MIT_RISKS_SHEET_URL is not set – please add it to your .env file.")

# Deterministic UUID namespaces (repeatable ingestion)
NAMESPACE_MIT = uuid.uuid5(uuid.NAMESPACE_URL, "mit-ai-risks")
NAMESPACE_PDF = uuid.uuid5(uuid.NAMESPACE_URL, "responsible-ai-papers")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu") # Replaced by LangChain's Embeddings
EMBED = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})

# Chroma client is now handled by the Chroma.from_documents method
# client = PersistentClient(path=str(CHROMA_PERSIST_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Google‑Sheet loader (gid‑aware)
# ─────────────────────────────────────────────────────────────────────────────


def _to_csv_export(url: str) -> str:
    """Convert a regular Google‑Sheets link into a CSV export link, preserving gid (tab)."""
    if "export?format=csv" in url:
        return url
    m = re.search(r"docs.google.com/spreadsheets/d/([\w-]+)", url)
    if not m:
        raise ValueError("Not a Google‑Sheets URL: " + url)
    sheet_id = m.group(1)
    gid_m = re.search(r"[?&]gid=(\d+)", url)
    gid_part = f"&gid={gid_m.group(1)}" if gid_m else ""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid_part}"


def load_mit_sheet(url: str) -> pd.DataFrame:
    csv_url = _to_csv_export(url)
    resp = requests.get(csv_url)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def row_to_docs(row: pd.Series) -> List[Document]:
    base_meta = {
        "title": row.get("risk"),
        "source_type": "mit_ai_risk_sheet",
        "url": row.get("source"),
        "license": "CC-BY-4.0",
    }
    text = "\n".join(
        f"{k.replace('_', ' ').capitalize()}: {v}"
        for k, v in row.items() if k not in ("id",)
    )
    docs = []
    for i, chunk in enumerate(text_splitter.split_text(text)):
        cleaned_metadata = _clean_meta({**base_meta, "row_index": int(row.name), "chunk_index": i})
        docs.append(Document(
            page_content=chunk,
            metadata=cleaned_metadata,
        ))
    return docs

# ─────────────────────────────────────────────────────────────────────────────
# 2. PDF loader
# ─────────────────────────────────────────────────────────────────────────────


def pdf_to_text(path: Path) -> str:
    out = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            out.append(page.extract_text() or "")
    return "\n".join(out)


def detect_license(first_page: str) -> str:
    cc = re.search(r"Creative\s+Commons[^\n]+", first_page, re.I)
    if cc:
        return cc.group(0).strip()
    if "arxiv" in first_page.lower():
        return "arXiv (see paper)"
    return "unknown"


def pdf_to_docs(pdf: Path) -> List[Document]:
    raw = pdf_to_text(pdf)
    if not raw.strip():
        return []
    header = raw[:800]
    title = (re.search(r"^(.*)\n", header) or [None, pdf.stem])[1].strip()
    authors_m = re.search(r"\n([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)+)\n", header)
    authors = authors_m.group(1).strip() if authors_m else "unknown"
    licence = detect_license(header)
    docs = []
    for i, chunk in enumerate(text_splitter.split_text(raw)):
        cleaned_metadata = _clean_meta({
            "title": title,
            "authors": authors,
            "source_type": "responsible_ai_paper",
            "file_name": pdf.name,
            "license": licence,
            "url": f"file://{pdf.resolve()}",
            "chunk_index": i,
        })
        docs.append(Document(
            page_content=chunk,
            metadata=cleaned_metadata
        ))
    return docs

# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding + sanitised upsert
# ─────────────────────────────────────────────────────────────────────────────

# def embed(texts: List[str]) -> List[List[float]]: # No longer needed if Chroma handles it
#     return EMBED.embed_documents(texts)


def _clean_meta(meta: Dict) -> Dict:
    return {k: v for k, v in meta.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}


def upsert(collection_name: str, docs: List[Document]):
    if not docs:
        return
    # The previous edit already changed this to Chroma.from_documents,
    # but we need to ensure it's called correctly in main() for persistence.
    # For now, let's assume main will handle the single call to from_documents
    # or use an add_documents approach.
    # This function might need to be removed or significantly changed
    # depending on the main() function's logic.

    # Placeholder for now, the actual Chroma creation/update will be in main.
    print(f"Prepared {len(docs)} documents for collection {collection_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main script
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("MIT AI Risks →", SHEET_URL)
    mit_docs = []
    for _, row in load_mit_sheet(SHEET_URL).iterrows():
        mit_docs.extend(row_to_docs(row))
    
    if mit_docs:
        print(f"Creating/updating MIT AI Risks collection with {len(mit_docs)} documents...")
        Chroma.from_documents(
            documents=mit_docs,
            embedding=EMBED,
            collection_name="mit_ai_risks",
            persist_directory=str(CHROMA_PERSIST_DIR)
        )
        print(f"↑ {len(mit_docs)} docs → mit_ai_risks")
    else:
        print("No documents found for MIT AI Risks.")

    print("\nPapers →", PAPERS_DIR)
    if not PAPERS_DIR.exists() or not PAPERS_DIR.is_dir():
        print(f"SKIPPING: Papers directory not found: {PAPERS_DIR}")
    else:
        pdf_docs = []
        for pdf_file in PAPERS_DIR.glob("*.pdf"):
            print(" ", pdf_file.name)
            pdf_docs.extend(pdf_to_docs(pdf_file))
        
        if pdf_docs:
            print(f"Creating/updating Responsible AI Papers collection with {len(pdf_docs)} documents...")
            Chroma.from_documents(
                documents=pdf_docs,
                embedding=EMBED,
                collection_name="responsible_ai_papers",
                persist_directory=str(CHROMA_PERSIST_DIR)
            )
            print(f"↑ {len(pdf_docs)} docs → responsible_ai_papers")
        else:
            print("No PDF documents found in specified directory.")

    print("\nDone.")


if __name__ == "__main__":
    main()
