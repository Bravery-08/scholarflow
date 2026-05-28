import io
import base64
import logging
import tempfile
import requests
from pathlib import Path
from datetime import datetime, timezone

import arxiv
import fitz                          # PyMuPDF
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.config import settings
from core.embedder import get_embedder
from db.models import Paper, Chunk
from db.chroma import upsert_chunks

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("data/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
# discard chunks shorter than this (headers, page nums)
MIN_CHUNK_LENGTH = 80
IMAGE_DPI = 150              # resolution for figure page renders

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ── ArXiv fetching ────────────────────────────────────────────────────────────

def fetch_arxiv_papers(query: str, max_results: int = 10) -> list[dict]:
    """
    Search ArXiv and return paper metadata. Does not download PDFs.
    """
    client = arxiv.Client(
        page_size=min(max_results, 50),   # don't request 100 when you want 3
        delay_seconds=5.0,                # be polite
        num_retries=2
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in client.results(search):
        arxiv_id = result.entry_id.split("/abs/")[-1].split("v")[0]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": result.title,
            "abstract": result.summary.replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "pdf_url": result.pdf_url
        })
    return papers


def fetch_paper_by_id(arxiv_id: str) -> dict:
    """
    Fetches a single paper by its ArXiv ID.
    Uses the id_list endpoint which is lighter and less rate-limited
    than a text search query.
    """
    client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))
    clean_id = result.entry_id.split("/abs/")[-1].split("v")[0]
    return {
        "arxiv_id": clean_id,
        "title": result.title,
        "abstract": result.summary.replace("\n", " "),
        "authors": [a.name for a in result.authors],
        "pdf_url": result.pdf_url
    }


# ── PDF download ──────────────────────────────────────────────────────────────

def download_pdf(arxiv_id: str, pdf_url: str, dest_dir: Path | None = None) -> Path:
    """
    Downloads a PDF and saves it to dest_dir (or a system temp dir).
    Returns the path to the saved file.
    """
    if dest_dir is None:
        dest_dir = Path(tempfile.gettempdir())

    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir / f"{arxiv_id}.pdf"

    if pdf_path.exists():
        logger.info(f"PDF already cached: {pdf_path}")
        return pdf_path

    logger.info(f"Downloading PDF: {pdf_url}")
    headers = {"User-Agent": "ScholarFlow/1.0 (research project)"}
    response = requests.get(pdf_url, headers=headers, timeout=60)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)
    logger.info(f"Saved to {pdf_path} ({len(response.content) // 1024} KB)")
    return pdf_path


# ── Text extraction and chunking ──────────────────────────────────────────────

def extract_text_chunks(pdf_path: Path) -> list[dict]:
    """
    Extracts text from every page using pdfplumber,
    splits into overlapping chunks, returns list of chunk dicts.
    """
    raw_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > MIN_CHUNK_LENGTH:
                raw_pages.append({"page_num": page_num, "text": text.strip()})

    if not raw_pages:
        logger.warning(f"No text extracted from {pdf_path}")
        return []

    chunks = []
    chunk_index = 0

    for page in raw_pages:
        page_chunks = _splitter.split_text(page["text"])
        for text in page_chunks:
            if len(text.strip()) < MIN_CHUNK_LENGTH:
                continue
            chunks.append({
                "chunk_index": chunk_index,
                "text": text.strip(),
                "page_num": page["page_num"],
                "has_figure": False    # updated below after figure detection
            })
            chunk_index += 1

    logger.info(f"Extracted {len(chunks)} text chunks from {pdf_path.name}")
    return chunks


# ── Figure detection and extraction ───────────────────────────────────────────

def extract_figure_pages(pdf_path: Path, arxiv_id: str) -> list[dict]:
    """
    Uses PyMuPDF to find pages that contain embedded images (figures).
    Renders those pages as PNG and saves them to disk.
    Returns list of {page_num, image_path, image_b64}.
    """
    figure_pages = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=False)

        if not images:
            continue

        # render the full page as an image at IMAGE_DPI
        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")

        # save to disk for Phase 4 (visual agent)
        image_path = FIGURES_DIR / f"{arxiv_id}_page_{page_num}.png"
        image_path.write_bytes(png_bytes)

        b64 = base64.b64encode(png_bytes).decode("utf-8")
        figure_pages.append({
            "page_num": page_num,
            "image_path": str(image_path),
            "image_b64": b64,
            "image_count": len(images)
        })

    doc.close()
    logger.info(
        f"Found {len(figure_pages)} figure pages in {pdf_path.name}"
    )
    return figure_pages


def _mark_figure_chunks(chunks: list[dict], figure_pages: list[dict]) -> list[dict]:
    """
    Sets has_figure=True on any chunk whose page_num appears in figure_pages.
    """
    figure_page_nums = {fp["page_num"] for fp in figure_pages}
    for chunk in chunks:
        if chunk["page_num"] in figure_page_nums:
            chunk["has_figure"] = True
    return chunks


# ── Postgres storage ──────────────────────────────────────────────────────────

def save_paper(session: Session, paper_data: dict) -> Paper | None:
    """
    Inserts a Paper row if it doesn't already exist.
    Returns the Paper ORM object, or None if it already existed.
    """
    existing = session.execute(
        select(Paper).where(Paper.arxiv_id == paper_data["arxiv_id"])
    ).scalar_one_or_none()

    if existing:
        logger.info(f"Paper {paper_data['arxiv_id']} already in DB, skipping.")
        return None

    paper = Paper(
        arxiv_id=paper_data["arxiv_id"],
        title=paper_data["title"],
        authors=paper_data["authors"],
        abstract=paper_data["abstract"],
        pdf_url=paper_data["pdf_url"],
        status="processing"
    )
    session.add(paper)
    session.flush()      # get the auto-incremented ID without committing yet
    return paper


def save_chunks(session: Session, paper: Paper,
                chunks: list[dict], chroma_ids: list[str]) -> None:
    """
    Inserts Chunk rows into Postgres, linking each to its Chroma ID.
    """
    for chunk_data, chroma_id in zip(chunks, chroma_ids):
        chunk = Chunk(
            paper_id=paper.id,
            chunk_index=chunk_data["chunk_index"],
            text=chunk_data["text"],
            page_num=chunk_data["page_num"],
            has_figure=chunk_data["has_figure"],
            chroma_id=chroma_id
        )
        session.add(chunk)


# ── Chroma storage ────────────────────────────────────────────────────────────

def embed_and_store_chunks(
    arxiv_id: str,
    chunks: list[dict],
    local: bool = False
) -> list[str]:
    """
    Embeds all chunk texts with Harrier and upserts into Chroma.
    Returns the list of Chroma IDs (one per chunk).
    """
    if not chunks:
        return []

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed(texts)

    # Chroma IDs must be unique strings — use arxiv_id + chunk index
    chroma_ids = [f"{arxiv_id}-chunk-{c['chunk_index']}" for c in chunks]

    metadatas = [
        {
            "paper_id": arxiv_id,
            "page_num": c["page_num"],
            "has_figure": c["has_figure"],
            "chunk_index": c["chunk_index"]
        }
        for c in chunks
    ]

    upsert_chunks(
        ids=chroma_ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
        local=local
    )
    logger.info(f"Upserted {len(chroma_ids)} chunks to Chroma")
    return chroma_ids


# ── Top-level pipeline ────────────────────────────────────────────────────────

def ingest_paper(
    paper_data: dict,
    session: Session,
    local: bool = False
) -> dict:
    """
    Full ingestion pipeline for one paper.
    Downloads PDF, extracts + embeds chunks, saves to Postgres and Chroma.

    Returns a summary dict with counts and status.
    """
    arxiv_id = paper_data["arxiv_id"]
    logger.info(f"Starting ingestion: {arxiv_id}")

    # 1. Save paper to Postgres (status=processing)
    paper = save_paper(session, paper_data)
    if paper is None:
        return {"arxiv_id": arxiv_id, "status": "skipped", "reason": "already exists"}

    try:
        # 2. Download PDF
        pdf_path = download_pdf(arxiv_id, paper_data["pdf_url"])

        # 3. Extract text chunks
        chunks = extract_text_chunks(pdf_path)
        if not chunks:
            raise ValueError("No text could be extracted from this PDF")

        # 4. Detect figure pages and render images
        figure_pages = extract_figure_pages(pdf_path, arxiv_id)

        # 5. Mark chunks that share a page with a figure
        chunks = _mark_figure_chunks(chunks, figure_pages)

        # 6. Embed and store in Chroma
        chroma_ids = embed_and_store_chunks(arxiv_id, chunks, local=local)

        # 7. Save chunks to Postgres
        save_chunks(session, paper, chunks, chroma_ids)

        # 8. Mark paper as done
        paper.status = "done"
        paper.ingested_at = datetime.now(timezone.utc)
        session.flush()

        logger.info(f"Ingestion complete: {arxiv_id} "
                    f"({len(chunks)} chunks, {len(figure_pages)} figure pages)")

        return {
            "arxiv_id": arxiv_id,
            "status": "done",
            "chunks": len(chunks),
            "figure_pages": len(figure_pages),
            "figure_page_nums": [fp["page_num"] for fp in figure_pages]
        }

    except Exception as e:
        paper.status = "failed"
        paper.error_message = str(e)
        session.flush()
        logger.error(f"Ingestion failed for {arxiv_id}: {e}")
        raise
