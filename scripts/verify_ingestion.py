import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)

from db.postgres import get_sync_session
from db.chroma import query_collection, get_collection
from db.models import Paper, Chunk
from agents.ingestion import (
    fetch_arxiv_papers,
    ingest_paper,
)
from core.embedder import get_embedder
from sqlalchemy import select
from agents.ingestion import (
    fetch_paper_by_id,
    ingest_paper,
)


ARXIV_ID = "1706.03762"   # Attention is All You Need


def test_fetch():
    print("\n── Fetch test ───────────────────────────────")
    # ID-based lookup — much lighter than a text search query
    paper_data = fetch_paper_by_id(ARXIV_ID)
    print(f"  Title   : {paper_data['title'][:60]}")
    print(f"  Authors : {paper_data['authors'][:2]}")
    print(f"  PDF URL : {paper_data['pdf_url']}")
    assert paper_data["arxiv_id"] == ARXIV_ID
    print("  Fetch: OK")
    return paper_data


def test_ingest(paper_data: dict):
    print("\n── Ingestion test ───────────────────────────")
    with get_sync_session(local=True) as session:
        result = ingest_paper(paper_data, session, local=True)

    print(f"  Status      : {result['status']}")
    if result["status"] == "skipped":
        print("  (paper already ingested — delete it from DB to re-run)")
        return
    print(f"  Chunks      : {result['chunks']}")
    print(f"  Figure pages: {result['figure_pages']}")
    print(f"  Fig page #s : {result['figure_page_nums']}")
    assert result["chunks"] > 0, "No chunks produced"
    print("  Ingestion: OK")


def test_postgres_state():
    print("\n── Postgres state ───────────────────────────")
    with get_sync_session(local=True) as session:
        paper = session.execute(
            select(Paper).where(Paper.arxiv_id == ARXIV_ID)
        ).scalar_one_or_none()

        assert paper is not None, f"Paper {ARXIV_ID} not found in DB"
        print(f"  Paper   : {paper.title[:60]}")
        print(f"  Status  : {paper.status}")
        print(f"  Authors : {paper.authors[:2]}")

        chunks = session.execute(
            select(Chunk).where(Chunk.paper_id == paper.id)
        ).scalars().all()

        print(f"  Chunks  : {len(chunks)}")
        figure_chunks = [c for c in chunks if c.has_figure]
        print(f"  Fig chunks: {len(figure_chunks)}")

        # spot-check one chunk
        sample = chunks[0]
        print(f"  Sample chunk[0]: '{sample.text[:80]}...'")
        print(f"  Chroma ID: {sample.chroma_id}")

    print("  Postgres state: OK")


def test_retrieval():
    print("\n── Retrieval test ───────────────────────────")
    embedder = get_embedder()

    queries = [
        "How does the attention mechanism work?",
        "What optimizer was used for training?",
        "What is the BLEU score result?"
    ]

    for query in queries:
        vec = embedder.embed_one(query)
        results = query_collection(
            query_embedding=vec,
            n_results=2,
            where={"paper_id": ARXIV_ID},
            local=True
        )
        top_doc = results["documents"][0][0]
        top_dist = results["distances"][0][0]
        print(f"\n  Query : {query}")
        print(f"  Match : {top_doc[:100]}...")
        print(f"  Dist  : {top_dist:.4f}")
        assert top_dist < 0.6, f"Match too distant: {top_dist}"

    print("\n  Retrieval: OK")


if __name__ == "__main__":
    paper_data = test_fetch()
    test_ingest(paper_data)
    test_postgres_state()
    test_retrieval()