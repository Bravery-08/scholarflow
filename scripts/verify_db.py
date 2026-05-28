import asyncio
from db.postgres import LocalAsyncSessionLocal, local_async_engine
from db.chroma import get_chroma_client, get_collection, upsert_chunks, query_collection
from db.models import Paper
from sqlalchemy import select, text


async def test_postgres():
    print("Testing PostgreSQL...")

    # first confirm the connection itself works
    async with local_async_engine.connect() as conn:
        result = await conn.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"  Connected to: {version[:40]}...")

    # now test ORM insert + read + delete
    async with LocalAsyncSessionLocal() as session:
        paper = Paper(
            arxiv_id="TEST001",
            title="Test Paper",
            authors=["Author One"],
            abstract="This is a test.",
            pdf_url="https://example.com",
            status="done"
        )
        session.add(paper)
        await session.commit()

        result = await session.execute(
            select(Paper).where(Paper.arxiv_id == "TEST001")
        )
        fetched = result.scalar_one()
        print(f"  Inserted and fetched: {fetched}")

        await session.delete(fetched)
        await session.commit()

    print("  PostgreSQL: OK\n")


def test_chroma():
    print("Testing Chroma...")
    client = get_chroma_client(local=True)
    heartbeat = client.heartbeat()
    print(f"  Chroma heartbeat: {heartbeat}")

    upsert_chunks(
        ids=["test-chunk-0"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["This is a test chunk."],
        metadatas=[{"paper_id": "TEST001", "page_num": 0,
                    "has_figure": False, "chunk_index": 0}],
        local=True
    )

    results = query_collection(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        n_results=1,
        local=True
    )
    print(f"  Retrieved: {results['documents'][0][0]}")

    get_collection(local=True).delete(ids=["test-chunk-0"])
    print("  Chroma: OK")


if __name__ == "__main__":
    asyncio.run(test_postgres())
    test_chroma()