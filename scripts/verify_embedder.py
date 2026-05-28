import logging
logging.basicConfig(level=logging.INFO)

from core.embedder import get_embedder
from db.chroma import upsert_chunks, query_collection, get_collection

def test_embedder():
    print("\nTesting embedder...")
    embedder = get_embedder()
    print(f"  Model     : {embedder.model.model_card_data.base_model or 'loaded'}")
    print(f"  Dimension : {embedder.dimension}")

    # embed a few sentences and sanity-check the output
    sentences = [
        "Transformers use self-attention to process sequences.",
        "Graph neural networks operate on non-Euclidean data.",
        "The mitochondria is the powerhouse of the cell."
    ]
    vectors = embedder.embed(sentences)

    assert len(vectors) == 3, "Wrong number of vectors returned"
    assert len(vectors[0]) == embedder.dimension, "Dimension mismatch"

    # check normalization — magnitude of each vector should be ~1.0
    import math
    for i, vec in enumerate(vectors):
        magnitude = math.sqrt(sum(x ** 2 for x in vec))
        assert 0.999 < magnitude < 1.001, f"Vector {i} not normalized: {magnitude}"

    print(f"  Vectors   : {len(vectors)} × {len(vectors[0])} ✓")
    print(f"  Normalized: all magnitudes ≈ 1.0 ✓")
    print("  Embedder: OK\n")
    return embedder


def test_chroma_with_real_embeddings(embedder):
    print("Testing Chroma with real embeddings...")

    texts = [
        "Attention is All You Need introduced the transformer architecture.",
        "BERT uses bidirectional transformers for language understanding.",
        "GPT models are autoregressive language models trained on large corpora.",
    ]
    vectors = embedder.embed(texts)

    upsert_chunks(
        ids=["e2e-chunk-0", "e2e-chunk-1", "e2e-chunk-2"],
        embeddings=vectors,
        documents=texts,
        metadatas=[
            {"paper_id": "1706.03762", "page_num": 0, "has_figure": False, "chunk_index": 0},
            {"paper_id": "1810.04805", "page_num": 0, "has_figure": False, "chunk_index": 0},
            {"paper_id": "2005.14165", "page_num": 0, "has_figure": False, "chunk_index": 0},
        ],
        local=True
    )
    print("  Upserted 3 chunks ✓")

    # query with a semantically related sentence (not in the index)
    query = "How do transformer models process input sequences?"
    query_vec = embedder.embed_one(query)

    results = query_collection(
        query_embedding=query_vec,
        n_results=2,
        local=True
    )

    print(f"  Query     : '{query}'")
    print(f"  Top match : '{results['documents'][0][0][:60]}...'")
    print(f"  Distance  : {results['distances'][0][0]:.4f}")

    # the top result should be semantically close (low cosine distance)
    assert results['distances'][0][0] < 0.5, \
        f"Top result too distant: {results['distances'][0][0]}"

    # clean up
    get_collection(local=True).delete(ids=["e2e-chunk-0", "e2e-chunk-1", "e2e-chunk-2"])
    print("  Chroma roundtrip: OK\n")


if __name__ == "__main__":
    embedder = test_embedder()
    test_chroma_with_real_embeddings(embedder)