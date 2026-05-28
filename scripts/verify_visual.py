import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
# suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from pathlib import Path
from agents.visual import process_paper_figures
from core.embedder import get_embedder
from db.chroma import query_collection

ARXIV_ID = "1706.03762"
PAPER_TITLE = "Attention Is All You Need"
FIGURES_DIR = Path("data/figures")


def check_figures_exist():
    print("\n── Checking saved figures ───────────────────")
    files = sorted(FIGURES_DIR.glob(f"{ARXIV_ID}_page_*.png"))
    if not files:
        print(f"  ERROR: No figures found in {FIGURES_DIR}")
        print("  Run verify_ingestion.py first to generate them.")
        raise SystemExit(1)
    print(f"  Found {len(files)} figure PNGs:")
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"    · {f.name} ({size_kb} KB)")
    return files


def test_visual_agent(figure_files: list[Path]):
    print(f"\n── Visual analysis ({len(figure_files)} figures) ───────────────")
    print(f"  Model : {__import__('core.config', fromlist=['settings']).settings.visual_model}")
    print(f"  Note  : each figure takes ~8s (reasoning model + rate limit delay)\n")

    results = process_paper_figures(ARXIV_ID, PAPER_TITLE, local=True)

    done = [r for r in results if r["status"] == "done"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n  Processed : {len(results)} figures")
    print(f"  Succeeded : {len(done)}")
    print(f"  Failed    : {len(failed)}")

    for r in done:
        print(f"\n  Page {r['page_num']} [{r['data_type']}]")
        print(f"    Description : {r['description']}...")
        print(f"    Keywords    : {r['keywords']}")
        print(f"    Chroma ID   : {r['chroma_id']}")

    assert len(done) > 0, "All figure analyses failed — check OpenRouter key and model"
    return done


def test_figure_retrieval(done_results: list[dict]):
    print("\n── Figure retrieval test ────────────────────")
    embedder = get_embedder()

    queries = [
        "model architecture diagram encoder decoder",
        "attention score visualization heatmap",
        "training loss performance comparison chart"
    ]

    for query in queries:
        vec = embedder.embed_one(query)
        results = query_collection(
            query_embedding=vec,
            n_results=3,
            local=True
        )

        docs = results["documents"][0]
        dists = results["distances"][0]

        # find if any figure chunk came back
        figure_hits = [
            (doc[:80], dist)
            for doc, dist in zip(docs, dists)
            if doc.startswith("[FIGURE]")
        ]

        print(f"\n  Query : '{query}'")
        if figure_hits:
            print(f"  Figure hit : '{figure_hits[0][0]}...'")
            print(f"  Distance   : {figure_hits[0][1]:.4f}")
        else:
            print("  No figure chunks in top 3 (text chunks ranked higher — OK)")

    print("\n  Retrieval: OK")


if __name__ == "__main__":
    figure_files = check_figures_exist()
    done = test_visual_agent(figure_files)
    test_figure_retrieval(done)