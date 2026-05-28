from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

from core.config import settings
from core.embedder import get_embedder
from core.schemas import FigureCaption
from db.chroma import upsert_chunks

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("data/figures")

# seconds to wait between Nemotron requests — it's a heavy reasoning model
REQUEST_DELAY = 8.0


_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                # OpenRouter uses these for analytics — fill in your project name
                "HTTP-Referer": "https://github.com/scholarflow",
                "X-Title": "ScholarFlow"
            }
        )
    return _client


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Extracts the last JSON object from a string.
    Handles markdown fences and reasoning model preamble.
    """
    # strip markdown fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # find the last {...} block — reasoning models often write the answer last
    matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL))
    if not matches:
        raise ValueError(f"No JSON object found in response:\n{text[:300]}")

    raw = matches[-1].group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\nRaw: {raw[:300]}")


# ── Single figure analysis ────────────────────────────────────────────────────

def analyze_figure(
    image_b64: str,
    paper_title: str,
    page_num: int,
    retries: int = 2
) -> FigureCaption | None:
    """
    Sends one figure page to Nemotron via OpenRouter.
    Returns a FigureCaption or None if the model fails to produce valid output.
    """
    client = _get_client()

    prompt = f"""You are analyzing a figure from a research paper titled: "{paper_title}"
This is page {page_num} of the paper.

Analyze the figure carefully and respond with ONLY a JSON object — no preamble, no explanation outside the JSON.

Required JSON format:
{{
  "description": "detailed description of what this figure shows",
  "data_type": "chart|diagram|table|photo|equation|other",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "searchable_keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
}}

Rules:
- key_findings: maximum 3 items, each a complete sentence
- searchable_keywords: 3-6 specific technical terms from the figure
- data_type: pick exactly one from the allowed values
- Output ONLY the JSON object, nothing else"""

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.visual_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=2048    # reasoning model needs room to think first
            )

            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", "") or ""

            # prefer content; fall back to extracting from reasoning
            source = content if content.strip() else reasoning
            if not source.strip():
                logger.warning(f"Empty response for page {page_num}, attempt {attempt + 1}")
                if attempt < retries:
                    time.sleep(REQUEST_DELAY)
                    continue
                return None

            data = _extract_json(source)

            # validate data_type before Pydantic sees it
            allowed = {"chart", "diagram", "table", "photo", "equation", "other"}
            if data.get("data_type") not in allowed:
                data["data_type"] = "other"

            # clamp lists to expected sizes
            data["key_findings"] = data.get("key_findings", [])[:3]
            data["searchable_keywords"] = data.get("searchable_keywords", [])[:6]

            return FigureCaption(**data)

        except RateLimitError:
            wait = REQUEST_DELAY * (attempt + 2)
            logger.warning(f"Rate limited by OpenRouter, waiting {wait}s...")
            time.sleep(wait)
        except (ValueError, KeyError) as e:
            logger.warning(f"Parse error on page {page_num}, attempt {attempt + 1}: {e}")
            if attempt < retries:
                time.sleep(REQUEST_DELAY)

    logger.error(f"All attempts failed for page {page_num}")
    return None


# ── Caption → searchable text ─────────────────────────────────────────────────

def caption_to_text(caption: FigureCaption, arxiv_id: str, page_num: int) -> str:
    """
    Converts a FigureCaption into a single searchable text string
    for embedding and storage in Chroma.
    """
    findings = " ".join(caption.key_findings)
    keywords = ", ".join(caption.searchable_keywords)
    return (
        f"[FIGURE] Page {page_num} of paper {arxiv_id}. "
        f"Type: {caption.data_type.value}. "
        f"{caption.description} "
        f"Key findings: {findings} "
        f"Keywords: {keywords}"
    )


# ── Process all figures for one paper ────────────────────────────────────────

def process_paper_figures(
    arxiv_id: str,
    paper_title: str,
    local: bool = False
) -> list[dict]:
    """
    Finds all figure PNGs for a paper, analyzes each with Nemotron,
    embeds the captions, and upserts them into Chroma.

    Returns a list of result dicts (one per figure page).
    """
    # find all saved figure PNGs for this paper
    figure_files = sorted(FIGURES_DIR.glob(f"{arxiv_id}_page_*.png"))

    if not figure_files:
        logger.warning(f"No figure files found for {arxiv_id} in {FIGURES_DIR}")
        return []

    logger.info(f"Processing {len(figure_files)} figure pages for {arxiv_id}")

    embedder = get_embedder()
    results = []

    for fig_path in figure_files:
        # extract page number from filename: "1706.03762_page_3.png" → 3
        page_num = int(fig_path.stem.split("_page_")[-1])

        # check if already processed in Chroma
        chroma_id = f"{arxiv_id}-figure-{page_num}"

        logger.info(f"  Analyzing figure: {fig_path.name}")
        image_b64 = fig_path.read_bytes()
        import base64
        image_b64_str = base64.b64encode(image_b64).decode("utf-8")

        caption = analyze_figure(image_b64_str, paper_title, page_num)

        if caption is None:
            logger.warning(f"  Skipping {fig_path.name} — no valid caption produced")
            results.append({
                "page_num": page_num,
                "status": "failed",
                "chroma_id": None
            })
            # still wait between requests even on failure
            time.sleep(REQUEST_DELAY)
            continue

        # convert to searchable text and embed
        text = caption_to_text(caption, arxiv_id, page_num)
        vector = embedder.embed_one(text)

        upsert_chunks(
            ids=[chroma_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{
                "paper_id": arxiv_id,
                "page_num": page_num,
                "has_figure": True,
                "chunk_index": -1,           # -1 signals this is a figure chunk
                "data_type": caption.data_type.value
            }],
            local=local
        )

        results.append({
            "page_num": page_num,
            "status": "done",
            "chroma_id": chroma_id,
            "data_type": caption.data_type.value,
            "description": caption.description[:80],
            "keywords": caption.searchable_keywords
        })

        logger.info(f"  Stored figure caption: page {page_num} ({caption.data_type.value})")

        # rate limit compliance — wait between requests
        time.sleep(REQUEST_DELAY)

    done = sum(1 for r in results if r["status"] == "done")
    logger.info(f"Figure processing complete: {done}/{len(results)} succeeded")
    return results