from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import settings

_docker_client = None
_local_client = None
_collection = None

COLLECTION_NAME = "scholarflow"


def _make_client(host: str) -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_chroma_client(local: bool = False) -> chromadb.HttpClient:
    """
    local=False → uses CHROMA_HOST (Docker service name, for containers)
    local=True  → uses CHROMA_HOST_LOCAL (localhost, for local scripts)
    """
    global _docker_client, _local_client

    if local:
        if _local_client is None:
            _local_client = _make_client(settings.chroma_host_local)
        return _local_client
    else:
        if _docker_client is None:
            _docker_client = _make_client(settings.chroma_host)
        return _docker_client


def get_collection(local: bool = False):
    global _collection
    if _collection is None:
        client = get_chroma_client(local=local)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def upsert_chunks(ids: list[str], embeddings: list[list[float]],
                  documents: list[str], metadatas: list[dict],
                  local: bool = False):
    collection = get_collection(local=local)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def query_collection(query_embedding: list[float], n_results: int = 8,
                     where: dict | None = None,
                     local: bool = False) -> dict:
    collection = get_collection(local=local)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )