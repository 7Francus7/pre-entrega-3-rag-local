"""Índice local versionado por contenido. Ejecutar con python ingesta.py."""
import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STORE = ROOT / "vectorstore"
MODEL = "minishlab/potion-multilingual-128M"
SIZE, OVERLAP = 550, 50
LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def embedder():
    return TextEmbedding(MODEL, cache_dir=str(ROOT / ".models"))


def ingest(data: Path = DATA, store: Path = STORE) -> int:
    files = {p.relative_to(data).as_posix(): p.read_text(encoding="utf-8")
             for p in sorted(data.rglob("*")) if p.is_file() and p.suffix.lower() in {".md", ".txt"}}
    if not any(value.strip() for value in files.values()):
        raise ValueError("La carpeta data debe contener documentos con texto.")
    signature = json.dumps([MODEL, SIZE, OVERLAP, "recursive-cl100k-v1", files], sort_keys=True)
    name = "rag-" + hashlib.sha256(signature.encode()).hexdigest()[:32]
    client = chromadb.PersistentClient(path=str(store))
    collection = client.get_or_create_collection(name, embedding_function=None,
                                                configuration={"hnsw": {"space": "cosine"}})
    if (collection.metadata or {}).get("listo"):
        LOG.info("Colección existente, se reutilizan %d fragmentos", collection.count())
    else:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=SIZE, chunk_overlap=OVERLAP,
            separators=[" ", "\n", ""], keep_separator=True, disallowed_special=(),
        )
        ids, texts, metadata = [], [], []
        for filename, content in files.items():
            for number, chunk in enumerate(splitter.split_text(content), 1):
                ids.append(f"{filename}#{number}")
                texts.append(chunk)
                metadata.append({"archivo": filename, "fragmento": number})
        for offset in range(0, len(ids), 64):
            batch = slice(offset, offset + 64)
            vectors = list(embedder().passage_embed(texts[batch]))
            collection.upsert(ids=ids[batch], documents=texts[batch],
                              metadatas=metadata[batch], embeddings=vectors)
        collection.modify(metadata={"listo": True, "modelo": MODEL})
        LOG.info("Indexados %d fragmentos de %d documentos", len(ids), len(files))
    with tempfile.NamedTemporaryFile(mode="w", dir=store, delete=False) as pending:
        pending.write(name)
    os.replace(pending.name, store / "activa")
    return collection.count()


def retrieve(query: str, store: Path = STORE) -> dict:
    active = store / "activa"
    if not active.exists():
        raise RuntimeError("Falta el índice. Ejecutá python ingesta.py.")
    collection = chromadb.PersistentClient(path=str(store)).get_collection(
        active.read_text(), embedding_function=None)
    if collection.metadata.get("modelo") != MODEL:
        raise RuntimeError("Cambió el modelo de embeddings. Ejecutá python ingesta.py.")
    hits = collection.query(query_embeddings=list(embedder().query_embed(query)),
                            n_results=min(3, collection.count()), include=["documents"])
    return dict(zip(hits["ids"][0], hits["documents"][0], strict=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest()
