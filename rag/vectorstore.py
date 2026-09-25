import json
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma

from .exceptions import EmbeddingMismatchError, IndexNotFoundError


def get_vectorstore(settings, embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def manifest_path(settings) -> Path:
    return Path(settings.persist_dir) / "ingest_manifest.json"


def read_manifest(settings) -> Optional[dict]:
    path = manifest_path(settings)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(settings, data: dict) -> None:
    path = manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def assert_index_compatible(settings, vectorstore: Chroma) -> None:
    manifest = read_manifest(settings)
    if manifest is None:
        raise IndexNotFoundError(
            "No existe un índice válido en PERSIST_DIR. Ejecutar `python ingest.py`."
        )
    if manifest.get("embedding_model") != settings.embedding_model:
        raise EmbeddingMismatchError(
            f"El índice fue creado con el modelo '{manifest.get('embedding_model')}' "
            f"pero la configuración actual usa '{settings.embedding_model}'. "
            "Ejecutar `python ingest.py --force`."
        )
    count = vectorstore._collection.count()
    if count == 0 or count != manifest.get("n_chunks"):
        raise IndexNotFoundError(
            "El índice existente no coincide con el manifiesto. Ejecutar `python ingest.py`."
        )
