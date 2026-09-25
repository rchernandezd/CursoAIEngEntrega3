class IndexNotFoundError(RuntimeError):
    """No existe un índice válido en PERSIST_DIR. Ejecutar `python ingest.py`."""


class EmbeddingMismatchError(RuntimeError):
    """El índice fue creado con otro modelo de embeddings. Ejecutar `python ingest.py --force`."""
