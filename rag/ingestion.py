import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Literal, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings, get_token_counter
from .vectorstore import get_vectorstore, read_manifest, write_manifest

logger = logging.getLogger("rag")

_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def _extract_title(text: str, is_markdown: bool) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("#").strip() if is_markdown else stripped
    return ""


def load_documents(data_dir) -> List[Document]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .txt/.md en {data_dir}")

    docs = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("archivo_vacio_ignorado: %s", path)
            continue
        source = path.as_posix()
        title = _extract_title(text, is_markdown=path.suffix == ".md")
        docs.append(Document(page_content=text, metadata={"source": source, "title": title}))
    return docs


def _warn_if_chunks_exceed_max_seq_length(chunks: List[Document], embedding_model: str) -> None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(embedding_model, local_files_only=True)
        max_len = getattr(tok, "model_max_length", 512) or 512
        if max_len > 100_000:
            max_len = 512
        for chunk in chunks:
            n = len(tok.encode(chunk.page_content, add_special_tokens=False))
            if n > max_len:
                logger.warning(
                    "chunk_excede_max_seq_length: chunk_id=%s tokens=%d max=%d",
                    chunk.metadata.get("chunk_id"),
                    n,
                    max_len,
                )
    except Exception:
        pass


def split_documents(docs: List[Document], settings, length_function: Optional[Callable[[str], int]] = None) -> List[Document]:
    using_custom_lf = length_function is not None
    lf = length_function or get_token_counter(settings.embedding_model)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=lf,
        separators=_SEPARATORS,
    )

    chunks: List[Document] = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        n_tokens_list = []
        for idx, piece in enumerate(pieces):
            n_tokens = lf(piece)
            n_tokens_list.append(n_tokens)
            chunk_id = f"{doc.metadata['source']}::{idx}"
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "chunk_id": chunk_id,
                        "n_tokens": n_tokens,
                    },
                )
            )
        if n_tokens_list:
            logger.info(
                "chunked_document: source=%s n_chunks=%d tokens_min=%d tokens_max=%d",
                doc.metadata["source"],
                len(n_tokens_list),
                min(n_tokens_list),
                max(n_tokens_list),
            )

    if using_custom_lf:
        _warn_if_chunks_exceed_max_seq_length(chunks, settings.embedding_model)

    return chunks


@dataclass
class IngestResult:
    status: Literal["indexed", "skipped"]
    n_documents: int
    n_chunks: int
    reason: str
    chunks_by_file: Optional[dict] = None


def _file_hashes(docs: List[Document]) -> dict:
    return {
        doc.metadata["source"]: hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        for doc in docs
    }


def _collection_count(settings) -> int:
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(settings.persist_dir))
        collection = client.get_collection(settings.collection_name)
        return collection.count()
    except Exception:
        return 0


def _reindex_reason(existing: Optional[dict], settings, file_hashes: dict, force: bool) -> str:
    if force:
        return "--force"
    if existing is None:
        return "sin índice previo"
    if existing.get("embedding_model") != settings.embedding_model:
        return "cambió embedding_model"
    if existing.get("chunk_size") != settings.chunk_size or existing.get("chunk_overlap") != settings.chunk_overlap:
        return "cambiaron parámetros de chunking"
    if existing.get("collection_name") != settings.collection_name:
        return "cambió collection_name"
    if existing.get("files") != file_hashes:
        changed = sorted(
            f for f in {**existing.get("files", {}), **file_hashes} if existing.get("files", {}).get(f) != file_hashes.get(f)
        )
        return f"cambiaron archivos: {changed}"
    return "índice inconsistente con el manifiesto"


def ingest(settings, *, force: bool = False, embeddings=None, length_function: Optional[Callable[[str], int]] = None) -> IngestResult:
    docs = load_documents(settings.data_dir)
    file_hashes = _file_hashes(docs)

    expected_params = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "collection_name": settings.collection_name,
        "files": file_hashes,
    }

    existing = read_manifest(settings)

    if not force and existing is not None:
        matches = all(existing.get(k) == v for k, v in expected_params.items())
        if matches:
            count = _collection_count(settings)
            if count > 0 and count == existing.get("n_chunks"):
                logger.info("ingest_skipped: índice vigente, n_chunks=%d", count)
                return IngestResult(
                    status="skipped",
                    n_documents=existing.get("n_documents", len(docs)),
                    n_chunks=count,
                    reason="índice vigente",
                )

    reason = _reindex_reason(existing, settings, file_hashes, force)

    embeddings = embeddings or get_embeddings(settings.embedding_model)
    chunks = split_documents(docs, settings, length_function=length_function)

    vectorstore = get_vectorstore(settings, embeddings)
    try:
        vectorstore.delete_collection()
    except Exception:
        logger.debug("delete_collection_noop", exc_info=True)
    vectorstore = get_vectorstore(settings, embeddings)

    ids = [c.metadata["chunk_id"] for c in chunks]
    vectorstore.add_documents(chunks, ids=ids)

    manifest = {
        **expected_params,
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    write_manifest(settings, manifest)

    chunks_by_file: dict = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        entry = chunks_by_file.setdefault(source, {"n_chunks": 0, "max_tokens": 0})
        entry["n_chunks"] += 1
        entry["max_tokens"] = max(entry["max_tokens"], chunk.metadata["n_tokens"])

    logger.info(
        "ingest_indexed: reason=%s n_documents=%d n_chunks=%d", reason, len(docs), len(chunks)
    )
    return IngestResult(
        status="indexed",
        n_documents=len(docs),
        n_chunks=len(chunks),
        reason=reason,
        chunks_by_file=chunks_by_file,
    )
