import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.ingestion import ingest, load_documents, split_documents
from rag.vectorstore import get_vectorstore, read_manifest
from .conftest import make_settings


def _word_length(text: str) -> int:
    return len(text.split())


def _write_data(data_dir, files: dict):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (data_dir / name).write_text(content, encoding="utf-8")


def _embeddings():
    return DeterministicFakeEmbedding(size=384)


LONG_TEXT = "Política de Prueba - TechCorp\n\n" + (
    "Esta es una oración de relleno para generar múltiples fragmentos de prueba. "
) * 400


@pytest.fixture
def data_settings(base_settings_kwargs):
    return make_settings(base_settings_kwargs)


def test_primera_ingesta_indexa(data_settings):
    _write_data(data_settings.data_dir, {"doc1.txt": "Contenido corto de prueba.\nSegunda línea."})
    result = ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)

    assert result.status == "indexed"
    assert result.n_chunks > 0

    vs = get_vectorstore(data_settings, _embeddings())
    assert vs._collection.count() == result.n_chunks

    manifest = read_manifest(data_settings)
    assert manifest is not None
    assert manifest["n_chunks"] == result.n_chunks


def test_segunda_ingesta_sin_cambios_se_saltea(data_settings):
    _write_data(data_settings.data_dir, {"doc1.txt": "Contenido corto de prueba.\nSegunda línea."})
    first = ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)
    second = ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)

    assert second.status == "skipped"
    vs = get_vectorstore(data_settings, _embeddings())
    assert vs._collection.count() == first.n_chunks


def test_modificar_archivo_reindexa_y_menciona_el_archivo(data_settings):
    _write_data(data_settings.data_dir, {"doc1.txt": "Contenido original.\nSegunda línea."})
    ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)

    _write_data(data_settings.data_dir, {"doc1.txt": "Contenido MODIFICADO por completo.\nOtra línea."})
    result = ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)

    assert result.status == "indexed"
    assert "doc1.txt" in result.reason


def test_cambio_de_embedding_model_reindexa(data_settings):
    _write_data(data_settings.data_dir, {"doc1.txt": "Contenido corto de prueba.\nSegunda línea."})
    ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)

    other_settings = make_settings(
        dict(
            data_dir=data_settings.data_dir,
            persist_dir=data_settings.persist_dir,
            collection_name=data_settings.collection_name,
            embedding_model="otro-modelo-distinto",
            chunk_size=data_settings.chunk_size,
            chunk_overlap=data_settings.chunk_overlap,
            top_k=data_settings.top_k,
            llm_model=data_settings.llm_model,
            llm_temperature=data_settings.llm_temperature,
            llm_timeout_s=data_settings.llm_timeout_s,
            max_concurrency=data_settings.max_concurrency,
        )
    )
    result = ingest(other_settings, embeddings=_embeddings(), length_function=_word_length)
    assert result.status == "indexed"
    assert "embedding_model" in result.reason


def test_force_reindexa_aunque_no_haya_cambios(data_settings):
    _write_data(data_settings.data_dir, {"doc1.txt": "Contenido corto de prueba.\nSegunda línea."})
    ingest(data_settings, embeddings=_embeddings(), length_function=_word_length)
    result = ingest(data_settings, force=True, embeddings=_embeddings(), length_function=_word_length)

    assert result.status == "indexed"
    assert result.reason == "--force"


def test_metadata_de_chunks(data_settings):
    _write_data(data_settings.data_dir, {"politica_x.txt": "Política X - TechCorp\n\nContenido de la política."})
    docs = load_documents(data_settings.data_dir)
    chunks = split_documents(docs, data_settings, length_function=_word_length)

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert "\\" not in chunk.metadata["source"]
    assert chunk.metadata["source"].endswith("politica_x.txt")
    assert chunk.metadata["title"] == "Política X - TechCorp"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["chunk_id"] == f"{chunk.metadata['source']}::0"


def test_documento_largo_produce_multiples_chunks_con_overlap(data_settings):
    _write_data(data_settings.data_dir, {"largo.txt": LONG_TEXT})
    docs = load_documents(data_settings.data_dir)
    chunks = split_documents(docs, data_settings, length_function=_word_length)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _word_length(chunk.page_content) <= data_settings.chunk_size

    words_a = chunks[0].page_content.split()
    words_b = chunks[1].page_content.split()
    overlap = set(words_a[-15:]) & set(words_b[:15])
    assert len(overlap) > 0


def test_data_dir_vacio_lanza_filenotfounderror(data_settings):
    data_settings.data_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        load_documents(data_settings.data_dir)
