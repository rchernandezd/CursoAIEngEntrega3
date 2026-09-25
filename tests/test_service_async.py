import asyncio
import inspect
import time

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag.exceptions import EmbeddingMismatchError, IndexNotFoundError
from rag.service import RAGService, get_rag_response
from rag.vectorstore import write_manifest
from .conftest import make_settings


class _FakeCollection:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


class _FakeVectorstore:
    def __init__(self, docs_with_scores, count):
        self._docs = docs_with_scores
        self._collection = _FakeCollection(count)

    def similarity_search_with_relevance_scores(self, query, k):
        return self._docs[:k]

    async def asimilarity_search_with_relevance_scores(self, query, k):
        return self._docs[:k]


def _docs():
    return [
        (Document(page_content="Los empleados con 5 años acceden a 21 días.", metadata={"source": "data/politica_vacaciones.txt", "title": "Vacaciones", "chunk_id": "data/politica_vacaciones.txt::0"}), 0.9),
        (Document(page_content="Otro fragmento.", metadata={"source": "data/otro.txt", "title": "Otro", "chunk_id": "data/otro.txt::0"}), 0.8),
        (Document(page_content="Tercer fragmento.", metadata={"source": "data/tercero.txt", "title": "Tercero", "chunk_id": "data/tercero.txt::0"}), 0.7),
    ]


def _write_valid_manifest(settings, n_chunks):
    write_manifest(
        settings,
        {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "collection_name": settings.collection_name,
            "files": {},
            "n_documents": 1,
            "n_chunks": n_chunks,
            "created_at": "2026-01-01T00:00:00-03:00",
        },
    )


def _slow_llm(content: str, delay: float):
    async def _ainvoke(prompt_value, config=None, **kwargs):
        await asyncio.sleep(delay)
        return AIMessage(content=content)

    def _invoke(prompt_value, config=None, **kwargs):
        raise NotImplementedError("solo async en este fake")

    return RunnableLambda(_invoke, afunc=_ainvoke)


def _make_service(settings, llm, docs=None, count=3):
    docs = docs if docs is not None else _docs()
    _write_valid_manifest(settings, n_chunks=count)
    vectorstore = _FakeVectorstore(docs, count=count)
    return RAGService(settings=settings, llm=llm, embeddings=object(), vectorstore=vectorstore)


def test_get_rag_response_es_corutina():
    assert inspect.iscoroutinefunction(get_rag_response)


async def test_aquery_encontrada_construye_fuentes_desde_metadata_real(base_settings_kwargs):
    settings = make_settings(base_settings_kwargs)
    llm = FakeListChatModel(
        responses=['{"encontrada": true, "respuesta": "21 días.", "fragmentos_citados": [1]}']
    )
    service = _make_service(settings, llm)

    resp = await service.aquery("¿Cuántos días de vacaciones?")

    assert resp.encontrada is True
    assert len(resp.fuentes) == 1
    assert resp.fuentes[0].archivo == "data/politica_vacaciones.txt"
    assert resp.fuentes[0].chunk_id == "data/politica_vacaciones.txt::0"


async def test_aquery_no_encontrada_normaliza_respuesta(base_settings_kwargs):
    settings = make_settings(base_settings_kwargs)
    llm = FakeListChatModel(
        responses=['{"encontrada": false, "respuesta": "Un texto libre cualquiera.", "fragmentos_citados": []}']
    )
    service = _make_service(settings, llm)

    resp = await service.aquery("¿Bonos?")

    assert resp.encontrada is False
    assert resp.fuentes == []
    assert resp.respuesta.startswith("No lo sé")


async def test_aquery_falla_closed_ante_salida_siempre_invalida(base_settings_kwargs):
    settings = make_settings(base_settings_kwargs)
    llm = FakeListChatModel(
        responses=['{"encontrada": true, "respuesta": "x", "fragmentos_citados": []}'] * 5
    )
    service = _make_service(settings, llm)

    resp = await service.aquery("¿algo?")

    assert resp.encontrada is False
    assert resp.fuentes == []
    assert resp.respuesta.startswith("No lo sé")


async def test_aquery_query_vacia_lanza_valueerror(base_settings_kwargs):
    settings = make_settings(base_settings_kwargs)
    llm = FakeListChatModel(responses=["no debería llamarse"])
    service = _make_service(settings, llm)

    with pytest.raises(ValueError):
        await service.aquery("   ")


def test_embedding_mismatch_error(base_settings_kwargs):
    kwargs = dict(base_settings_kwargs)
    kwargs["embedding_model"] = "modelo-nuevo"
    settings = make_settings(kwargs)
    _write_valid_manifest(make_settings({**kwargs, "embedding_model": "modelo-viejo"}), n_chunks=3)

    with pytest.raises(EmbeddingMismatchError):
        RAGService(settings=settings, llm=FakeListChatModel(responses=[]), embeddings=object(), vectorstore=_FakeVectorstore(_docs(), count=3))


def test_sin_indice_lanza_indexnotfounderror(base_settings_kwargs):
    settings = make_settings(base_settings_kwargs)
    with pytest.raises(IndexNotFoundError):
        RAGService(settings=settings, llm=FakeListChatModel(responses=[]), embeddings=object(), vectorstore=_FakeVectorstore(_docs(), count=3))


async def test_concurrencia_gather_mas_rapido_que_secuencial(base_settings_kwargs):
    kwargs = dict(base_settings_kwargs)
    kwargs["max_concurrency"] = 3
    settings = make_settings(kwargs)
    delay = 0.2
    content = '{"encontrada": true, "respuesta": "ok", "fragmentos_citados": [1]}'
    llm = _slow_llm(content, delay)
    service = _make_service(settings, llm)

    t0 = time.perf_counter()
    await asyncio.gather(*[service.aquery(f"pregunta {i}") for i in range(3)])
    elapsed = time.perf_counter() - t0

    assert elapsed < 2 * delay


async def test_timeout_lanza_asyncio_timeouterror(base_settings_kwargs):
    kwargs = dict(base_settings_kwargs)
    kwargs["llm_timeout_s"] = 0.05
    settings = make_settings(kwargs)
    llm = _slow_llm('{"encontrada": true, "respuesta": "ok", "fragmentos_citados": [1]}', delay=0.3)
    service = _make_service(settings, llm)

    with pytest.raises(asyncio.TimeoutError):
        await service.aquery("¿algo?")
