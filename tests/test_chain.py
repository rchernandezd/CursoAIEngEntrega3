import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rag.chain import build_rag_chain


def _fake_docs():
    return [
        (Document(page_content="Contenido 1", metadata={"source": "a.txt", "title": "A", "chunk_id": "a.txt::0"}), 0.9),
        (Document(page_content="Contenido 2", metadata={"source": "b.txt", "title": "B", "chunk_id": "b.txt::0"}), 0.8),
        (Document(page_content="Contenido 3", metadata={"source": "c.txt", "title": "C", "chunk_id": "c.txt::0"}), 0.7),
    ]


def _counting_retriever(call_counter, docs=None):
    docs = docs if docs is not None else _fake_docs()

    def _sync(query: str):
        call_counter["n"] += 1
        return docs

    async def _async(query: str):
        call_counter["n"] += 1
        return docs

    return RunnableLambda(_sync, afunc=_async)


def _json_respuesta(encontrada=True, respuesta="Ok", fragmentos_citados=None):
    citas = fragmentos_citados if fragmentos_citados is not None else [1]
    return (
        '{"encontrada": %s, "respuesta": "%s", "fragmentos_citados": %s}'
        % (str(encontrada).lower(), respuesta, citas)
    )


async def test_citas_validas_se_conservan():
    call_counter = {"n": 0}
    retriever = _counting_retriever(call_counter)
    llm = FakeListChatModel(responses=[_json_respuesta(fragmentos_citados=[2])])
    chain = build_rag_chain(retriever, llm)

    out = await chain.ainvoke({"pregunta": "¿algo?"})
    assert out["citas"] == [2]


async def test_citas_invalidas_se_descartan():
    call_counter = {"n": 0}
    retriever = _counting_retriever(call_counter)
    llm = FakeListChatModel(responses=[_json_respuesta(fragmentos_citados=[2, 7])])
    chain = build_rag_chain(retriever, llm)

    out = await chain.ainvoke({"pregunta": "¿algo?"})
    assert out["citas"] == [2]


async def test_reintento_tras_json_invalido():
    call_counter = {"n": 0}
    retriever = _counting_retriever(call_counter)
    llm = FakeListChatModel(responses=["esto no es json", _json_respuesta(fragmentos_citados=[1])])
    chain = build_rag_chain(retriever, llm)

    out = await chain.ainvoke({"pregunta": "¿algo?"})
    assert out["citas"] == [1]


async def test_encontrada_true_sin_citas_agota_reintentos():
    call_counter = {"n": 0}
    retriever = _counting_retriever(call_counter)
    llm = FakeListChatModel(
        responses=[_json_respuesta(encontrada=True, fragmentos_citados=[])] * 5
    )
    chain = build_rag_chain(retriever, llm)

    with pytest.raises(OutputParserException):
        await chain.ainvoke({"pregunta": "¿algo?"})


async def test_retriever_se_llama_una_sola_vez_pese_a_reintentos():
    call_counter = {"n": 0}
    retriever = _counting_retriever(call_counter)
    llm = FakeListChatModel(responses=["esto no es json", _json_respuesta(fragmentos_citados=[1])])
    chain = build_rag_chain(retriever, llm)

    await chain.ainvoke({"pregunta": "¿algo?"})
    assert call_counter["n"] == 1
