import logging
from operator import itemgetter

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import ValidationError

from .prompts import build_prompt
from .schemas import RespuestaLLM

logger = logging.getLogger("rag")


def formatear_contexto(resultados) -> str:
    if not resultados:
        return "(no se recuperaron fragmentos)"

    partes = []
    for i, (doc, _score) in enumerate(resultados, start=1):
        fuente = doc.metadata.get("source", "?")
        titulo = doc.metadata.get("title", "?")
        partes.append(f"[{i}] (fuente: {fuente} — {titulo})\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


def build_rag_chain(retriever, llm):
    parser = PydanticOutputParser(pydantic_object=RespuestaLLM)
    prompt = build_prompt(parser)

    generar = prompt | llm | parser  # output SIEMPRE pasa por PydanticOutputParser

    def _validar_citas(x: dict) -> dict:
        salida: RespuestaLLM = x["salida"]
        n = len(x["resultados"])
        validas = [i for i in salida.fragmentos_citados if 1 <= i <= n]
        invalidas = [i for i in salida.fragmentos_citados if i not in validas]
        if invalidas:
            logger.warning("citas_invalidas_descartadas: %s (recuperados=%d)", invalidas, n)
        if salida.encontrada and not validas:
            # Respuesta "encontrada" sin respaldo verificable -> tratar como salida mal formada y reintentar
            raise OutputParserException("encontrada=true sin fragmentos_citados válidos")
        return {**x, "citas": validas}

    generacion = (
        RunnablePassthrough.assign(salida=generar) | RunnableLambda(_validar_citas)
    ).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    return (
        RunnablePassthrough.assign(resultados=itemgetter("pregunta") | retriever)  # 1. retrieve (async)
        | RunnablePassthrough.assign(contexto=lambda x: formatear_contexto(x["resultados"]))  # 2. docs -> texto
        | generacion  # 3. prompt -> LLM -> parser -> validación
    )
