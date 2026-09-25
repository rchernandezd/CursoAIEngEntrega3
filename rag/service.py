import asyncio
import logging
import time
from typing import Optional

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from .chain import build_rag_chain
from .config import Settings
from .embeddings import get_embeddings
from .llm import get_llm
from .retrieval import build_retriever
from .schemas import RESPUESTA_NO_SE, Fuente, RAGResponse
from .vectorstore import assert_index_compatible, get_vectorstore

logger = logging.getLogger("rag")

MAX_QUERY_LEN = 1000


class RAGService:
    def __init__(self, settings: Optional[Settings] = None, *, llm=None, embeddings=None, vectorstore=None):
        self.settings = settings or Settings.from_env()
        self.embeddings = embeddings or get_embeddings(self.settings.embedding_model)
        self.vectorstore = vectorstore or get_vectorstore(self.settings, self.embeddings)
        assert_index_compatible(self.settings, self.vectorstore)
        self.llm = llm or get_llm(self.settings)
        self.chain = build_rag_chain(build_retriever(self.vectorstore, self.settings.top_k), self.llm)
        self._sem = asyncio.Semaphore(self.settings.max_concurrency)

    async def aquery(self, query: str) -> RAGResponse:
        query = query.strip()
        if not query:
            raise ValueError("La pregunta no puede estar vacía")
        if len(query) > MAX_QUERY_LEN:
            raise ValueError(f"La pregunta supera el máximo de {MAX_QUERY_LEN} caracteres")

        logger.info("rag_query_start: query=%r", query[:80])

        async with self._sem:
            t0 = time.perf_counter()
            try:
                out = await asyncio.wait_for(
                    self.chain.ainvoke({"pregunta": query}), timeout=self.settings.llm_timeout_s
                )
            except asyncio.TimeoutError:
                logger.error("rag_query_timeout: query=%r timeout_s=%s", query[:80], self.settings.llm_timeout_s)
                raise
            except (OutputParserException, ValidationError):
                logger.warning("rag_failed_closed: query=%r", query[:80])
                latencia_ms = (time.perf_counter() - t0) * 1000
                response = RAGResponse(
                    pregunta=query,
                    respuesta=RESPUESTA_NO_SE,
                    encontrada=False,
                    fuentes=[],
                    fragmentos_recuperados=0,
                    latencia_ms=latencia_ms,
                )
                logger.info(
                    "rag_query_done: encontrada=%s n_fuentes=%d latencia_ms=%.1f",
                    response.encontrada,
                    len(response.fuentes),
                    response.latencia_ms,
                )
                return response
            except Exception:
                logger.exception("rag_query_error: query=%r", query[:80])
                raise

            latencia_ms = (time.perf_counter() - t0) * 1000

        salida = out["salida"]
        resultados = out["resultados"]
        citas = out["citas"]

        if salida.encontrada and citas:
            fuentes = []
            for i in citas:
                doc, score = resultados[i - 1]
                extracto = doc.page_content[:200].strip()
                if len(doc.page_content) > 200:
                    extracto += "…"
                fuentes.append(
                    Fuente(
                        archivo=doc.metadata.get("source", "?"),
                        chunk_id=doc.metadata.get("chunk_id", "?"),
                        titulo=doc.metadata.get("title", "?"),
                        score=round(score, 4),
                        extracto=extracto,
                    )
                )
            respuesta = RAGResponse(
                pregunta=query,
                respuesta=salida.respuesta,
                encontrada=True,
                fuentes=fuentes,
                fragmentos_recuperados=len(resultados),
                latencia_ms=latencia_ms,
            )
        else:
            respuesta = RAGResponse(
                pregunta=query,
                respuesta=RESPUESTA_NO_SE,
                encontrada=False,
                fuentes=[],
                fragmentos_recuperados=len(resultados),
                latencia_ms=latencia_ms,
            )

        logger.info(
            "rag_query_done: encontrada=%s n_fuentes=%d latencia_ms=%.1f",
            respuesta.encontrada,
            len(respuesta.fuentes),
            respuesta.latencia_ms,
        )
        return respuesta


_service: Optional[RAGService] = None


def get_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service


async def get_rag_response(query: str) -> RAGResponse:
    """a) búsqueda de similitud en ChromaDB  b) prompt con fragmentos
    c) LLM async (ainvoke)  d) parseo a Pydantic con texto + referencias."""
    return await get_service().aquery(query)
