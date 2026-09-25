import logging

from langchain_core.runnables import RunnableLambda

logger = logging.getLogger("rag")


def build_retriever(vectorstore, top_k: int) -> RunnableLambda:
    def _log_results(query: str, results):
        logger.info(
            "retrieval_done: query=%r top_k=%d results=%s",
            query[:80],
            top_k,
            [(doc.metadata.get("chunk_id"), round(score, 4)) for doc, score in results],
        )
        return results

    def _sync(query: str):
        results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        return _log_results(query, results)

    async def _async(query: str):
        results = await vectorstore.asimilarity_search_with_relevance_scores(query, k=top_k)
        return _log_results(query, results)

    return RunnableLambda(_sync, afunc=_async, name="chroma_retriever")
