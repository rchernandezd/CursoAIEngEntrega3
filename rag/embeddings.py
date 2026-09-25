from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

_E5_PREFIXES = ("intfloat/multilingual-e5", "intfloat/e5")


class _E5Embeddings(HuggingFaceEmbeddings):
    """Fallback para versiones de langchain-huggingface sin soporte de
    query_encode_kwargs: agrega el prefijo 'query: ' manualmente antes de embeber."""

    def embed_query(self, text: str):
        return super().embed_query(f"query: {text}")


@lru_cache(maxsize=4)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Única función que construye embeddings. Se usa tanto en ingesta como en consulta."""
    encode_kwargs = {"normalize_embeddings": True}
    query_encode_kwargs = {"normalize_embeddings": True}
    is_e5 = model_name.startswith(_E5_PREFIXES)
    if is_e5:
        encode_kwargs["prompt"] = "passage: "
        query_encode_kwargs["prompt"] = "query: "

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
        )
    except TypeError:
        # Versión instalada sin soporte de query_encode_kwargs.
        encode_kwargs.pop("prompt", None) if not is_e5 else None
        if is_e5:
            embeddings = _E5Embeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
            )
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return embeddings


@lru_cache(maxsize=4)
def get_token_counter(model_name: str):
    """Cuenta tokens con el tokenizer del MISMO modelo de embeddings (sin tokens especiales)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda text: len(tok.encode(text, add_special_tokens=False))
