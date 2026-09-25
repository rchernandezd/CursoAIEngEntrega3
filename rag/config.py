import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    persist_dir: Path
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_model: str
    llm_temperature: float
    llm_timeout_s: float
    max_concurrency: int

    def __post_init__(self):
        if self.chunk_size < 500:
            raise ValueError("CHUNK_SIZE debe ser >= 500 tokens (consigna)")
        if self.chunk_overlap < 50:
            raise ValueError("CHUNK_OVERLAP debe ser >= 50 tokens (consigna)")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        if not 3 <= self.top_k <= 5:
            raise ValueError("TOP_K debe estar entre 3 y 5 (evitar 'contexto infinito')")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_dotenv()
        values = dict(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            persist_dir=Path(os.getenv("PERSIST_DIR", "vectorstore")),
            collection_name=os.getenv("COLLECTION_NAME", "techcorp_policies"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "70")),
            top_k=int(os.getenv("TOP_K", "4")),
            llm_model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "3")),
        )
        values.update(overrides)
        return cls(**values)
