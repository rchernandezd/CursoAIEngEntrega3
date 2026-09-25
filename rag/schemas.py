from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

RESPUESTA_NO_SE = "No lo sé. No tengo acceso a esa información en los documentos disponibles."


class RespuestaLLM(BaseModel):
    """Lo que genera el LLM (parseado con PydanticOutputParser)."""

    encontrada: bool = Field(
        description="true SOLO si el CONTEXTO contiene la información necesaria para responder; si no, false."
    )
    respuesta: str = Field(
        description="Respuesta basada EXCLUSIVAMENTE en el CONTEXTO. Si encontrada=false, exactamente: "
        f"'{RESPUESTA_NO_SE}'"
    )
    fragmentos_citados: List[int] = Field(
        default_factory=list,
        description="Números [n] de los fragmentos del CONTEXTO usados para responder. Lista vacía si encontrada=false.",
    )

    @field_validator("fragmentos_citados")
    @classmethod
    def _dedupe_preserving_order(cls, v: List[int]) -> List[int]:
        seen = set()
        result = []
        for n in v:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result


class Fuente(BaseModel):
    archivo: str  # metadata["source"]
    chunk_id: str  # metadata["chunk_id"]
    titulo: str  # metadata["title"]
    score: float  # relevancia devuelta por Chroma
    extracto: str = Field(max_length=240)  # primeros ~200 caracteres del chunk + "…"


class RAGResponse(BaseModel):
    """Objeto final de get_rag_response: texto + referencias verificables."""

    pregunta: str
    respuesta: str
    encontrada: bool
    fuentes: List[Fuente]
    fragmentos_recuperados: int = Field(ge=0, le=5)
    latencia_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _coherencia(self):
        if not self.encontrada:
            if self.fuentes:
                raise ValueError("Una respuesta 'No lo sé' no puede tener fuentes")
            if not self.respuesta.startswith("No lo sé"):
                raise ValueError("Si encontrada=false la respuesta debe comenzar con 'No lo sé'")
        else:
            if not self.fuentes:
                raise ValueError("Una respuesta encontrada debe citar al menos una fuente")
        return self
