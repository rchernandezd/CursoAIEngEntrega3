import os

from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(settings):
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Falta GOOGLE_API_KEY en el entorno. Completar .env a partir de .env.example."
        )
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_retries=2,
    )
