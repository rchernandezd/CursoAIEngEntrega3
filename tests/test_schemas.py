import pytest
from pydantic import ValidationError

from rag.config import Settings
from rag.schemas import RAGResponse, RespuestaLLM, Fuente, RESPUESTA_NO_SE
from .conftest import make_settings


def _fuente(**overrides):
    base = dict(
        archivo="data/politica_vacaciones.txt",
        chunk_id="data/politica_vacaciones.txt::0",
        titulo="Política de Vacaciones - TechCorp",
        score=0.8,
        extracto="Los empleados con 5 años...",
    )
    base.update(overrides)
    return Fuente(**base)


def _response(**overrides):
    base = dict(
        pregunta="¿Cuántos días de vacaciones?",
        respuesta="21 días.",
        encontrada=True,
        fuentes=[_fuente()],
        fragmentos_recuperados=4,
        latencia_ms=120.0,
    )
    base.update(overrides)
    return RAGResponse(**base)


def test_encontrada_false_con_fuentes_falla():
    with pytest.raises(ValidationError):
        _response(encontrada=False, fuentes=[_fuente()], respuesta=RESPUESTA_NO_SE)


def test_encontrada_false_sin_texto_no_se_falla():
    with pytest.raises(ValidationError):
        _response(encontrada=False, fuentes=[], respuesta="No tengo idea.")


def test_encontrada_true_sin_fuentes_falla():
    with pytest.raises(ValidationError):
        _response(encontrada=True, fuentes=[])


def test_fragmentos_recuperados_fuera_de_rango_falla():
    with pytest.raises(ValidationError):
        _response(fragmentos_recuperados=6)


def test_respuesta_no_se_valida_ok():
    r = _response(encontrada=False, fuentes=[], respuesta=RESPUESTA_NO_SE)
    assert r.encontrada is False
    assert r.fuentes == []


def test_fragmentos_citados_dedupe_preservando_orden():
    r = RespuestaLLM(encontrada=True, respuesta="algo", fragmentos_citados=[1, 1, 2])
    assert r.fragmentos_citados == [1, 2]


@pytest.mark.parametrize(
    "override,field",
    [
        ({"chunk_size": 400}, "chunk_size"),
        ({"chunk_overlap": 40}, "chunk_overlap"),
        ({"top_k": 2}, "top_k"),
        ({"top_k": 6}, "top_k"),
    ],
)
def test_settings_rechaza_valores_invalidos(base_settings_kwargs, override, field):
    with pytest.raises(ValueError):
        make_settings(base_settings_kwargs, **override)


def test_settings_acepta_valores_validos(base_settings_kwargs):
    s = Settings(**base_settings_kwargs)
    assert s.chunk_size == 500
    assert 3 <= s.top_k <= 5
