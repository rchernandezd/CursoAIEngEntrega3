# RAG Local — TechCorp

Sistema de recuperación semántica local (RAG) sobre políticas internas de una
empresa ficticia ("TechCorp"). Ingesta documentos `.txt`/`.md` desde `data/`,
los fragmenta por tokens, los persiste en **ChromaDB local** (`./vectorstore`),
recupera los fragmentos más relevantes para una pregunta y genera una respuesta
**grounded** (basada exclusivamente en esos fragmentos) mediante una cadena
**LCEL asíncrona** con Google Gemini, validada con `PydanticOutputParser`. Si la
respuesta no está en los documentos, el sistema responde **"No lo sé"** en vez
de inventar información.

## 1. Flujo del sistema

**Ingesta (offline, `ingest.py`):**

```
data/*.txt, *.md
   │  load_documents()
   ▼
split (RecursiveCharacterTextSplitter, 500 tokens / 70 overlap,
       medidos con el tokenizer de intfloat/multilingual-e5-small)
   │
   ▼
embeddings E5 multilingües (prefijo "passage: ")
   │
   ▼
ChromaDB persistente (./vectorstore) + manifiesto de ingesta
```

**Consulta (online, `get_rag_response`):**

```
pregunta
   │  embedding con prefijo "query: " (mismo modelo E5)
   ▼
top-k (3–5) en ChromaDB, con score de relevancia
   │
   ▼
formateo de fragmentos numerados [1] [2] ...
   │
   ▼
prompt "filtro de veracidad" → Gemini (ainvoke, async)
   │
   ▼
PydanticOutputParser → RespuestaLLM (encontrada, respuesta, fragmentos_citados)
   │
   ▼
validación de citas contra fragmentos realmente recuperados
   │
   ▼
RAGResponse (respuesta + fuentes verificables desde metadata real)
```

## 2. Decisiones de diseño

- **Tokenizer del embedder para medir chunks**: `chunk_size`/`chunk_overlap` se miden
  con el tokenizer real de `intfloat/multilingual-e5-small` (no caracteres, no
  otro tokenizer), así se garantiza que cada fragmento entra completo en la
  ventana de contexto del embedder (512 tokens) y no se trunca en silencio.
- **E5 multilingüe en vez de `all-MiniLM-L6-v2`**: el modelo del notebook original
  está entrenado en inglés y trunca a 256 tokens. Con chunks de 500 tokens en
  español, perdería la mitad de cada fragmento al embeber. `multilingual-e5-small`
  soporta 512 tokens y está entrenado para múltiples idiomas, incluido español.
- **Manifiesto anti-reindexado**: guarda modelo de embeddings, parámetros de
  chunking, nombre de colección y hash sha256 de cada archivo fuente. Una
  segunda corrida de `ingest.py` sin cambios detecta que el índice sigue vigente
  y no vuelve a calcular embeddings. Cambiar el modelo de embeddings, los
  parámetros de chunking o cualquier archivo de `data/` fuerza un reindexado.
- **Referencias construidas desde metadata real, no generadas por el LLM**: el LLM
  solo indica qué números de fragmento usó (`fragmentos_citados`); el código
  arma las `fuentes` a partir de la metadata real de esos fragmentos recuperados.
  El LLM nunca "inventa" un archivo o un score.
- **Citas validadas**: si el LLM cita un número de fragmento que no existe entre
  los recuperados, se descarta con un warning. Si dice `encontrada=true` pero no
  cita ningún fragmento válido, se trata como salida mal formada y se reintenta.
- **Fail-closed**: si tras los reintentos la salida del LLM sigue sin poder
  validarse, el sistema responde "No lo sé" en vez de propagar una excepción o
  arriesgar una respuesta no verificable.
- **Retry acotado**: `.with_retry()` solo envuelve la generación (prompt → LLM →
  parser → validación de citas), nunca la recuperación en ChromaDB. Un reintento
  por JSON mal formado no vuelve a consultar el vectorstore. No se reintenta ante
  errores de autenticación, cuota agotada o timeout: esos se propagan.
- **Semáforo de concurrencia**: `MAX_CONCURRENCY` limita las llamadas simultáneas
  al LLM, dado que el free tier de Gemini tiene límites de requests por minuto.
- **`gemini-flash-lite-latest` en vez de `gemini-flash-latest`**: al implementar
  y correr `main.py` de punta a punta, `gemini-flash-latest` resolvió a un modelo
  preview (`gemini-3.8-flash`) con un free tier de apenas 5 req/min **y 20 req/día**,
  insuficiente para una corrida completa de las pruebas. `gemini-flash-lite-latest`
  tiene una cuota separada y notablemente más generosa, y en la práctica respondió
  más rápido. Sigue siendo configurable vía `GEMINI_MODEL` si se dispone de una
  cuenta con mayor cuota.

## 3. Requisitos

- Python 3.12
- API key gratuita de Google Gemini: https://aistudio.google.com/apikey

## 4. Setup

**PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env   # y completar GOOGLE_API_KEY
```

**bash:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && cp .env.example .env
```

> `sentence-transformers` instala `torch`. En Windows sin GPU, instalar primero
> el wheel CPU (como arriba) evita una descarga mucho más pesada con soporte CUDA.

## 5. Ejecución

```bash
python ingest.py              # indexa (primera vez) o "skipped" si ya está vigente
python ingest.py --force      # fuerza reindexado completo
python main.py                # pruebas obligatorias + demo de concurrencia
python main.py --interactive  # + modo interactivo de preguntas
```

## 6. Tests

```bash
pytest -q                 # suite offline, sin red ni API key
pytest -m integration -q  # requiere GOOGLE_API_KEY real y red
```

Resultado real (2026-09-24):

```
$ pytest -q
33 passed, 7 deselected in 11.91s

$ pytest -m integration -q
7 passed, 33 deselected in 72.40s
```

## 7. Evidencia de ejecución

Corrida real de `python main.py` (2026-09-24, `GEMINI_MODEL=gemini-flash-lite-latest`).

**Prueba 1 — respuesta en los documentos:**

```json
{
  "pregunta": "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?",
  "respuesta": "Los empleados con 5 años o más, y hasta 10 años, acceden a 21 días corridos de vacaciones por año calendario.",
  "encontrada": true,
  "fuentes": [
    {
      "archivo": "data/politica_vacaciones.txt",
      "chunk_id": "data/politica_vacaciones.txt::0",
      "titulo": "Política de Vacaciones - TechCorp",
      "score": 0.9027,
      "extracto": "Política de Vacaciones - TechCorp\n\nTodos los empleados en relación de dependencia directa de TechCorp tienen derecho a\ndías de descanso anual pago, calculados según su antigüedad en la empresa.\n\nLos e…"
    }
  ],
  "fragmentos_recuperados": 4,
  "latencia_ms": 1648.37
}
```

**Prueba 2 — pregunta trampa (sin respuesta en los documentos):**

```json
{
  "pregunta": "¿Cuál es la política de bonos por rendimiento anual en TechCorp?",
  "respuesta": "No lo sé. No tengo acceso a esa información en los documentos disponibles.",
  "encontrada": false,
  "fuentes": [],
  "fragmentos_recuperados": 4,
  "latencia_ms": 1484.37
}
```

**Tabla PASS/FAIL final:**

```
[PASS] PRUEBA 1 — Respuesta en los documentos (vacaciones)
       encontrada=True (esperado True) | contiene '21': True | fuente 'data/politica_vacaciones.txt' presente: True
[PASS] PRUEBA 2 — Pregunta trampa (sin respuesta en los documentos)
       encontrada=False (esperado False) | sin fuentes y 'No lo sé': True
[PASS] PRUEBA 3 — Trampa de conocimiento general
       encontrada=False (esperado False) | sin fuentes y 'No lo sé': True
[PASS] PRUEBA 4 — Trampa de prompt injection
       encontrada=False (esperado False) | sin fuentes y 'No lo sé': True
[PASS] PRUEBA 5 — Respuesta que solo está en el .md (licencia parental)
       encontrada=True (esperado True) | contiene '10': True | fuente 'data/manual_beneficios_y_licencias.md' presente: True

TODO PASS ✅
```

**Demo de concurrencia — secuencial vs `asyncio.gather`:**

```
Secuencial : 3.01s
gather     : 7.71s
Speedup    : 0.39x
```

A nivel de **cliente**, la concurrencia funciona como se espera: los logs muestran
que las 3 consultas del `gather` inician en el mismo milisegundo y que las 3
recuperaciones en ChromaDB (`retrieval_done`) terminan casi simultáneamente
(dentro de 5 ms entre sí), lo que confirma que `asyncio.gather` efectivamente
dispara las 3 corrutinas en paralelo y no las serializa. La wall-clock final,
sin embargo, salió más lenta que la secuencial: al recibir 3 conexiones
simultáneas, el free tier de Gemini respondió más lento por request (4.3s–7.7s
por llamada) que en la corrida secuencial (0.6s–1.5s por llamada), consistente
con throttling del lado del proveedor ante ráfagas concurrentes en una cuenta
gratuita. La prueba controlada de que el *código* del servicio sí logra
speedup real con un LLM sin ese throttling está en
`tests/test_service_async.py::test_concurrencia_gather_mas_rapido_que_secuencial`
(LLM fake con latencia artificial, sin llamadas de red), que corre en la suite
offline y pasa de forma consistente.

## 8. Uso programático

```python
import asyncio
from rag import get_rag_response

resp = asyncio.run(get_rag_response("¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?"))
print(resp.respuesta)
print(resp.fuentes)
```

## 9. Seguridad

- `GOOGLE_API_KEY` solo vive en `.env`, que está excluido en `.gitignore` y nunca
  se commitea. `.env.example` es la plantilla sin valores reales.
- La primera ejecución de `ingest.py` descarga el modelo de embeddings
  (`intfloat/multilingual-e5-small`, ~470 MB) a la caché local de HuggingFace.
- `vectorstore/` (el índice de ChromaDB) tampoco se commitea: se regenera con
  `python ingest.py`.

## 10. Limitaciones conocidas y posibles mejoras

- No hay umbral mínimo de score de relevancia (`MIN_RELEVANCE_SCORE`): siempre se
  consultan los top-k fragmentos, aunque su score sea bajo. Podría activarse un
  corte para evitar contexto poco relevante y ahorrar costo de LLM.
- Sin re-ranking con cross-encoder ni búsqueda híbrida (BM25 + vectorial).
- Sin memoria conversacional: cada pregunta se resuelve de forma independiente.
