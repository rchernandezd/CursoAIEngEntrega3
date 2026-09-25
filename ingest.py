import sys

from rag.config import Settings
from rag.ingestion import ingest
from rag.logging_config import setup_logging


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    setup_logging()
    force = "--force" in sys.argv[1:]

    settings = Settings.from_env()
    result = ingest(settings, force=force)

    if result.status == "skipped":
        print(f"♻️  Índice vigente — sin reindexar ({result.n_chunks} chunks)")
    else:
        print(f"✅ Índice reconstruido ({result.reason})")
        print(f"   Documentos: {result.n_documents}  |  Chunks: {result.n_chunks}")
        if result.chunks_by_file:
            print(f"\n   {'archivo':<45} {'n_chunks':>10} {'tokens_max':>12}")
            for source, info in sorted(result.chunks_by_file.items()):
                print(f"   {source:<45} {info['n_chunks']:>10} {info['max_tokens']:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
