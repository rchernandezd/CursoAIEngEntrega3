import pytest

from rag.config import Settings


@pytest.fixture
def base_settings_kwargs(tmp_path):
    return dict(
        data_dir=tmp_path / "data",
        persist_dir=tmp_path / "vectorstore",
        collection_name="test_collection",
        embedding_model="fake-embedding-model",
        chunk_size=500,
        chunk_overlap=70,
        top_k=4,
        llm_model="fake-llm",
        llm_temperature=0.0,
        llm_timeout_s=5.0,
        max_concurrency=3,
    )


@pytest.fixture
def settings(base_settings_kwargs):
    return Settings(**base_settings_kwargs)


def make_settings(base_kwargs, **overrides):
    kwargs = dict(base_kwargs)
    kwargs.update(overrides)
    return Settings(**kwargs)
