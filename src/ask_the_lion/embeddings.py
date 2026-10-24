from langchain_core.embeddings import Embeddings

from ask_the_lion.config import Settings


class E5Embeddings(Embeddings):
    """Adds the "query: " / "passage: " prefixes the E5 model family expects."""

    def __init__(self, base: Embeddings):
        self._base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(f"query: {text}")


def build_embeddings(settings: Settings) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings are not installed; run: pip install -e '.[local-embeddings]'"
        ) from exc

    base = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    if "e5" in settings.embedding_model.lower():
        return E5Embeddings(base)
    return base
