import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from ask_the_lion.config import Settings
from ask_the_lion.ingest import build_vector_store, index_documents, split_into_questions

PAGES = [
    (
        1,
        "CAPÍTULO 1 - OBRIGATORIEDADE\n"
        "001 — Quem está obrigado a apresentar a declaração do imposto de renda?\n"
        "Está obrigado a declarar quem recebeu rendimentos tributáveis acima do limite anual.\n"
        "002 — Qual o prazo para entregar a declaração do imposto de renda?\n"
        "O prazo de entrega da declaração começa em março e termina em maio.",
    ),
    (
        2,
        "CAPÍTULO 2 - DEPENDENTES\n"
        "003 — Posso incluir dependentes na declaração do imposto de renda?\n"
        "Sim, filhos até 21 anos podem ser incluídos como dependentes na declaração.\n"
        "Há dedução anual por dependente.",
    ),
]


STOPWORDS = {"de", "da", "do", "a", "o", "e", "em", "na", "no", "para", "que", "um", "uma", "os", "as", "por", "com", "se", "ao"}


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: shared words => high cosine similarity."""

    dims = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in re.findall(r"\w+", text.lower()):
            if token in STOPWORDS:
                continue
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        chroma_dir=tmp_path / "chroma",
        pdf_path=tmp_path / "qa.pdf",
        min_relevance=0.3,
        top_k=3,
        rate_limit="100/minute",
    )


@pytest.fixture
def docs():
    return split_into_questions(PAGES)


@pytest.fixture
def store(settings, docs):
    embeddings = HashEmbeddings()
    index_documents(docs, embeddings, settings)
    return build_vector_store(embeddings, settings)


class RecordingLLM:
    """Runnable fake LLM that records the prompts it receives."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt_value) -> str:
        self.prompts.append(prompt_value.to_string())
        return self.reply

    def runnable(self):
        return RunnableLambda(self)


@pytest.fixture
def make_llm():
    return RecordingLLM
