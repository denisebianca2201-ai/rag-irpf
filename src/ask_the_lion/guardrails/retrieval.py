from langchain_core.documents import Document

ScoredDoc = tuple[Document, float]


def filter_relevant(results: list[ScoredDoc], min_relevance: float) -> list[ScoredDoc]:
    """Keep only chunks whose relevance reaches the threshold, best first.

    An empty result means the question is outside the document's scope, and the
    caller must not call the LLM.
    """
    kept = [(doc, score) for doc, score in results if score >= min_relevance]
    return sorted(kept, key=lambda pair: pair[1], reverse=True)
