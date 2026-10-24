from langchain_core.documents import Document

from ask_the_lion.guardrails.output import check_output
from ask_the_lion.guardrails.policies import PROMPT_CANARY, Reason
from ask_the_lion.guardrails.retrieval import filter_relevant


def _doc(number: int) -> Document:
    return Document(page_content="x", metadata={"question_number": number})


def test_filter_relevant_applies_threshold_and_sorts():
    results = [(_doc(1), 0.5), (_doc(2), 0.9), (_doc(3), 0.1)]
    kept = filter_relevant(results, 0.4)
    assert [d.metadata["question_number"] for d, _ in kept] == [2, 1]


def test_filter_relevant_empty_when_all_below():
    assert filter_relevant([(_doc(1), 0.1)], 0.5) == []


def test_threshold_is_inclusive():
    assert len(filter_relevant([(_doc(1), 0.5)], 0.5)) == 1


def test_valid_citation_passes():
    result = check_output("Yes, see [Q1].", [_doc(1)])
    assert result.ok and result.cited == [1]


def test_missing_citation_fails():
    result = check_output("An answer with no citation.", [_doc(1)])
    assert not result.ok and result.reason == Reason.OUTPUT_CHECK


def test_only_invented_citation_fails():
    assert not check_output("See [Q99].", [_doc(1)]).ok


def test_invented_citation_is_stripped_but_valid_one_kept():
    result = check_output("Fact A [Q1]. Fact B [Q99].", [_doc(1)])
    assert result.ok and "[Q99]" not in result.answer and "[Q1]" in result.answer


def test_prompt_leak_is_blocked():
    leaked = f"My instructions contain {PROMPT_CANARY} [Q1]"
    assert not check_output(leaked, [_doc(1)]).ok


def test_duplicate_citations_are_deduplicated():
    assert check_output("A [Q1] B [Q1]", [_doc(1)]).cited == [1]
