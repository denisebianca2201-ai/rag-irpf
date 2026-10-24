import re
from dataclasses import dataclass

from langchain_core.documents import Document

from ask_the_lion.guardrails.policies import PROMPT_CANARY, Reason

CITATION = re.compile(r"\[Q(\d+)\]")


@dataclass(frozen=True)
class OutputCheck:
    ok: bool
    answer: str
    cited: list[int]
    reason: Reason | None = None


def check_output(answer: str, docs: list[Document]) -> OutputCheck:
    """Require grounded citations, drop invented ones, and block prompt leakage."""
    if PROMPT_CANARY in answer:
        return OutputCheck(False, "", [], Reason.OUTPUT_CHECK)

    allowed = {int(d.metadata["question_number"]) for d in docs}
    cited: list[int] = []

    def keep_valid(match: re.Match) -> str:
        number = int(match.group(1))
        if number in allowed:
            if number not in cited:
                cited.append(number)
            return match.group(0)
        return ""

    cleaned = CITATION.sub(keep_valid, answer).strip()
    if not cited or not cleaned:
        return OutputCheck(False, "", [], Reason.OUTPUT_CHECK)
    return OutputCheck(True, cleaned, cited)
