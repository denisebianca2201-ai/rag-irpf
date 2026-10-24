from enum import StrEnum


class Reason(StrEnum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INJECTION = "injection"
    NO_CONTEXT = "no_context"
    OUTPUT_CHECK = "output_check"


REFUSALS: dict[Reason, str] = {
    Reason.EMPTY: "Please type a question about the Brazilian income tax (IRPF).",
    Reason.TOO_LONG: "Your question is too long. Please shorten it and try again.",
    Reason.INJECTION: "I can only answer questions about the income tax (IRPF) and cannot follow that request.",
    Reason.NO_CONTEXT: (
        "I could not find an answer to that in the official Receita Federal "
        "Q&A document. Try rephrasing, or check the Receita Federal website."
    ),
    Reason.OUTPUT_CHECK: (
        "I could not produce a reliable answer grounded in the official document. "
        "Please rephrase or consult the Receita Federal."
    ),
}

DISCLAIMER = (
    "\n\n_Informational only, based on the official Receita Federal Q&A. "
    "Confirm on the Receita Federal website or with a tax professional._"
)

# Canary placed in the system prompt; it must never appear in an answer.
PROMPT_CANARY = "LION-PROMPT-CANARY-7f3a"
# Token the model returns when the context does not answer the question.
NO_ANSWER_TOKEN = "NO_ANSWER"
