import re
from dataclasses import dataclass

from ask_the_lion.guardrails.policies import Reason

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"\bact\s+as\s+(a|an|if)\b",
    r"\bjailbreak\b|\bDAN\s+mode\b|developer\s+mode",
    r"ignor[ae]\s+(todas?\s+)?(as\s+)?(instru[cç][õo]es|regras)\s+(anteriores|acima)",
    r"desconsidere\s+(as\s+)?(instru[cç][õo]es|regras)",
    r"(revele|mostre|repita|exiba)\s+(o\s+)?(seu\s+|o\s+)?(prompt|instru[cç][õo]es)\s*(do\s+sistema)?",
    r"(finja|aja\s+como|voc[eê]\s+agora\s+[eé])\b",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)

# Order matters: CNPJ and CPF must be matched before the looser phone pattern.
_PII_PATTERNS = [
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}(?!\d)")),
    ("ACCOUNT", re.compile(r"(?i)\b(?:ag(?:[eê]ncia)?|conta|c/c)\.?\s*:?\s*\d[\d.-]{3,}\d\b")),
]


@dataclass(frozen=True)
class InputCheck:
    ok: bool
    text: str
    reason: Reason | None = None


def redact_pii(text: str) -> str:
    for label, pattern in _PII_PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text


def check_input(raw: str, max_chars: int) -> InputCheck:
    """Validate and sanitize a user question; the returned text is PII-free."""
    if len(raw) > max_chars:
        return InputCheck(False, "", Reason.TOO_LONG)

    text = _CONTROL_CHARS.sub("", raw)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return InputCheck(False, "", Reason.EMPTY)

    if INJECTION_RE.search(text):
        return InputCheck(False, "", Reason.INJECTION)

    return InputCheck(True, redact_pii(text))
