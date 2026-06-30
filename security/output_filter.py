import re
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

_SENSITIVE_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",
    r"openai-api-key",
    r"api[-_]?key[-_]?\s*[:=]\s*\S+",
    r"OPENROUTER_API_KEY",
    r"OLLAMA_BASE_URL",
    r"CHROMA_DB_DIR",
    r"\b(eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})\b",
]


class OutputFilter:
    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in _SENSITIVE_PATTERNS
        ]

    def filter(self, text: str) -> str:
        if not text:
            return text
        for pattern in self.compiled_patterns:
            text = pattern.sub("[REDACTED]", text)
        return text
