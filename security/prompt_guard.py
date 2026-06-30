import re
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+context",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"reveal\s+(your\s+)?(developer|internal)\s+prompt",
    r"print\s+(your\s+)?(hidden\s+)?prompt",
    r"show\s+(your\s+)?(internal\s+)?chain(\s+of\s+thought)?",
    r"execute\s+(code|commands|scripts)",
    r"access\s+(files|file\s+system|local\s+files)",
    r"read\s+\.env",
    r"(reveal|show|give|what\s+is)\s+(your\s+)?api\s+key",
    r"override\s+(safety|security|guardrails|restrictions)",
    r"act\s+as\s+(developer|admin|root|system)",
    r"(system|developer)\s+override",
    r"role[-\s]?(play|switch|change)",
    r"DAN\b",
    r"do\s+anything\s+now",
    r"you\s+(are|were)\s+(now\s+)?(free|unleashed|unbounded)",
    r"new\s+persona",
    r"hypothetical.*(previous|prior).*instructions",
    r"(pretend|imagine).*(initial|first|original).*(message|prompt|instruction)",
]


class PromptGuard:
    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS
        ]

    def is_safe(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                logger.warning(
                    "Prompt injection detected — matched: %s | snippet: %s",
                    pattern.pattern[:40],
                    text[:80],
                )
                return False
        return True

    def validate(self, text: str) -> bool:
        result = self.is_safe(text)
        if not result:
            logger.warning("Prompt injection blocked")
        return result
