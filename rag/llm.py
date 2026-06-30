from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, RateLimitError, APITimeoutError
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, TEMPERATURE, TOP_P
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Please add it to your .env file."
            )
        self.llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/anomalyco/Multi-Document-Comparator",
                "X-Title": "Multi-Document Comparator",
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            "LLM call failed (attempt %d), retrying...",
            retry_state.attempt_number,
        ),
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        logger.info("Generating LLM response")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm.invoke(messages)
        logger.info("LLM response received (tokens: I=%d O=%d)",
                     response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                     response.response_metadata.get("token_usage", {}).get("completion_tokens", 0))
        return response.content
