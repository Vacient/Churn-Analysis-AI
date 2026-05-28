import json
import os
import re
from dataclasses import dataclass

from groq import Groq


_MODEL_NAME = "llama-3.1-8b-instant"
_SYSTEM_PROMPT = (
    "You are a sentiment classifier for customer support tickets. "
    'Return ONLY valid JSON in this exact shape: {"sentiment":"positive|neutral|negative"}. '
    "No extra keys, no markdown, no commentary."
)
_USER_PROMPT_TEMPLATE = (
    "Classify sentiment for this ticket description:\n"
    "{description}\n\n"
    'Output only JSON: {{"sentiment":"positive|neutral|negative"}}'
)
_SENTIMENT_TO_SCORE = {"positive": 1, "neutral": 0, "negative": -1}
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_SENTIMENT_WORD_RE = re.compile(r"\b(positive|neutral|negative)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SentimentResult:
    sentiment: str
    score: int


def _neutral_result() -> SentimentResult:
    return SentimentResult(sentiment="neutral", score=0)


def _parse_sentiment_label(raw_content: str) -> str:
    content = raw_content.strip()
    if not content:
        return "neutral"

    match = _JSON_BLOCK_RE.search(content)
    json_payload = match.group(0) if match else content

    try:
        parsed = json.loads(json_payload)
        sentiment = str(parsed.get("sentiment", "")).strip().lower()
        if sentiment in _SENTIMENT_TO_SCORE:
            return sentiment
    except json.JSONDecodeError:
        pass

    # Fallback parser for non-JSON or malformed JSON responses.
    word_match = _SENTIMENT_WORD_RE.search(content.lower())
    if word_match:
        return word_match.group(1).lower()

    return "neutral"


def get_sentiment_result(description: str) -> SentimentResult:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _neutral_result()

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=_MODEL_NAME,
            temperature=0,
            max_tokens=30,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(description=description),
                },
            ],
            response_format={"type": "json_object"},
        )
        raw_content = completion.choices[0].message.content or ""
        sentiment = _parse_sentiment_label(raw_content)
        return SentimentResult(sentiment=sentiment, score=_SENTIMENT_TO_SCORE[sentiment])
    except Exception:
        return _neutral_result()
