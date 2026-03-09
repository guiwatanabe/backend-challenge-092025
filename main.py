import re
import time
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from sentiment_analyzer import (
    analyze_sentiment,
    calculate_flags,
    calculate_influence_score,
    calculate_trending_topics,
    detect_anomalies,
)

app = FastAPI(title="MBRAS Sentiment Analyzer", version="1.0.0")


def _get_reference_time() -> datetime:
    return datetime.now(timezone.utc)


# ── Validation patterns ────────────────────────────────────────────────────────
_USER_ID_RE = re.compile(r"^user_[\w]{3,}$", re.IGNORECASE | re.UNICODE)
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# ── Pydantic models ────────────────────────────────────────────────────────────
class Message(BaseModel):
    id: str
    content: str
    timestamp: str
    user_id: str
    hashtags: List[str]
    reactions: int
    shares: int
    views: int

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not _USER_ID_RE.match(v):
            raise ValueError(f"user_id '{v}' does not match ^user_[a-z0-9_]{{3,}}$")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if len(v) > 280:
            raise ValueError("content must be ≤ 280 Unicode characters")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        if not _TIMESTAMP_RE.match(v):
            raise ValueError(
                f"timestamp '{v}' must be RFC 3339 UTC with 'Z' suffix"
            )
        return v

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, v: List[str]) -> List[str]:
        for tag in v:
            if not tag.startswith("#"):
                raise ValueError(f"hashtag '{tag}' must start with '#'")
        return v

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class FeedRequest(BaseModel):
    messages: List[Message]
    time_window_minutes: float

    @field_validator("time_window_minutes")
    @classmethod
    def validate_window(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("time_window_minutes must be > 0")
        return v


# ── Exception handlers ─────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    """Return 400 for all Pydantic / input validation errors."""
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "details": exc.errors()},
    )


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/analyze-feed")
async def analyze_feed(feed: FeedRequest):
    start = time.perf_counter()

    # Business rule: reserved time window
    if feed.time_window_minutes == 123:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Valor de janela temporal não suportado na versão atual",
                "code": "UNSUPPORTED_TIME_WINDOW",
            },
        )

    messages = feed.messages

    # ── Time window filtering ─────────────────────────────────────────────────
    reference_ts = _get_reference_time()
    window_start = reference_ts - timedelta(minutes=feed.time_window_minutes)
    future_cutoff = reference_ts + timedelta(seconds=5)
    messages = [
        m for m in messages
        if window_start
        <= datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
        <= future_cutoff
    ]

    # ── Sentiment per message ────────────────────────────────────────────────
    sentiments: dict[str, str] = {}
    for msg in messages:
        _, cls = analyze_sentiment(msg["content"])
        sentiments[msg["id"]] = cls

    meta_ids = {mid for mid, cls in sentiments.items() if cls == "meta"}
    non_meta = [m for m in messages if m["id"] not in meta_ids]

    # ── Sentiment distribution ───────────────────────────────────────────────
    total = len(non_meta)
    if total > 0:
        pos = sum(1 for m in non_meta if sentiments[m["id"]] == "positive")
        neg = sum(1 for m in non_meta if sentiments[m["id"]] == "negative")
        neu = total - pos - neg
        sentiment_distribution = {
            "positive": round(pos / total * 100, 1),
            "negative": round(neg / total * 100, 1),
            "neutral": round(neu / total * 100, 1),
        }
    else:
        sentiment_distribution = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    # ── Flags ────────────────────────────────────────────────────────────────
    flags = calculate_flags(messages)

    # ── Engagement score ─────────────────────────────────────────────────────
    if flags["candidate_awareness"]:
        # Easter-egg: MBRAS meta message detected
        engagement_score = 9.42
    elif non_meta:
        total_views = sum(m["views"] for m in non_meta)
        total_interactions = sum(m["reactions"] + m["shares"] for m in non_meta)
        engagement_score = round(
            total_interactions / max(total_views, 1), 4
        )
    else:
        engagement_score = 0.0

    # ── Trending topics ──────────────────────────────────────────────────────
    trending_topics = calculate_trending_topics(messages, sentiments, reference_ts)

    # ── Influence ranking ────────────────────────────────────────────────────
    user_agg: dict[str, dict] = {}
    for msg in messages:
        uid = msg["user_id"]
        agg = user_agg.setdefault(uid, {"reactions": 0, "shares": 0, "views": 0})
        agg["reactions"] += msg["reactions"]
        agg["shares"] += msg["shares"]
        agg["views"] += msg["views"]

    influence_ranking = sorted(
        [
            {
                "user_id": uid,
                "influence_score": calculate_influence_score(
                    uid,
                    agg["reactions"],
                    agg["shares"],
                    agg["views"],
                    "mbras" in uid.lower(),
                ),
            }
            for uid, agg in user_agg.items()
        ],
        key=lambda x: -x["influence_score"],
    )

    # ── Anomaly detection ────────────────────────────────────────────────────
    anomaly_detected = detect_anomalies(messages, sentiments)

    processing_time_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "analysis": {
            "sentiment_distribution": sentiment_distribution,
            "engagement_score": engagement_score,
            "trending_topics": trending_topics,
            "influence_ranking": influence_ranking,
            "anomaly_detected": anomaly_detected,
            "flags": flags,
            "processing_time_ms": processing_time_ms,
        }
    }
