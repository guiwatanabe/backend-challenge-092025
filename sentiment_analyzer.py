import hashlib
import math
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Tuple

# ── Lexicons ──────────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "adorei", "gostei", "excelente", "perfeito", "otimo", "bom", "top",
    "incrivel", "maravilhoso", "feliz", "adoravel", "fantastico", "amo",
    "amei", "lindo", "bonito", "brilhante", "otima", "boa", "melhor",
    "poderoso", "agradavel", "eficiente", "rapido", "pratico", "aprovei",
    "recomendo", "confiavel", "surpreendente", "adoro", "sensacional",
}

NEGATIVE_WORDS = {
    "pessimo", "terrivel", "horrivel", "ruim", "mal", "odiei", "detestei",
    "pessima", "horrendo", "horroroso", "decepcionante", "lamentavel",
    "droga", "chato", "fraco", "falhou", "problema", "erro", "pior",
    "cancelei",
}

INTENSIFIERS = {
    "muito", "super", "extremamente", "bastante", "demais", "tao",
}

NEGATIONS = {"nao", "nem", "nunca", "jamais", "sem"}

META_PHRASE = "teste técnico mbras"
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618…

TOKEN_RE = re.compile(r"(?:#\w+(?:-\w+)*)|\b\w+\b", re.UNICODE)


# ── Text utilities ────────────────────────────────────────────────────────────
def normalize_for_matching(token: str) -> str:
    """Lowercase → NFKD → strip combining diacritics."""
    token = token.lower()
    nfkd = unicodedata.normalize("NFKD", token)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


# Pre-normalised lexicon sets (computed once at import time)
_NORM_POSITIVE = {normalize_for_matching(w) for w in POSITIVE_WORDS}
_NORM_NEGATIVE = {normalize_for_matching(w) for w in NEGATIVE_WORDS}
_NORM_INTENSIFIERS = {normalize_for_matching(w) for w in INTENSIFIERS}
_NORM_NEGATIONS = {normalize_for_matching(w) for w in NEGATIONS}


# ── Sentiment analysis ────────────────────────────────────────────────────────
def analyze_sentiment(content: str) -> Tuple[float, str]:
    """
    Returns (score, classification).
    classification is 'positive' | 'negative' | 'neutral' | 'meta'.
    """
    if content.strip().lower() == META_PHRASE:
        return 0.0, "meta"

    tokens = tokenize(content)
    if not tokens:
        return 0.0, "neutral"

    norm = [normalize_for_matching(t) for t in tokens]
    n = len(norm)

    # Step 1 – intensifier positions
    intensifier_pos = {i for i, t in enumerate(norm) if t in _NORM_INTENSIFIERS}

    # Step 2 – negation scopes: negation at i covers positions i+1 … i+3
    neg_count: Dict[int, int] = {}
    for i, t in enumerate(norm):
        if t in _NORM_NEGATIONS:
            for j in range(i + 1, min(i + 4, n)):
                neg_count[j] = neg_count.get(j, 0) + 1

    # Step 3 – score each polarity word
    total_score = 0.0
    for i, t in enumerate(norm):
        if t in _NORM_POSITIVE:
            base = 1.0
        elif t in _NORM_NEGATIVE:
            base = -1.0
        else:
            continue

        # a) intensifier (previous token must be an intensifier)
        if i > 0 and (i - 1) in intensifier_pos:
            base *= 1.5

        # b) negation (odd count → invert; even count → cancel, keep sign)
        nc = neg_count.get(i, 0)
        if nc % 2 == 1:
            base = -base

        # c) MBRAS rule: ×2 for positives only
        if base > 0:
            base *= 2.0

        total_score += base

    score = total_score / n if n > 0 else 0.0

    if score > 0.1:
        return score, "positive"
    if score < -0.1:
        return score, "negative"
    return score, "neutral"


# ── Follower / influence helpers ───────────────────────────────────────────────
def _is_unicode_special(user_id: str) -> bool:
    """True when NFKD normalisation changes the string (non-ASCII user_id)."""
    return unicodedata.normalize("NFKD", user_id) != user_id


_FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _sha256_followers(user_id: str) -> int:
    digest = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return (digest % 10_000) + 100


def calculate_followers(user_id: str) -> int:
    # Unicode trap → 4242
    if _is_unicode_special(user_id):
        return 4242

    # 13-char user_id → 13th Fibonacci (233)
    if len(user_id) == 13:
        return _FIBONACCI[12]

    # Ends with "_prime" → next prime ≥ SHA-256 base
    if user_id.lower().endswith("_prime"):
        base = _sha256_followers(user_id)
        while not _is_prime(base):
            base += 1
        return base

    return _sha256_followers(user_id)


def calculate_influence_score(
    user_id: str,
    reactions: int,
    shares: int,
    views: int,
    is_mbras_employee: bool,
) -> float:
    followers = calculate_followers(user_id)
    engagement_rate = (reactions + shares) / views if views > 0 else 0.0

    # Golden-ratio adjustment when total interactions are a multiple of 7
    if (reactions + shares) > 0 and (reactions + shares) % 7 == 0:
        engagement_rate *= 1 + 1 / PHI

    score = (followers * 0.4) + (engagement_rate * 0.6)

    # Penalty: user_id ends with "007"
    if user_id.endswith("007"):
        score *= 0.5

    # Bonus: MBRAS employee
    if is_mbras_employee:
        score += 2.0

    return round(score, 4)


# ── Trending topics ───────────────────────────────────────────────────────────
def calculate_trending_topics(
    messages: List[Dict],
    sentiments: Dict[str, str],
    reference_ts: datetime,
) -> List[str]:
    weights: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    sentiment_weights: Dict[str, float] = {}

    for msg in messages:
        ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
        minutes_ago = (reference_ts - ts).total_seconds() / 60
        temporal = 1 + 1 / max(minutes_ago, 0.01)

        cls = sentiments.get(msg["id"], "neutral")
        sent_mod = 1.2 if cls == "positive" else (0.8 if cls == "negative" else 1.0)

        for tag in msg.get("hashtags", []):
            tag_text = tag[1:] if tag.startswith("#") else tag
            w = temporal * sent_mod
            if len(tag_text) > 8:
                w *= math.log10(len(tag_text)) / math.log10(8)
            weights[tag] = weights.get(tag, 0.0) + w
            counts[tag] = counts.get(tag, 0) + 1
            sentiment_weights[tag] = sentiment_weights.get(tag, 0.0) + sent_mod

    return sorted(
        weights.keys(),
        key=lambda h: (-weights[h], -counts[h], -sentiment_weights[h], h),
    )[:5]


# ── Anomaly detection ──────────────────────────────────────────────────────────
def detect_anomalies(
    messages: List[Dict],
    sentiments: Dict[str, str],
) -> Dict[str, bool]:
    result = {
        "burst": False,
        "sentiment_alternation": False,
        "synchronized_posting": False,
    }

    # Group by user
    by_user: Dict[str, List[Dict]] = {}
    for msg in messages:
        by_user.setdefault(msg["user_id"], []).append(msg)

    # Burst: >10 messages from same user in any 5-minute window
    for _, msgs in by_user.items():
        if len(msgs) <= 10:
            continue
        tss = sorted(
            datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
            for m in msgs
        )
        for i, start in enumerate(tss):
            window = [t for t in tss[i:] if (t - start).total_seconds() <= 300]
            if len(window) > 10:
                result["burst"] = True
                break

    # Exact sentiment alternation: +−+− … in ≥10 messages per user
    for _, msgs in by_user.items():
        if len(msgs) < 10:
            continue
        sorted_msgs = sorted(msgs, key=lambda m: m["timestamp"])
        polarities = [
            sentiments.get(m["id"], "neutral")
            for m in sorted_msgs
            if sentiments.get(m["id"], "neutral") in ("positive", "negative")
        ]
        if len(polarities) >= 10:
            if all(polarities[i] != polarities[i - 1] for i in range(1, len(polarities))):
                result["sentiment_alternation"] = True

    # Synchronized posting: ≥3 messages within a 4-second span (±2 s)
    if len(messages) >= 3:
        tss = sorted(
            datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
            for m in messages
        )
        for i in range(len(tss) - 2):
            window = [t for t in tss[i:] if (t - tss[i]).total_seconds() <= 4]
            if len(window) >= 3:
                result["synchronized_posting"] = True
                break

    return result


# ── Flags ─────────────────────────────────────────────────────────────────────
def calculate_flags(messages: List[Dict]) -> Dict[str, bool]:
    flags = {
        "mbras_employee": False,
        "candidate_awareness": False,
        "special_pattern": False,
    }
    for msg in messages:
        if "mbras" in msg["user_id"].lower():
            flags["mbras_employee"] = True
        if msg["content"].strip().lower() == META_PHRASE:
            flags["candidate_awareness"] = True
        if len(msg["content"]) == 42 and "mbras" in msg["content"].lower():
            flags["special_pattern"] = True
    return flags
