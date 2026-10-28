"""Score and select the agent cards most relevant to a turn (pure; no I/O)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence, Union

from .cards import STOPWORDS, AgentCard, as_utc, is_common, normalize_name

_TOKEN_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|[a-z0-9]+")
_PINNED_RE = re.compile(r"^\s*\[(?:SUCCESS|FAILED)\]\s*([^:\n]+?)\s*:", re.MULTILINE)

BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class RetrievalConfig:
    w_kw: float = 0.45
    w_ent: float = 0.40
    w_rec: float = 0.15
    recency_days: float = 7.0
    last_routed_boost: float = 0.15
    k: int = 8
    hot_max: int = 5
    hot_window_hours: float = 1.0
    min_score: float = 0.1
    fuzzy_min_len: int = 4
    fuzzy_weight: float = 0.5


DEFAULT_CONFIG = RetrievalConfig()


def _stem(token: str) -> str:
    if "@" in token or len(token) < 4 or token.endswith("ss"):
        return token
    return token[:-1] if token.endswith("s") else token


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (emails kept whole), possessives and plurals folded, stopwords dropped."""
    lowered = (text or "").lower().replace("’", "'")
    lowered = re.sub(r"'s\b", "", lowered).replace("'", "")
    return [_stem(tok) for tok in _TOKEN_RE.findall(lowered) if tok not in STOPWORDS]


def content_tokens(text: str) -> list[str]:
    """Tokens that can identify a person or topic: `tokenize` minus common instruction words."""
    return [tok for tok in tokenize(text) if not is_common(tok)]


def edit_distance_le1(a: str, b: str) -> bool:
    """True when `a` and `b` differ by at most one insert, delete, substitute or adjacent transposition."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and a[i] == b[j] and a[j] == b[i]
        return False
    if la > lb:
        a, b = b, a
    # b is one longer: skipping one char of b must yield a.
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


def _doc_tokens(card: AgentCard) -> list[str]:
    return tokenize(f"{card.name} {card.purpose} {card.last_status}")


def _entity_terms(card: AgentCard) -> set[str]:
    terms = set(content_tokens(" ".join(card.entities)))
    terms.update(content_tokens(card.name))
    return terms


def _bm25(query_terms: list[str], docs: list[list[str]]) -> list[float]:
    n = len(docs)
    if not n or not query_terms:
        return [0.0] * n
    avg_len = sum(len(doc) for doc in docs) / n or 1.0
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    scores = []
    for doc in docs:
        counts = Counter(doc)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len)
        total = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * tf * (BM25_K1 + 1) / (tf + norm)
        scores.append(total)
    return scores


def _entity_overlap(query_terms: set[str], card_terms: list[set[str]], cfg: RetrievalConfig) -> list[float]:
    n = len(card_terms)
    df: Counter[str] = Counter()
    for terms in card_terms:
        df.update(terms)
    idf = {term: math.log(1 + n / count) for term, count in df.items()}
    # Fuzzy partners per query term, computed once over the vocabulary rather than per card.
    fuzzy: dict[str, set[str]] = {}
    for q in query_terms:
        if len(q) < cfg.fuzzy_min_len or q in df:
            continue
        fuzzy[q] = {t for t in df if len(t) >= cfg.fuzzy_min_len and edit_distance_le1(q, t)}
    scores = []
    for terms in card_terms:
        total = 0.0
        for q in query_terms:
            if q in terms:
                total += idf[q]
                continue
            partners = fuzzy.get(q, set()) & terms
            if partners:
                total += cfg.fuzzy_weight * max(idf[t] for t in partners)
        scores.append(total)
    return scores


def _normalize(values: list[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top for v in values] if top > 0 else [0.0] * len(values)


def recency(card: AgentCard, now: datetime, cfg: RetrievalConfig = DEFAULT_CONFIG) -> float:
    age_days = max(0.0, (as_utc(now) - as_utc(card.last_active_at)).total_seconds() / 86400)
    return math.exp(-age_days / cfg.recency_days)


def score_cards(
    query: str,
    cards: Sequence[AgentCard],
    now: datetime,
    last_routed: Iterable[str] = (),
    cfg: Optional[RetrievalConfig] = None,
) -> list[tuple[AgentCard, float]]:
    """Score every card against `query`, best first (ties broken by most recently active)."""
    cfg = cfg or DEFAULT_CONFIG
    cards = list(cards)
    if not cards:
        return []
    terms = tokenize(query)
    kw = _normalize(_bm25(terms, [_doc_tokens(card) for card in cards]))
    query_terms = {t for t in terms if not is_common(t)}
    ent = _normalize(_entity_overlap(query_terms, [_entity_terms(card) for card in cards], cfg))
    boosted = set(last_routed)
    scored = []
    for card, kw_score, ent_score in zip(cards, kw, ent):
        score = cfg.w_kw * kw_score + cfg.w_ent * ent_score + cfg.w_rec * recency(card, now, cfg)
        if card.id in boosted:
            score += cfg.last_routed_boost
        scored.append((card, score))
    scored.sort(key=lambda item: (-item[1], -as_utc(item[0].last_active_at).timestamp()))
    return scored


def select(
    query: str,
    cards: Sequence[AgentCard],
    now: datetime,
    last_routed: Iterable[str] = (),
    cfg: Optional[RetrievalConfig] = None,
    pinned: Iterable[Union[AgentCard, str]] = (),
    open_work: Iterable[str] = (),
) -> list[AgentCard]:
    """Pinned cards, then the hot set, then the top K relevant non-archived cards; deduped, in that order."""
    cfg = cfg or DEFAULT_CONFIG
    by_id = {card.id: card for card in cards}
    chosen: list[AgentCard] = []
    seen: set[str] = set()

    def take(card: Optional[AgentCard]) -> None:
        if card is not None and card.id not in seen:
            seen.add(card.id)
            chosen.append(card)

    for item in pinned:
        take(by_id.get(item) if isinstance(item, str) else by_id.get(item.id, item))

    live = [card for card in cards if card.status != "archived"]
    open_ids = set(open_work)
    window = timedelta(hours=cfg.hot_window_hours)
    hot = [
        card
        for card in live
        if card.id in open_ids
        or (card.status != "dormant" and as_utc(now) - as_utc(card.last_active_at) <= window)
    ]
    hot.sort(key=lambda card: as_utc(card.last_active_at), reverse=True)
    for card in hot[: cfg.hot_max]:
        take(card)

    added = 0
    for card, score in score_cards(query, live, now, last_routed, cfg):
        if added >= cfg.k or score < cfg.min_score:
            break
        if card.id not in seen:
            take(card)
            added += 1
    return chosen


def search(
    query: str,
    cards: Sequence[AgentCard],
    now: datetime,
    cfg: Optional[RetrievalConfig] = None,
    limit: int = 10,
) -> list[AgentCard]:
    """Best matches over all cards, archived included."""
    cfg = cfg or DEFAULT_CONFIG
    scored = score_cards(query, cards, now, (), cfg)
    return [card for card, score in scored if score >= cfg.min_score][:limit]


def build_query(message: str, recent_texts: Sequence[str] = ()) -> str:
    """The new message alone; the last 3 conversation turns only when it has no content words.

    Names and topics from earlier turns otherwise outrank the request itself (on the lived world,
    adding context dropped recall@shown at 1,000 agents from 1.00 to 0.76). A contentless follow-up
    ("any update?") still needs context, and the last-routed agents reach it through the hot set.
    """
    if content_tokens(message):
        return message
    parts = [message, *list(recent_texts)[-3:]]
    return "\n".join(part for part in parts if part)


def pinned_from_agent_message(message: str, cards: Sequence[AgentCard]) -> list[AgentCard]:
    """Cards named by `[SUCCESS|FAILED] <name>:` lines in an execution-agent result."""
    by_name = {card.name: card for card in cards}
    by_norm = {normalize_name(card.name): card for card in cards}
    by_id = {card.id: card for card in cards}
    pinned: list[AgentCard] = []
    for name in _PINNED_RE.findall(message or ""):
        card = by_name.get(name) or by_norm.get(normalize_name(name)) or by_id.get(name)
        if card is not None and all(card.id != p.id for p in pinned):
            pinned.append(card)
    return pinned


__all__ = [
    "DEFAULT_CONFIG",
    "RetrievalConfig",
    "build_query",
    "content_tokens",
    "edit_distance_le1",
    "pinned_from_agent_message",
    "recency",
    "score_cards",
    "search",
    "select",
    "tokenize",
]
