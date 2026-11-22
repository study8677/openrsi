"""Compact coding trajectories by selecting verbatim evidence across turns."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, NamedTuple

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
_AUTHORITY = frozenset(("system", "developer", "user"))
_TOOL = frozenset(("tool", "function"))

_OVERRIDE = re.compile(
    r"\b(?:supersed\w*|overrid\w*|correction|clarification|instead|"
    r"no longer|rather than|not anymore|ignore (?:the )?(?:previous|earlier)|"
    r"changed (?:to|from)|revert(?:ed)?|final decision)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(?:do not|don't|must not|never|avoid|pitfall|gotcha|regression|"
    r"not supported|not allowed|wrong|failed attempt|does not work|"
    r"did not work|fails? (?:if|when)|breaks? (?:if|when))\b",
    re.IGNORECASE,
)
_REQUIREMENT = re.compile(
    r"\b(?:must|required|shall|only|always|exact(?:ly)?|preserve|"
    r"ensure|expected|contract|acceptance criteria|constraint\w*|"
    r"implement|return(?:s|ed)?|raise(?:s|d)?|reject(?:s|ed)?|"
    r"default(?:s)?|boundary|inclusive|exclusive|idempoten\w*)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC = re.compile(
    r"\b(?:warn(?:ing)?|error|failed|failure|traceback|exception|"
    r"deprecated|mismatch|assertion|fatal|invalid|unsupported)\b|"
    r"\b\w+(?:Error|Exception)\b",
    re.IGNORECASE,
)
_LITERAL = re.compile(
    r"`[^`\n]+`|[\"'][^\"'\n]{1,100}[\"']|"
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|"
    r"\b\w+(?:\.\w+)+\b|\b\w+\([^()\n]{0,120}\)|"
    r"(?:<=|>=|==|!=|->)|\b\d+(?:\.\d+)?\b"
)
_NOISE = re.compile(
    r"^\s*(?:downloading|collecting|installing collected packages|"
    r"receiving objects|resolving deltas|enumerating objects|"
    r"counting objects|compressing objects|[\d.]+%)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s+\S.*|(?:[A-Za-z][\w /&-]{1,70}):)\s*$"
)
_FENCE = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})(?P<suffix>.*)$")
_SENTENCE = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-Z][a-z]|[A-Z]{2,}\b))|"
    r"(?<=;)\s+|(?<=\|)\s+"
)
_ANAPHORA = re.compile(
    r"^\s*(?:[-*]\s+)?(?:this|that|it|they|these|those|otherwise|"
    r"except|instead|also|however|but|if so|for example)\b",
    re.IGNORECASE,
)
_TRACE_FRAME = re.compile(r'^\s*File ".+", line \d+')
_FINGERPRINT = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\s*|"
    r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*"
)


class _Piece(NamedTuple):
    message: int
    text: str
    previous: int | None
    heading: int | None
    fenced: bool


def _encode(text: str) -> list[int]:
    return _ENC.encode_ordinary(text)


def count_tokens(text: str) -> int:
    """Count literal text tokens using cl100k_base."""
    return len(_encode(text))


def _content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text", part.get("content"))
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(value)


def _pieces(contents: list[str]) -> list[_Piece]:
    result: list[_Piece] = []
    for message, content in enumerate(contents):
        previous = None
        heading = None
        fence = ""
        for line in content.splitlines():
            if not line.strip():
                previous = None
                continue
            marker = _FENCE.match(line)
            if marker:
                marker_text = marker["marker"]
                if not fence:
                    fence = marker_text
                    continue
                if (
                    marker_text[0] == fence[0]
                    and len(marker_text) >= len(fence)
                    and not marker["suffix"].strip()
                ):
                    fence = ""
                    continue

            is_heading = not fence and bool(_HEADING.fullmatch(line))
            parts = (
                _SENTENCE.split(line)
                if not fence and not is_heading and len(line) > 480
                else [line]
            )
            for part in parts:
                if not part.strip():
                    continue
                position = len(result)
                result.append(
                    _Piece(message, part.rstrip(), previous, heading, bool(fence))
                )
                if is_heading:
                    heading = position
                previous = position
    return result


def _priority(piece: _Piece, role: str, message_count: int) -> float:
    text = piece.text
    authority = role in _AUTHORITY
    if authority:
        score = 12.0 if role == "user" else 14.0
    elif role in _TOOL:
        score = 0.6
    else:
        score = 3.0

    if _OVERRIDE.search(text):
        score += 10.0 if authority else 6.0
    if _NEGATIVE.search(text):
        score += 7.0 if authority else 5.5
    if _REQUIREMENT.search(text):
        score += 5.0
    if _DIAGNOSTIC.search(text):
        score += 8.0 if role in _TOOL else 4.0
    if _LITERAL.search(text):
        score += 2.5
    if piece.fenced and re.match(
        r"\s*(?:async\s+def|def|class|interface|type|export|function)\s+", text
    ):
        score += 5.0
    if not authority and _NOISE.match(text) and not _DIAGNOSTIC.search(text):
        score *= 0.15
    if _TRACE_FRAME.match(text):
        score -= 2.5
    score += 2.2 * piece.message / max(1, message_count - 1)
    return max(0.15, score)


def _render(
    selected: dict[int, str], pieces: list[_Piece], headers: list[str]
) -> str:
    output: list[str] = []
    previous_message = -1
    previous_position = -2
    for position in sorted(selected):
        message = pieces[position].message
        if message != previous_message:
            if output:
                output.append("")
            output.append(headers[message])
        elif position != previous_position + 1:
            output.append("…")
        output.append(selected[position])
        previous_message = message
        previous_position = position
    return "\n".join(output)


def _excerpt(text: str, tokens: list[int], limit: int) -> str:
    """Keep the start and the window around the strongest buried signal."""
    if len(tokens) <= limit:
        return text
    if limit <= 0:
        return ""
    reserve = max(0, limit - 3)
    if not reserve:
        return _ENC.decode(tokens[:limit], errors="ignore").rstrip()

    signal = (
        _OVERRIDE.search(text)
        or _NEGATIVE.search(text)
        or _DIAGNOSTIC.search(text)
        or _REQUIREMENT.search(text)
    )
    if signal:
        signal_at = count_tokens(text[:signal.start()])
        if signal_at >= reserve // 2:
            head = reserve // 4
            width = reserve - head
            start = max(head, min(signal_at - width // 4, len(tokens) - width))
            prefix = _ENC.decode(tokens[:head], errors="ignore").rstrip()
            window = _ENC.decode(tokens[start:start + width], errors="ignore").strip()
            return prefix + " … " + window + (
                " …" if start + width < len(tokens) else ""
            )
    return _ENC.decode(tokens[:reserve], errors="ignore").rstrip() + " …"


def compact(
    messages: list[dict[str, Any]],
    max_tokens: int,
    llm_call: Callable[[str], str] | None = None,
) -> str:
    """Preserve instructions and diagnostic evidence without rewriting them.

    Selection uses separate soft budgets for instructions and historical
    evidence, then spends the remaining budget on the best material from
    either group. Output always follows source order: conflicting statements
    retain their roles and chronology rather than being merged into a new rule.
    """
    if max_tokens <= 0 or not messages:
        return ""

    roles = [str(message.get("role") or "unknown").lower() for message in messages]
    contents = [_content(message.get("content")) for message in messages]
    headers = [
        f"[Turn {message.get('turn', index + 1)} | {roles[index].upper()}]"
        for index, message in enumerate(messages)
    ]
    original = "\n\n".join(
        f"{header}\n{content}"
        for header, content in zip(headers, contents)
        if content.strip()
    )
    if count_tokens(original) <= max_tokens:
        return original

    pieces = _pieces(contents)
    if not pieces:
        return ""
    authority = [
        position for position, piece in enumerate(pieces)
        if roles[piece.message] in _AUTHORITY
    ]
    evidence = [
        position for position, piece in enumerate(pieces)
        if roles[piece.message] not in _AUTHORITY
    ]
    token_lists = [_encode(piece.text) for piece in pieces]
    priorities = [
        _priority(piece, roles[piece.message], len(messages)) for piece in pieces
    ]
    header_cost = [count_tokens(header) + 2 for header in headers]

    def key(position: int) -> tuple[float, int]:
        length = len(token_lists[position])
        return priorities[position] / math.sqrt(1 + length / 65), position

    ranked = sorted(range(len(pieces)), key=key, reverse=True)
    selected: dict[int, str] = {}
    selection_order: list[int] = []
    opened: set[int] = set()
    repetitions: Counter[tuple[str, str]] = Counter()
    spent = 0

    def add(position: int, ceiling: int, allow_excerpt: bool) -> bool:
        nonlocal spent
        if position in selected:
            return True
        piece = pieces[position]
        role = roles[piece.message]
        fingerprint = (role, _FINGERPRINT.sub("", piece.text.strip()))
        if role not in _AUTHORITY and repetitions[fingerprint]:
            return False
        overhead = 3 + (0 if piece.message in opened else header_cost[piece.message])
        available = ceiling - spent - overhead
        tokens = token_lists[position]
        if available <= 0:
            return False
        if len(tokens) <= available:
            text = piece.text
        elif allow_excerpt and available >= 20:
            text = _excerpt(piece.text, tokens, available)
        else:
            return False
        cost = count_tokens(text) + overhead
        if not text.strip() or spent + cost > ceiling:
            return False
        selected[position] = text
        selection_order.append(position)
        opened.add(piece.message)
        repetitions[fingerprint] += 1
        spent += cost
        return True

    def select(positions: list[int], ceiling: int) -> None:
        for position in positions:
            if spent >= ceiling:
                return
            piece = pieces[position]
            large = len(token_lists[position]) > max(96, max_tokens // 5)
            if not add(
                position,
                min(ceiling, spent + max(64, max_tokens // 5)) if large else ceiling,
                allow_excerpt=large and priorities[position] >= 8,
            ):
                continue
            previous = piece.previous
            heading = piece.heading
            if previous is not None and (
                _ANAPHORA.match(piece.text)
                or (
                    _DIAGNOSTIC.search(piece.text)
                    and _TRACE_FRAME.match(pieces[previous].text)
                )
            ):
                if len(token_lists[previous]) <= 65:
                    add(previous, ceiling, allow_excerpt=False)
            if heading is not None and len(token_lists[heading]) <= 35:
                add(heading, ceiling, allow_excerpt=False)

    if authority and evidence:
        select(sorted(authority, key=key, reverse=True), max_tokens * 3 // 5)
        select(sorted(evidence, key=key, reverse=True), min(
            max_tokens, spent + max_tokens * 2 // 5
        ))
    select(ranked, max_tokens)

    result = _render(selected, pieces, headers)
    while result and count_tokens(result) > max_tokens:
        del selected[selection_order.pop()]
        result = _render(selected, pieces, headers)
    if result:
        return result

    best_position = ranked[0]
    piece = pieces[best_position]
    header = headers[piece.message] + "\n"
    tokens = token_lists[best_position]
    for prefix in (header, ""):
        available = min(len(tokens), max_tokens - count_tokens(prefix))
        while available > 0:
            excerpt = _excerpt(piece.text, tokens, available)
            candidate = prefix + excerpt
            if excerpt.strip() and count_tokens(candidate) <= max_tokens:
                return candidate
            available -= 1
    return ""
