"""Lexical diagnostics for factual answers, without semantic-accuracy claims.

The first-answer match is a contiguous token-span check, not an entailment
judge. Review flags never change the match result. References must be frozen
before evaluating a model; this module does not invent aliases.
"""

import re
import string
import unicodedata
from collections import Counter

FACT_ANSWER_SCORING_VERSION = "fact-answer-v1"

_TOKENS = re.compile(
    r"(?<![\w.])[+\-±]?(?:\d+(?:[.,]\d+)*|\.\d+)(?!\w|[.,'’‐‑–—-]\w)"
    r"|[^\W_]+(?:[.,'’‐‑–—-][^\W_]+)*",
    re.UNICODE,
)
_ANSWER_LABEL = re.compile(
    r"^(?:#{1,6}\s*)?(?:(?:final|short)\s+)?(?:answer|response)\s*[:：]\s*",
    re.IGNORECASE,
)
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st vs etc no fig vol inc ltd dept approx "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_NEGATION_OR_CORRECTION = re.compile(
    r"\b(?:no|not|never|neither|nor|cannot|incorrect|wrong|instead|rather|"
    r"however|actually|correction|but)\b|\b\w+n['’]t\b",
    re.IGNORECASE,
)
_ALTERNATIVE_OR_UNCERTAIN = re.compile(
    r"\b(?:either|or|maybe|perhaps|possibly|probably|unsure|uncertain)\b|"
    r"\b(?:i think|i believe|not sure|do not know)\b",
    re.IGNORECASE,
)


def normalized_answer_tokens(text):
    """Casefold Unicode words/numbers; preserve decimal boundaries and articles.

    Signs and decimal separators stay with numbers; alphanumeric identifiers
    stay whole. No synonyms, stemming, reordering, article deletion, or aliases
    are used. Thus ``Bini`` cannot match ``combining`` and ``-3.5`` cannot match
    ``3.5``. Letter-only dotted initials are split consistently with spaced ones.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("−", "-")
    tokens = []
    for token in _TOKENS.findall(text):
        if re.fullmatch(r"[^\W\d_](?:\.[^\W\d_])+", token, re.UNICODE):
            tokens.extend(token.split("."))
        else:
            tokens.append(token)
    return tokens


def _references(references):
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, (list, tuple)) or not references:
        raise ValueError("Fact-answer scoring requires nonempty text references.")
    if any(
        not isinstance(reference, str) or not normalized_answer_tokens(reference)
        for reference in references
    ):
        raise ValueError("Every fact-answer reference must contain a word or number.")
    return list(dict.fromkeys(references))


def _span_details(text):
    lines = str(text).strip().splitlines()
    line = ""
    remaining = ""
    for index, value in enumerate(lines):
        value = _ANSWER_LABEL.sub("", value.strip(), count=1).strip()
        if value:
            line = value
            remaining = "\n".join(lines[index + 1 :]).strip()
            break
    for index, character in enumerate(line):
        if character not in ".!?。！？":
            continue
        end = index + 1
        while end < len(line) and line[end] in "\"'”’)]}":
            end += 1
        if end < len(line) and not line[end].isspace():
            continue
        if character == ".":
            prefix = line[: index + 1]
            # Keep initials/initialisms and common abbreviations intact. A
            # sentence after an initialism may remain in this conservative span.
            if re.search(r"\b(?:[A-Za-z]\.)+$", prefix):
                continue
            word = re.search(r"([^\W\d_]+)\.$", prefix, re.UNICODE)
            if word and word.group(1).casefold() in _ABBREVIATIONS:
                continue
        suffix = line[end:].strip()
        return line[:end].strip(), bool(suffix or remaining)
    return line, bool(remaining)


def first_answer_span(text):
    """Return the first nonempty answer line, bounded by a conservative sentence.

    Only an anchored Answer/Response label is removed. Decimal punctuation,
    initials and common abbreviations are not sentence boundaries. This is a
    diagnostic extraction rule, not a claim that the selected text is an answer.
    """
    return _span_details(text)[0]


def _contains_tokens(text_tokens, reference_tokens):
    size = len(reference_tokens)
    return bool(size) and any(
        text_tokens[start : start + size] == reference_tokens
        for start in range(len(text_tokens) - size + 1)
    )


def _format_tokens(text):
    # Retain the existing short_answer full-output EM/F1 semantics. These are
    # format diagnostics; first-answer matching uses the stricter tokens above.
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def fact_answer_scores(generated, references):
    """Return format scores, first-answer lexical matches, and review hints.

    ``answer_exact_match``/``answer_token_f1`` retain legacy short_answer
    normalization. ``first_answer_token_match`` uses contiguous Unicode token
    spans; a match can still be negated, coincidental or contradicted later.
    ``needs_review`` is a conservative hint, never an automatic deduction.
    """
    references = _references(references)
    span, trailing_text = _span_details(generated)
    span_tokens = normalized_answer_tokens(span)
    reference_tokens = [normalized_answer_tokens(ref) for ref in references]
    matched = [
        ref
        for ref, tokens in zip(references, reference_tokens)
        if _contains_tokens(span_tokens, tokens)
    ]
    predicted = _format_tokens(generated)
    format_references = [_format_tokens(ref) for ref in references]
    f1 = []
    for tokens in format_references:
        overlap = sum((Counter(predicted) & Counter(tokens)).values())
        f1.append(2 * overlap / (len(predicted) + len(tokens)) if overlap else 0.0)
    flags = []
    if not span_tokens:
        flags.append("empty_answer")
    if _NEGATION_OR_CORRECTION.search(str(generated)):
        flags.append("negation_or_correction_language")
    if _ALTERNATIVE_OR_UNCERTAIN.search(str(generated)):
        flags.append("alternative_or_uncertain_language")
    if trailing_text:
        flags.append("text_after_first_answer_span")
    if "?" in span or "？" in span:
        flags.append("question_in_answer_span")
    return {
        "fact_answer_scoring_version": FACT_ANSWER_SCORING_VERSION,
        "answer_exact_match": bool(predicted) and predicted in format_references,
        "answer_token_f1": max(f1, default=0.0),
        "first_answer_span": span,
        "first_answer_exact_match": bool(span_tokens)
        and span_tokens in reference_tokens,
        "first_answer_token_match": bool(matched),
        "matched_reference": matched[0] if matched else None,
        "matched_references": matched,
        "review_flags": flags,
        "needs_review": bool(flags),
    }
