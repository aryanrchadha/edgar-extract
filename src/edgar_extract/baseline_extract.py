"""Rule-based first-pass role assignment for numeric mentions ("regex" extractor).

Roles, checked in this order (the first rule that fires wins):

1. ``GUIDANCE``: the sentence has a company forward-looking cue (``we expect``, ``we
   anticipate``, ``we plan/intend/target/estimate``, ``is expected to``, ``our ...
   outlook/guidance``) and the mention is an amount, a percentage or a per-share figure.
   ``expected credit losses`` (an accounting term) is not a cue.
2. ``RESTATED_PRIOR``: the sentence says a figure was restated or recast (``as restated``,
   ``have been recast``, ``restated to``, ``previously reported``, ``as revised``) and the
   mention is a dollar amount. ``amended and restated`` (contracts) does not count.
3. For dollar amounts only, the **nearest keyword to the left** within the sentence (at
   most 150 characters back) decides between ``LITIGATION_RESERVE``,
   ``ONE_TIME_CHARGE``, ``RD_EXPENSE`` and ``REVENUE``. If there is none on the left, the
   first keyword within 60 characters to the right decides ("$1.2 billion of revenue").
4. Everything else is ``OTHER_NUMERIC``.

The roles map onto the NER label set as ``RD_EXPENSE`` and ``LITIGATION_RESERVE`` to
``OTHER_NUMERIC`` (see :data:`NER_ROLE`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from edgar_extract.numeric_spans import NumericMention

EXTRACTOR_NAME = "regex_v1"


class Role(StrEnum):
    REVENUE = "REVENUE"
    RD_EXPENSE = "RD_EXPENSE"
    LITIGATION_RESERVE = "LITIGATION_RESERVE"
    GUIDANCE = "GUIDANCE"
    ONE_TIME_CHARGE = "ONE_TIME_CHARGE"
    RESTATED_PRIOR = "RESTATED_PRIOR"
    OTHER_NUMERIC = "OTHER_NUMERIC"


# Label set of the token classifier (M4).
NER_ROLE: dict[Role, str] = {
    Role.REVENUE: "REVENUE",
    Role.GUIDANCE: "GUIDANCE",
    Role.ONE_TIME_CHARGE: "ONE_TIME_CHARGE",
    Role.RESTATED_PRIOR: "RESTATED_PRIOR",
    Role.RD_EXPENSE: "OTHER_NUMERIC",
    Role.LITIGATION_RESERVE: "OTHER_NUMERIC",
    Role.OTHER_NUMERIC: "OTHER_NUMERIC",
}

LEFT_WINDOW = 150
RIGHT_WINDOW = 60

_GUIDANCE_RE = re.compile(
    r"\b(?:we|management)\s+(?:currently\s+|now\s+)?(?:expects?|anticipates?|plans?|intends?|"
    r"targets?|projects?|forecasts?|estimates?)\b"
    r"|\b(?:is|are)\s+(?:currently\s+)?(?:expected|anticipated|projected|forecast(?:ed)?)\s+to\b"
    r"|\bour\s+(?:[\w-]+\s+){0,3}(?:outlook|guidance)\b",
    re.I,
)
_NOT_GUIDANCE_RE = re.compile(r"expected\s+credit\s+loss", re.I)
_RESTATED_RE = re.compile(
    r"\bas\s+restated\b|\brestatement\s+of\b|\bhave\s+been\s+(?:restated|recast|revised)\b"
    r"|\b(?:was|were|been)\s+(?:restated|recast)\s+(?:to|for)\b|\brestated\s+(?:to|for)\b"
    r"|\bpreviously\s+reported\b|\bas\s+revised\b|\brevised\s+to\s+reflect\b",
    re.I,
)
_NOT_RESTATED_RE = re.compile(r"amended\s+and\s+restated", re.I)
_CUES: dict[Role, re.Pattern[str]] = {
    Role.LITIGATION_RESERVE: re.compile(
        r"\b(?:litigation|legal\s+(?:reserves?|accruals?|proceedings|matters|settlements?|costs?|"
        r"charges?|expenses?)|lawsuits?|loss\s+contingenc(?:y|ies)|opioid|talc|"
        r"settlements?\s+(?:of|with|related\s+to)\s+(?:the\s+)?(?:litigation|lawsuits?|claims|"
        r"legal|SEC|DOJ|plaintiffs|states|attorneys\s+general))\b",
        re.I,
    ),
    Role.ONE_TIME_CHARGE: re.compile(
        r"\b(?:impairments?|restructuring|write-?(?:offs?|downs?)|one-time|non-?recurring|"
        r"severance|exit\s+costs?|special\s+(?:items?|charges?)|loss(?:es)?\s+on\s+"
        r"(?:extinguishment|divestiture|sale)|(?:pre-?tax|non-?cash|one-?time|other\s+operating|"
        r"acquisition-related|transformation)\s+charges?|recorded\s+(?:an?\s+)?(?:[\w-]+\s+){0,3}charges?)\b",
        re.I,
    ),
    Role.RD_EXPENSE: re.compile(r"\b(?:research\s+and\s+development|R&D)\b", re.I),
    Role.REVENUE: re.compile(r"\b(?:revenues?|net\s+sales|total\s+sales|sales)\b", re.I),
}
# A REVENUE cue inside one of these phrases is not about revenue.
_NOT_REVENUE_RE = re.compile(
    r"cost\s+of\s+(?:net\s+)?(?:revenues?|sales)|sales\s+and\s+marketing|selling,?\s+general|"
    r"sales\s+(?:tax|taxes)|sales-type|sales\s+of\s+(?:property|assets|investments|securities|"
    r"businesses)|revenue\s+recognition|deferred\s+revenue|unearned\s+revenue|revenue\s+bonds",
    re.I,
)


@dataclass(frozen=True)
class RoleDecision:
    role: Role
    cue: str  # the keyword or rule that decided the role, for auditing


def _cue_matches(role: Role, text: str) -> list[re.Match[str]]:
    matches = list(_CUES[role].finditer(text))
    if role is Role.REVENUE:
        bad = [(b.start(), b.end()) for b in _NOT_REVENUE_RE.finditer(text)]
        matches = [m for m in matches if not any(s <= m.start() < e for s, e in bad)]
    return matches


def assign_role(sentence: str, mention: NumericMention) -> RoleDecision:
    money = mention.unit == "USD" and mention.numeric_kind in ("level", "delta")
    if mention.unit in ("USD", "USD/share", "percent"):
        cleaned = _NOT_GUIDANCE_RE.sub("", sentence)
        g = _GUIDANCE_RE.search(cleaned)
        if g:
            return RoleDecision(Role.GUIDANCE, g.group(0))
    if not money:
        return RoleDecision(Role.OTHER_NUMERIC, "")
    r = _RESTATED_RE.search(_NOT_RESTATED_RE.sub("", sentence))
    if r:
        return RoleDecision(Role.RESTATED_PRIOR, r.group(0))

    left_start = max(0, mention.start - LEFT_WINDOW)
    left = sentence[left_start : mention.start]
    best: tuple[int, Role, str] | None = None
    for role in _CUES:
        for m in _cue_matches(role, left):
            if best is None or m.end() > best[0]:
                best = (m.end(), role, m.group(0))
    if best is not None:
        return RoleDecision(best[1], best[2])

    right = sentence[mention.end : mention.end + RIGHT_WINDOW]
    first: tuple[int, Role, str] | None = None
    for role in _CUES:
        for m in _cue_matches(role, right):
            if first is None or m.start() < first[0]:
                first = (m.start(), role, m.group(0))
    if first is not None:
        return RoleDecision(first[1], first[2])
    return RoleDecision(Role.OTHER_NUMERIC, "")
