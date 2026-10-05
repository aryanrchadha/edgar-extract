"""Reconcile text-extracted values with XBRL and classify every mismatch.

For each extraction with a specific role, the steps are:

1. **Field.** The role selects canonical field(s): REVENUE -> Revenues, RD_EXPENSE ->
   ResearchAndDevelopmentExpense, ONE_TIME_CHARGE -> ImpairmentCharges or
   RestructuringCharges (by keyword), LITIGATION_RESERVE -> LitigationExpense /
   LitigationReserve. RESTATED_PRIOR uses the nearest field keyword in the sentence.
2. **Period.** The fiscal year is the year written next to the number ("$5.2 billion in
   2023"), or the matching year of a "..., respectively" list; "prior year" / "last year" /
   "nonrecurrence" wording means the year before the filing's; otherwise it is the filing's
   own fiscal year. Sub-annual figures (quarters, six months) have no annual counterpart.
3. **Fact.** The value as reported in the same filing; otherwise the latest value filed
   on or before that filing's date. Nothing filed later is ever used for the comparison.
4. **Status.** ``MATCH`` when |text| and |fact| agree within the text's rounding (half a
   unit of the last displayed digit), else ``MISMATCH``. ``NO_XBRL_COUNTERPART`` when
   there is no field, no fact for the period, or the figure is forward-looking or a
   percentage.

**Change amounts** ("revenue increased $1.0 billion", ``numeric_kind="delta"``) are
compared with fact(t) - fact(t-1), both as reported in the same filing
(``comparison="change"``). "compared to 2023" / "from 2023" names the base year, so t is
2024. A change mismatch is classified by restatement (the change computed with another
version of either year), segment (the same member's change, corroborated like levels), then
scale/rounding, else extraction error. The detail notes when the "change" equals a
year's level. Non-GAAP adjustments are not assessed for changes.

Every MISMATCH gets exactly one cause; the first rule that fires wins:

1. ``RESTATEMENT``: another filing carries a different value for the same field and
   period, and the text equals one of those other values (original or restated).
2. ``NON_GAAP_ADJUSTMENT``: the sentence has a non-GAAP cue ("adjusted", "excluding",
   "non-GAAP", "organic", "core", "constant currency", ...) and the difference from the GAAP
   fact equals another dollar amount in the sentence or a one-time-charge fact
   (restructuring, impairment, litigation) for the same period.
3. ``SEGMENT_VS_CONSOLIDATED``: the text equals a dimensional (segment, product,
   geography) inline-XBRL fact of the field's concepts for the same period.
4. ``SCALE_OR_ROUNDING``: the text equals the fact times 10^+-3, 10^+-6 or 10^+-9, or is
   off by at most one unit in its last displayed digit.
5. ``EXTRACTION_ERROR``: none of the above. ``cause_detail`` notes when the value
   matches the same field in another fiscal year (a period-attribution error).

Signs are compared in absolute value: prose writes "a loss of $3.7 billion" for a fact
stored as -3.7e9.

Precision: "$1.2 billion" only pins a value to +/-4%, and a 10-K holds hundreds of facts,
so rules that search many candidate values would otherwise fire on coincidences. Hence:
a restatement needs versions more than 2x the text's rounding apart; a non-GAAP
reconciliation needs a difference larger than 2x the rounding; a segment match needs the
member's name in the sentence or a number precise to 0.1%; a scale error must be a
thousand/million/billion slip on a number precise to 0.5%.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from edgar_extract.baseline_extract import _CUES, Role
from edgar_extract.ner.labeling import rounding_tolerance
from edgar_extract.universe import fiscal_year_from_period_end
from edgar_extract.xbrl import TagMap


class Status(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NO_XBRL_COUNTERPART = "NO_XBRL_COUNTERPART"


class Cause(StrEnum):
    RESTATEMENT = "RESTATEMENT"
    NON_GAAP_ADJUSTMENT = "NON_GAAP_ADJUSTMENT"
    SEGMENT_VS_CONSOLIDATED = "SEGMENT_VS_CONSOLIDATED"
    SCALE_OR_ROUNDING = "SCALE_OR_ROUNDING"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"


RECONCILED_ROLES = {
    "REVENUE",
    "RD_EXPENSE",
    "LITIGATION_RESERVE",
    "ONE_TIME_CHARGE",
    "RESTATED_PRIOR",
    "GUIDANCE",
}
ROLE_FIELDS: dict[str, list[str]] = {
    "REVENUE": ["Revenues"],
    "RD_EXPENSE": ["ResearchAndDevelopmentExpense"],
    "LITIGATION_RESERVE": ["LitigationExpense", "LitigationReserve"],
    "ONE_TIME_CHARGE": ["ImpairmentCharges", "RestructuringCharges"],
}
CUE_FIELDS: dict[Role, list[str]] = {
    Role.REVENUE: ["Revenues"],
    Role.RD_EXPENSE: ["ResearchAndDevelopmentExpense"],
    Role.LITIGATION_RESERVE: ["LitigationExpense", "LitigationReserve"],
    Role.ONE_TIME_CHARGE: ["ImpairmentCharges", "RestructuringCharges"],
}
ADJUSTMENT_FIELDS = ("RestructuringCharges", "ImpairmentCharges", "LitigationExpense")

_SUB_ANNUAL_RE = re.compile(
    r"\b(?:quarters?|three\s+months|six\s+months|nine\s+months|Q[1-4]|first\s+half|"
    r"second\s+half|month\s+ended)\b",
    re.I,
)
_NON_GAAP_RE = re.compile(
    r"\b(?:adjusted|excluding|exclud(?:e|es|ed)|non-?GAAP|organic|core|constant[- ]currency|"
    r"comparable|underlying|pro\s+forma|ex-)\b",
    re.I,
)
# A year: four digits not inside a larger number ("2,023,456", "$2023", "2023.5").
_YEAR_RE = re.compile(r"(?<![\d$.,])((?:19|20)\d{2})(?!\d|,\d|\.\d)")
_MONEY_RE = re.compile(r"\$\s*\(?\d")
_PRIOR_YEAR_RE = re.compile(
    r"\b(?:prior[- ]year|previous\s+year|last\s+year|year[- ]ago|nonrecurrence|non-recurrence)\b",
    re.I,
)
# A rule that claims a match against one of many candidate values needs a number precise
# enough that a chance match is unlikely (see the "Precision" note in the module docstring).
PRECISE_RELATIVE_TOL = 0.001
SCALE_RELATIVE_TOL = 0.005
_GENERIC_MEMBER_WORDS = {
    "member", "segment", "segments", "other", "others", "total", "revenue", "revenues",
    "product", "products", "service", "services", "and", "the", "all", "operating",
    "reportable", "corporate", "country", "statement", "axis",
}  # fmt: skip


def member_words(dims: str) -> set[str]:
    """Words of the dimension members: "us-gaap:...=jnj:OrthopaedicsMember" -> {"orthopaedics"}."""
    words: set[str] = set()
    for part in dims.split(";"):
        member = part.split("=")[-1].split(":")[-1].removesuffix("Member")
        for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", member):
            w = w.lower()
            if len(w) >= 3 and w not in _GENERIC_MEMBER_WORDS:
                words.add(w)
    return words


# Years a 10-K's comparatives can refer to, relative to its own fiscal year.
COMPARATIVE_YEARS = range(-3, 2)
YEAR_AFTER_WINDOW = 40
EPS = 1e-6


@dataclass(frozen=True)
class Period:
    fiscal_year: int
    source: str  # "explicit_year", "respectively", "prior_year_wording", "default_current"


def _clause_bounds(sentence: str, start: int, end: int) -> tuple[int, int]:
    left = max(sentence.rfind(";", 0, start), sentence.rfind(":", 0, start)) + 1
    right_candidates = [i for i in (sentence.find(";", end), sentence.find(":", end)) if i >= 0]
    return left, min(right_candidates) if right_candidates else len(sentence)


_BASE_YEAR_RE = re.compile(
    r"(?:compared\s+(?:to|with)|from|versus|vs\.?|over|relative\s+to)\s+"
    r"(?:the\s+)?(?:fiscal\s+(?:year\s+)?|FY\s?)?((?:19|20)\d{2})(?!\d)",
    re.I,
)


def change_period(
    sentence: str, start: int, end: int, current_fy: int, usd_spans: Sequence[tuple[int, int]]
) -> Period:
    """Fiscal year *t* of a change amount, where the change is fact(t) - fact(t-1).

    "increased $1.0 billion compared to 2023" names the base year, so t = 2024; otherwise
    the year is read as for a level ("increased $1.0 billion in 2024").
    """
    left, right = _clause_bounds(sentence, start, end)
    base = _BASE_YEAR_RE.search(sentence[left:right])
    if base:
        return Period(int(base.group(1)) + 1, "base_year")
    return mention_period(sentence, start, end, current_fy, usd_spans)


def mention_period(
    sentence: str, start: int, end: int, current_fy: int, usd_spans: Sequence[tuple[int, int]]
) -> Period:
    """Fiscal year a dollar amount refers to (see module docstring, step 2)."""
    if "respectively" in sentence.lower() and len(usd_spans) > 1:
        tail = sentence[usd_spans[-1][1] :]
        years = [int(y) for y in _YEAR_RE.findall(tail)]
        if len(years) == len(usd_spans) and (start, end) in usd_spans:
            return Period(years[usd_spans.index((start, end))], "respectively")
    left, right = _clause_bounds(sentence, start, end)
    after = sentence[end : min(right, end + YEAR_AFTER_WINDOW)]
    nxt = _MONEY_RE.search(after)
    if nxt:
        after = after[: nxt.start()]  # do not borrow the next amount's year
    m = _YEAR_RE.search(after)
    if m:
        return Period(int(m.group(1)), "explicit_year")
    before = [int(y) for y in _YEAR_RE.findall(sentence[left:start])]
    if before:
        return Period(before[-1], "explicit_year")
    if _PRIOR_YEAR_RE.search(sentence[left:right]):
        return Period(current_fy - 1, "prior_year_wording")
    return Period(current_fy, "default_current")


_FIELD_CUES: dict[str, re.Pattern[str]] = {
    "ImpairmentCharges": re.compile(r"impairment|write-?(?:off|down)", re.I),
    "RestructuringCharges": re.compile(r"restructuring|severance|exit\s+costs?", re.I),
    "LitigationReserve": re.compile(r"reserve|accru(?:al|ed)|liabilit", re.I),
    "LitigationExpense": re.compile(r"expense|charge|settlement|cost", re.I),
}


def _nearest_first(fields: list[str], sentence: str, start: int) -> list[str]:
    """Order candidate fields by how close their keyword is to the number (left side first)."""

    def distance(field: str) -> int:
        rx = _FIELD_CUES.get(field)
        if rx is None:
            return 10**6
        left = [start - m.end() for m in rx.finditer(sentence[:start])]
        right = [m.start() - start + 1000 for m in rx.finditer(sentence[start:])]
        return min([*left, *right], default=10**6)

    return sorted(fields, key=distance)


def candidate_fields(role: str, sentence: str, start: int) -> list[str]:
    if role == "RESTATED_PRIOR":
        left = sentence[:start]
        best: tuple[int, Role] | None = None
        for cue_role, rx in _CUES.items():
            for m in rx.finditer(left):
                if best is None or m.end() > best[0]:
                    best = (m.end(), cue_role)
        return CUE_FIELDS.get(best[1], []) if best else []
    return _nearest_first(list(ROLE_FIELDS.get(role, [])), sentence, start)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(abs(a) - abs(b)) <= tol + EPS * max(abs(a), abs(b), 1.0)


class Reconciler:
    """Holds the XBRL data for one run; :meth:`reconcile_row` handles one extraction."""

    def __init__(self, canonical: pd.DataFrame, inline: pd.DataFrame, tag_map: TagMap) -> None:
        self.tag_map = tag_map
        c = canonical.copy()
        c["fy_label"] = c["period_end"].map(lambda d: fiscal_year_from_period_end(d.date()))
        self.annual = c[c["period_type"] == "annual"]
        self.instant = c[c["period_type"] == "instant"]
        self.inline = inline
        self._annual_by = {k: g for k, g in self.annual.groupby(["cik", "canonical_field"])}
        self._instant_by = {k: g for k, g in self.instant.groupby(["cik", "canonical_field"])}
        self._inline_by = {k: g for k, g in inline.groupby("accession")} if len(inline) else {}

    # -- fact lookup ---------------------------------------------------------------------

    def _versions(self, cik: int, field: str, fy: int) -> pd.DataFrame:
        spec = self.tag_map.fields[field]
        if spec.period == "duration":
            g = self._annual_by.get((cik, field))
            return g[g["fy_label"] == fy] if g is not None else pd.DataFrame()
        g = self._instant_by.get((cik, field))
        if g is None:
            return pd.DataFrame()
        ends = self.annual[(self.annual["cik"] == cik) & (self.annual["fy_label"] == fy)][
            "period_end"
        ]
        return g[g["period_end"].isin(set(ends))]

    def fact_for(
        self, cik: int, field: str, fy: int, accession: str, filing_date: pd.Timestamp
    ) -> tuple[pd.Series | None, pd.DataFrame]:
        """The comparison fact and all versions for the same field and period."""
        versions = self._versions(cik, field, fy)
        if versions.empty:
            return None, versions
        same = versions[versions["accession"] == accession]
        if not same.empty:
            return same.iloc[0], versions
        known = versions[versions["filed"] <= filing_date].sort_values("filed")
        if known.empty:
            return None, versions
        return known.iloc[-1], versions

    # -- causes --------------------------------------------------------------------------

    def classify(
        self,
        text_value: float,
        tol: float,
        fact: pd.Series,
        versions: pd.DataFrame,
        field: str,
        sentence: str,
        other_amounts: Sequence[tuple[float, float]],
        cik: int,
        fy: int,
        accession: str,
    ) -> tuple[Cause, str]:
        fact_value = float(fact["value"])
        # 1. restatement: the text equals a *distinguishably* different version of the fact
        for v in versions.to_dict("records"):
            version_value = float(v["value"])
            distinguishable = abs(abs(version_value) - abs(fact_value)) > 2 * tol
            if distinguishable and _close(text_value, version_value, tol):
                filed = pd.Timestamp(v["filed"])
                when = "earlier" if filed < fact["filed"] else "later"
                return Cause.RESTATEMENT, (
                    f"text equals the {when} version {version_value:,.0f} filed {filed.date()} "
                    f"in {v['accession']}; compared fact {fact_value:,.0f}"
                )
        # 2. non-GAAP: cue + a difference larger than rounding that another amount explains
        diff = abs(abs(text_value) - abs(fact_value))
        if _NON_GAAP_RE.search(sentence) and diff > 2 * tol:
            for amount, amount_tol in other_amounts:
                if _close(diff, amount, max(tol, amount_tol)):
                    return Cause.NON_GAAP_ADJUSTMENT, (
                        f"difference {diff:,.0f} equals another amount in the sentence "
                        f"({amount:,.0f})"
                    )
            for adj in ADJUSTMENT_FIELDS:
                adj_fact, _ = self.fact_for(cik, adj, fy, accession, fact["filed"])
                if adj_fact is not None and _close(diff, float(adj_fact["value"]), tol):
                    return Cause.NON_GAAP_ADJUSTMENT, (
                        f"difference {diff:,.0f} equals {adj} {float(adj_fact['value']):,.0f}"
                    )
        # 3. segment / dimensional fact, corroborated by the member name or a precise number
        inline = self._inline_by.get(accession)
        if inline is not None and fact.get("period_type") == "annual":
            concepts = set(self.tag_map.fields[field].tags)
            dim = inline[
                inline["is_dimensional"]
                & inline["concept"].isin(concepts)
                & (inline["period_end"] == fact["period_end"])
                & (inline["period_start"] == fact["period_start"])
            ]
            precise = tol <= PRECISE_RELATIVE_TOL * abs(text_value)
            words = set(re.findall(r"[a-z]{3,}", sentence.lower()))
            for d in dim.itertuples(index=False):
                if not _close(text_value, float(d.value), tol):
                    continue
                named = bool(member_words(str(d.dims)) & words)
                if named or precise:
                    why = "member named in sentence" if named else "precise number"
                    return Cause.SEGMENT_VS_CONSOLIDATED, f"equals {d.concept} [{d.dims}] ({why})"
        # 4. scale (a thousand/million/billion slip) or rounding
        if fact_value and text_value and tol <= SCALE_RELATIVE_TOL * abs(text_value):
            for k in (-9, -6, -3, 3, 6, 9):
                if _close(text_value, fact_value * 10.0**k, tol):
                    return Cause.SCALE_OR_ROUNDING, f"text equals fact x 10^{k}"
        if _close(text_value, fact_value, 2 * tol):
            return Cause.SCALE_OR_ROUNDING, "off by at most one unit in the last displayed digit"
        # 5. extraction error
        detail = ""
        g = self._annual_by.get((cik, field))
        if g is not None:
            # Only years this filing's comparatives can refer to; older matches are chance.
            nearby = g[
                (g["fy_label"] != fy) & g["fy_label"].isin({fy + k for k in COMPARATIVE_YEARS})
            ]
            hits = nearby[nearby["value"].map(lambda v: _close(text_value, float(v), tol))]
            if not hits.empty:
                detail = f"matches the FY{int(hits['fy_label'].iloc[0])} value (period attribution)"
        return Cause.EXTRACTION_ERROR, detail

    # -- one row ---------------------------------------------------------------------------

    def reconcile_row(
        self, row: dict[str, Any], sentence_usd: Sequence[tuple[int, int, float, float]]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "canonical_field": None, "fiscal_year_ref": None, "period_source": None,
            "period_start": pd.NaT, "period_end": pd.NaT, "xbrl_tag": None,
            "xbrl_value": None, "xbrl_accession": None, "xbrl_filed": pd.NaT,
            "status": Status.NO_XBRL_COUNTERPART.value, "mismatch_cause": None,
            "cause_detail": "", "abs_diff": None, "rel_diff": None, "comparison": "level",
        }  # fmt: skip
        role = str(row["role"])
        sentence = str(row["sentence"])
        s = int(row["span_start"]) - int(row["sentence_start"])
        e = int(row["span_end"]) - int(row["sentence_start"])
        if role == "GUIDANCE":
            out["cause_detail"] = "forward-looking figure"
            return out
        if row["unit"] != "USD" or row["numeric_kind"] not in ("level", "delta"):
            out["cause_detail"] = f"not a dollar amount ({row['unit']}, {row['numeric_kind']})"
            return out
        if _SUB_ANNUAL_RE.search(sentence):
            out["cause_detail"] = "sub-annual figure; only annual facts are compared"
            return out
        fields = candidate_fields(role, sentence, s)
        if not fields:
            out["cause_detail"] = f"no XBRL field for role {role}"
            return out
        spans = [(a, b) for a, b, _, _ in sentence_usd]
        if row["numeric_kind"] == "delta":
            return self._reconcile_change(row, out, fields, sentence, s, e, spans)
        period = mention_period(sentence, s, e, int(row["fiscal_year"]), spans)
        out["fiscal_year_ref"], out["period_source"] = period.fiscal_year, period.source
        text_value = float(row["value"])
        tol = rounding_tolerance(str(row["raw_text"]), float(row["unit_scale"])) or 0.0
        filing_date = pd.Timestamp(row["filing_date"])
        compared: tuple[str, pd.Series, pd.DataFrame] | None = None
        for f in fields:
            fact, versions = self.fact_for(
                int(row["cik"]), f, period.fiscal_year, str(row["accession"]), filing_date
            )
            if fact is None:
                continue
            if compared is None:
                compared = (f, fact, versions)
            if _close(text_value, float(fact["value"]), tol):
                compared = (f, fact, versions)
                break
        if compared is None:
            out["canonical_field"] = fields[0]
            out["cause_detail"] = f"no XBRL fact for {', '.join(fields)} FY{period.fiscal_year}"
            return out
        field, fact, versions = compared
        fact_value = float(fact["value"])
        diff = abs(abs(text_value) - abs(fact_value))
        out.update(
            canonical_field=field,
            period_start=fact["period_start"],
            period_end=fact["period_end"],
            xbrl_tag=fact["tag"],
            xbrl_value=fact_value,
            xbrl_accession=fact["accession"],
            xbrl_filed=fact["filed"],
            abs_diff=diff,
            rel_diff=diff / abs(fact_value) if fact_value else None,
        )
        if _close(text_value, fact_value, tol):
            out["status"] = Status.MATCH.value
            return out
        out["status"] = Status.MISMATCH.value
        others = [(v, t) for a, b, v, t in sentence_usd if (a, b) != (s, e)]
        cause, detail = self.classify(
            text_value, tol, fact, versions, field, sentence, others,
            int(row["cik"]), period.fiscal_year, str(row["accession"]),
        )  # fmt: skip
        out["mismatch_cause"], out["cause_detail"] = cause.value, detail
        return out

    # -- change amounts ("increased $1.0 billion") ---------------------------------------

    def _reconcile_change(
        self,
        row: dict[str, Any],
        out: dict[str, Any],
        fields: list[str],
        sentence: str,
        s: int,
        e: int,
        spans: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        """Compare a change amount with fact(t) - fact(t-1) from the same filing."""
        out["comparison"] = "change"
        period = change_period(sentence, s, e, int(row["fiscal_year"]), spans)
        out["fiscal_year_ref"], out["period_source"] = period.fiscal_year, period.source
        t = period.fiscal_year
        text_value = abs(float(row["value"]))
        tol = rounding_tolerance(str(row["raw_text"]), float(row["unit_scale"])) or 0.0
        filing_date = pd.Timestamp(row["filing_date"])
        cik, accession = int(row["cik"]), str(row["accession"])
        chosen: tuple[str, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame] | None = None
        for f in fields:
            cur, cur_versions = self.fact_for(cik, f, t, accession, filing_date)
            prev, prev_versions = self.fact_for(cik, f, t - 1, accession, filing_date)
            if cur is None or prev is None:
                continue
            candidate = (f, cur, prev, cur_versions, prev_versions)
            if chosen is None:
                chosen = candidate
            if _close(text_value, float(cur["value"]) - float(prev["value"]), tol):
                chosen = candidate
                break
        if chosen is None:
            out["canonical_field"] = fields[0]
            out["cause_detail"] = (
                f"no XBRL facts for {', '.join(fields)} FY{t} and FY{t - 1} to take a change"
            )
            return out
        field, cur, prev, cur_versions, prev_versions = chosen
        expected = float(cur["value"]) - float(prev["value"])
        diff = abs(text_value - abs(expected))
        out.update(
            canonical_field=field,
            period_start=cur["period_start"],
            period_end=cur["period_end"],
            xbrl_tag=cur["tag"],
            xbrl_value=expected,
            xbrl_accession=cur["accession"],
            xbrl_filed=cur["filed"],
            abs_diff=diff,
            rel_diff=diff / abs(expected) if expected else None,
        )
        base_note = f"FY{t} {float(cur['value']):,.0f} - FY{t - 1} {float(prev['value']):,.0f}"
        if _close(text_value, expected, tol):
            out["status"] = Status.MATCH.value
            out["cause_detail"] = base_note
            return out
        out["status"] = Status.MISMATCH.value
        cause, detail = self._classify_change(
            text_value, tol, field, cur, prev, cur_versions, prev_versions, accession, sentence
        )
        out["mismatch_cause"], out["cause_detail"] = (
            cause.value,
            f"{detail}; {base_note}".strip("; "),
        )
        return out

    def _classify_change(
        self,
        text_value: float,
        tol: float,
        field: str,
        cur: pd.Series,
        prev: pd.Series,
        cur_versions: pd.DataFrame,
        prev_versions: pd.DataFrame,
        accession: str,
        sentence: str,
    ) -> tuple[Cause, str]:
        """Ordered causes for change amounts (non-GAAP is not assessed for changes)."""
        cur_v, prev_v = float(cur["value"]), float(prev["value"])
        expected = cur_v - prev_v
        # 1. restatement: the change computed with another version of either year
        for which, versions, other_fixed in (
            ("FY t-1", prev_versions, cur_v),
            ("FY t", cur_versions, prev_v),
        ):
            for v in versions.to_dict("records"):
                version_value = float(v["value"])
                base = prev_v if which == "FY t-1" else cur_v
                if abs(abs(version_value) - abs(base)) <= 2 * tol:
                    continue  # not distinguishable from the compared version
                alt = (
                    other_fixed - version_value
                    if which == "FY t-1"
                    else version_value - other_fixed
                )
                if _close(text_value, alt, tol):
                    return Cause.RESTATEMENT, (
                        f"equals the change using the {which} version {version_value:,.0f} "
                        f"filed {pd.Timestamp(v['filed']).date()}"
                    )
        # 2. segment: the same dimensional member's change between the two years
        inline = self._inline_by.get(accession)
        if inline is not None:
            concepts = set(self.tag_map.fields[field].tags)
            dims = inline[inline["is_dimensional"] & inline["concept"].isin(concepts)]

            def at(f: pd.Series) -> pd.DataFrame:
                same = (dims["period_end"] == f["period_end"]) & (
                    dims["period_start"] == f["period_start"]
                )
                return pd.DataFrame(dims[same])

            merged = at(cur).merge(at(prev), on=["concept", "dims"], suffixes=("_t", "_p"))
            precise = tol <= PRECISE_RELATIVE_TOL * text_value
            words = set(re.findall(r"[a-z]{3,}", sentence.lower()))
            for d in merged.to_dict("records"):
                change = float(d["value_t"]) - float(d["value_p"])
                if not _close(text_value, change, tol):
                    continue
                named = bool(member_words(str(d["dims"])) & words)
                if named or precise:
                    why = "member named in sentence" if named else "precise number"
                    return Cause.SEGMENT_VS_CONSOLIDATED, (
                        f"equals the change in {d['concept']} [{d['dims']}] ({why})"
                    )
        # 3. scale or rounding
        if expected and text_value and tol <= SCALE_RELATIVE_TOL * text_value:
            for k in (-9, -6, -3, 3, 6, 9):
                if _close(text_value, expected * 10.0**k, tol):
                    return Cause.SCALE_OR_ROUNDING, f"text equals the change x 10^{k}"
        if _close(text_value, expected, 2 * tol):
            return Cause.SCALE_OR_ROUNDING, "off by at most one unit in the last displayed digit"
        # 4. extraction error, noting a level read as a change
        for label, level in (("FY t", cur_v), ("FY t-1", prev_v)):
            if _close(text_value, level, tol):
                return Cause.EXTRACTION_ERROR, f"equals the {label} level, not a change"
        return Cause.EXTRACTION_ERROR, ""


def reconcile(
    extractions: pd.DataFrame, canonical: pd.DataFrame, inline: pd.DataFrame, tag_map: TagMap
) -> pd.DataFrame:
    """One output row per extraction with a reconcilable role (see module docstring)."""
    rec = Reconciler(canonical, inline, tag_map)
    todo = extractions[extractions["role"].isin(RECONCILED_ROLES)]
    usd_levels = extractions[
        (extractions["unit"] == "USD") & (extractions["numeric_kind"] == "level")
    ]
    per_sentence: dict[tuple[str, str, int], list[tuple[int, int, float, float]]] = {}
    for r in usd_levels.to_dict("records"):
        key = (str(r["accession"]), str(r["section"]), int(r["sentence_start"]))
        tol = rounding_tolerance(str(r["raw_text"]), float(r["unit_scale"])) or 0.0
        per_sentence.setdefault(key, []).append(
            (
                int(r["span_start"]) - int(r["sentence_start"]),
                int(r["span_end"]) - int(r["sentence_start"]),
                abs(float(r["value"])),
                tol,
            )
        )
    rows = []
    for r in todo.to_dict("records"):
        key = (str(r["accession"]), str(r["section"]), int(r["sentence_start"]))
        record = {str(k): v for k, v in r.items()}
        rows.append({**record, **rec.reconcile_row(record, per_sentence.get(key, []))})
    return pd.DataFrame(rows)
