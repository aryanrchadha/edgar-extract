"""edgar-extract QA dashboard (Streamlit + Plotly). Reads built artifacts only.

Launch with ``edgar-extract dashboard`` (or ``streamlit run`` on this file with
``EDGAR_EXTRACT_CONFIG`` pointing at config.yaml).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edgar_extract import qa
from edgar_extract.config import load_settings

CAUSE_ORDER = [
    "RESTATEMENT",
    "NON_GAAP_ADJUSTMENT",
    "SEGMENT_VS_CONSOLIDATED",
    "SCALE_OR_ROUNDING",
    "EXTRACTION_ERROR",
]


def config_path() -> Path:
    if "--config" in sys.argv:
        return Path(sys.argv[sys.argv.index("--config") + 1])
    return Path(os.environ.get("EDGAR_EXTRACT_CONFIG", "config.yaml"))


@st.cache_data(show_spinner=False)
def load_artifacts(path: str) -> dict[str, Any]:
    settings = load_settings(path)
    interim = settings.paths.interim_dir
    out: dict[str, Any] = {"missing": []}
    for name in ("universe", "extractions", "canonical_facts"):
        p = interim / f"{name}.parquet"
        if p.is_file():
            out[name] = pd.read_parquet(p)
        else:
            out["missing"].append(p.name)
    recs = []
    for p in sorted(interim.glob("reconciliation*.parquet")):
        df = pd.read_parquet(p)
        if not df.empty:
            recs.append(df)
    if recs:
        out["reconciliation"] = pd.concat(recs, ignore_index=True)
    else:
        out["missing"].append("reconciliation.parquet")
    pit = settings.paths.pit_db
    out["pit_coverage"] = qa.pit_coverage(pit) if pit.is_file() else None
    if not pit.is_file():
        out["missing"].append(pit.name)
    metrics = settings.paths.reports_dir / "ner_metrics.md"
    out["ner_metrics"] = metrics.read_text() if metrics.is_file() else None
    return out


def rate_bar(df: pd.DataFrame, x: str, rate: str, n_col: str, title: str) -> go.Figure:
    d = df.dropna(subset=[rate]).copy()
    d["err_plus"] = d[f"{rate}_hi"] - d[rate]
    d["err_minus"] = d[rate] - d[f"{rate}_lo"]
    d["sample"] = d["small_n"].map({True: "n < 30", False: "n >= 30"})
    fig = px.bar(
        d, x=x, y=rate, error_y="err_plus", error_y_minus="err_minus", color="sample",
        color_discrete_map={"n < 30": "#bbbbbb", "n >= 30": "#3b6ea8"},
        hover_data={n_col: True, f"{rate}_lo": ":.0%", f"{rate}_hi": ":.0%", "sample": False},
        labels={rate: rate.replace("_", " "), "sample": ""}, title=title,
    )  # fmt: skip
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(type="category")
    return fig


def main() -> None:
    st.set_page_config(page_title="edgar-extract QA", layout="wide")
    st.title("edgar-extract: extraction quality")
    cfg = config_path()
    data = load_artifacts(str(cfg))
    st.caption(
        f"Config `{cfg}`. Everything below is computed from local artifacts with no network "
        "access. Both rates are measured **against XBRL**, which is not the same as being "
        "correct: segment and non-GAAP numbers are correct readings that still mismatch "
        "consolidated XBRL. Grey bars have fewer than 30 observations."
    )
    if data["missing"]:
        st.warning(
            "Missing artifacts: " + ", ".join(data["missing"]) + ". Run `edgar-extract run`."
        )
    rec: pd.DataFrame | None = data.get("reconciliation")
    if rec is None or "universe" not in data or "canonical_facts" not in data:
        st.stop()
    assert rec is not None

    extractors = sorted(rec["extractor"].dropna().unique())
    extractor = st.sidebar.selectbox("Extractor", extractors)
    tickers = sorted(rec["ticker"].unique())
    chosen = st.sidebar.multiselect("Companies", tickers, default=tickers)
    r = rec[(rec["extractor"] == extractor) & rec["ticker"].isin(chosen)]
    universe = data["universe"][data["universe"]["ticker"].isin(chosen)]
    canonical = data["canonical_facts"]

    st.header("1. Precision and recall against XBRL")
    tab_co, tab_yr, tab_cause = st.tabs(["By company", "By fiscal year", "By mismatch category"])
    with tab_co:
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            rate_bar(qa.agreement(r, ["ticker"]), "ticker", "agreement", "n_comparable",
                     "Agreement with XBRL (MATCH / comparable)"),
            width="stretch",
        )  # fmt: skip
        c2.plotly_chart(
            rate_bar(qa.recall(r, canonical, universe, ["ticker"]), "ticker", "recall", "n_facts",
                     "XBRL recall (own key facts matched in prose)"),
            width="stretch",
        )  # fmt: skip
    with tab_yr:
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            rate_bar(qa.agreement(r, ["fiscal_year"]), "fiscal_year", "agreement", "n_comparable",
                     "Agreement with XBRL by fiscal year"),
            width="stretch",
        )  # fmt: skip
        c2.plotly_chart(
            rate_bar(qa.recall(r, canonical, universe, ["fiscal_year"]), "fiscal_year", "recall",
                     "n_facts", "XBRL recall by fiscal year"),
            width="stretch",
        )  # fmt: skip
    with tab_cause:
        causes = (
            r[r["status"] == "MISMATCH"]["mismatch_cause"]
            .value_counts()
            .reindex(CAUSE_ORDER)
            .fillna(0)
        )
        share = (causes / max(causes.sum(), 1)).rename("share").reset_index()
        share.columns = ["cause", "share"]
        fig = px.bar(
            share, x="cause", y="share", title="Mismatches by cause (share of all mismatches)"
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(qa.agreement(r, ["role"]), width="stretch")

    st.header("2. Mismatch-category breakdown by company")
    cb = qa.cause_breakdown(r, "ticker")
    fig = px.bar(
        cb, x="ticker", y="n", color="mismatch_cause",
        category_orders={"mismatch_cause": CAUSE_ORDER}, title="Mismatches by company and cause",
    )  # fmt: skip
    st.plotly_chart(fig, width="stretch")

    st.header("3. Role coverage per filing")
    if "extractions" in data:
        ex = data["extractions"]
        ex = ex[(ex["extractor"] == extractor) & ex["ticker"].isin(chosen)]
        cov = qa.role_coverage(ex)
        cols = st.columns(2)
        for col, section in zip(cols, ("7", "1A"), strict=True):
            part = cov[cov["section"] == section].sort_values("coverage", ascending=False)
            fig = px.bar(
                part, x="filing", y="coverage", hover_data=["structured", "n_spans", "accession"],
                title=f"Item {section}: share of numeric spans given a specific role",
            )  # fmt: skip
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            col.plotly_chart(fig, width="stretch")
        st.caption(
            "One bar per filing. Item 1A has few numbers; filings whose Item 1A contains no "
            "numeric span do not appear."
        )

    st.header("4. Point-in-time database coverage")
    pc = data.get("pit_coverage")
    if pc is None or pc.empty:
        st.info("No PIT database yet. Run `edgar-extract run --stage pit`.")
    else:
        pc = pc[pc["ticker"].isin(chosen) & (pc["fiscal_year"] >= pc["fiscal_year"].max() - 9)]
        grid = pc.pivot(index="ticker", columns="fiscal_year", values="versions")
        fig = px.imshow(
            grid, aspect="auto", color_continuous_scale="Blues", text_auto=True,
            labels={"color": "versioned rows"},
            title="Versioned XBRL facts per company and fiscal year (annual + balance-sheet)",
        )  # fmt: skip
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each cell counts stored rows (versions) for periods in that fiscal year: every "
            "filing that reported a value adds a row. Recent years are lighter because later "
            "10-Ks have not re-reported them yet, not because data is missing."
        )

    st.header("5. Extraction accuracy over time")
    over = qa.agreement(rec[rec["ticker"].isin(chosen)], ["extractor", "fiscal_year"])
    fig = px.line(
        over, x="fiscal_year", y="agreement", color="extractor", markers=True,
        hover_data=["n_comparable", "n_match"], title="Agreement with XBRL by fiscal year",
    )  # fmt: skip
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch")

    st.header("Gold-label metrics (human labels)")
    if data["ner_metrics"]:
        st.markdown(data["ner_metrics"])
    else:
        st.info(
            "No gold-label metrics yet: `reports/ner_metrics.md` appears after "
            "`edgar-extract label` and `edgar-extract ner evaluate`."
        )


main()
