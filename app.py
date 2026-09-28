"""Spend Data Quality Workbench.

A small analytics tool for the first step of any procurement analytics project:
making spend data trustworthy before anyone builds a dashboard on top of it.

Pipeline
    1. Standardise the input schema (supplier, amount, date, category code).
    2. Normalise supplier names and cluster variants (token-sort similarity +
       average-linkage hierarchical clustering).
    3. Run rule- and statistics-based data quality checks.
    4. Translate the findings into spend impact and a "fix first" list.

The demo dataset is synthetic and carries hidden ground truth, so the supplier
matching can be evaluated objectively (precision / recall / F1).
See README.md for the methodology.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rapidfuzz import fuzz, process
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SEED = 42
MAX_UNIQUE_KEYS = 4000  # guardrail for the O(n^2) similarity matrix
DEFAULT_THRESHOLD = 88
OUTLIER_CUTOFF = 3.5  # Iglewicz & Hoaglin (1993) modified z-score rule of thumb

INK, TEAL, SLATE, AMBER, BRICK = "#16232E", "#0F6E7C", "#5A6B77", "#C7781E", "#B23A48"
COLORWAY = ["#0F6E7C", "#3B5B92", "#C7781E", "#7A4E7E", "#5E8C4A", "#B23A48", "#8A9BA8", "#A8893A"]

LEGAL_FORMS = {
    "oy", "oyj", "ab", "ky", "tmi", "ltd", "limited", "gmbh", "as", "inc",
    "llc", "plc", "sa", "bv", "nv", "co", "corp",
}

# Simplified labels for CPV divisions (first two digits of the CPV code).
CPV_DIVISIONS = {
    "03": "Agriculture & forestry", "09": "Fuels & energy", "14": "Mining products",
    "15": "Food & beverages", "18": "Clothing & footwear", "22": "Printed matter",
    "24": "Chemicals", "30": "Office & computing equipment", "31": "Electrical equipment",
    "32": "Telecom equipment", "33": "Medical equipment", "34": "Transport equipment",
    "35": "Security equipment", "38": "Laboratory equipment", "39": "Furniture & fittings",
    "41": "Water", "42": "Industrial machinery", "44": "Construction materials",
    "45": "Construction work", "48": "Software packages", "50": "Repair & maintenance",
    "51": "Installation services", "55": "Hotel & catering", "60": "Transport services",
    "63": "Transport support services", "64": "Postal & telecom services",
    "65": "Utilities", "66": "Financial & insurance", "70": "Real estate",
    "71": "Engineering services", "72": "IT services", "73": "Research & development",
    "75": "Public administration", "76": "Oil & gas services", "77": "Agricultural services",
    "79": "Business services", "80": "Education & training", "85": "Health & social",
    "90": "Cleaning & environmental", "92": "Recreation & culture", "98": "Other services",
}

# Demo generator settings ---------------------------------------------------- #
STEMS = ["Nordic", "Baltic", "Kaleva", "Aurora", "Helsinki", "Tampere", "Suomi", "Lakeland",
         "Polar", "Vega", "Siirto", "Terra", "Koivu", "Salmi", "Metsa", "Laine", "Virta",
         "Tuuli", "Arctic", "Boreal"]
SECTOR_DIVISIONS = {
    "Freight": ["60", "63"], "Consulting": ["79", "71"], "Cleaning": ["90"],
    "Software": ["48", "72"], "Construction": ["45"], "Engineering": ["71"],
    "Logistics": ["60", "63"], "Facility": ["50", "90"], "Digital": ["72"], "Office": ["30", "39"],
}
FORMS = ["Oy", "Oy Ab", "Ltd", "Oyj", "AB", "GmbH"]
MESSINESS = {  # probabilities per row
    "Low": dict(name=0.15, no_cat=0.03, no_amt=0.010, no_sup=0.003, bad_cat=0.010, dup=0.015, out=0.005, neg=0.003),
    "Medium": dict(name=0.35, no_cat=0.08, no_amt=0.020, no_sup=0.005, bad_cat=0.030, dup=0.040, out=0.012, neg=0.006),
    "High": dict(name=0.60, no_cat=0.18, no_amt=0.050, no_sup=0.012, bad_cat=0.060, dup=0.080, out=0.025, neg=0.012),
}

st.set_page_config(page_title="Spend Data Quality Workbench", layout="wide")


# --------------------------------------------------------------------------- #
# Synthetic benchmark with ground truth
# --------------------------------------------------------------------------- #

def _typo(text: str, rng: np.random.Generator) -> str:
    """Delete, transpose or double one interior character."""
    if len(text) < 4:
        return text
    i = int(rng.integers(1, len(text) - 1))
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return text[:i] + text[i + 1:]
    if kind == 1:
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text[:i] + text[i] + text[i:]


def render_name(core: str, form: str, rng: np.random.Generator, p_variant: float) -> str:
    """Return the canonical name, or with probability p_variant a realistic corruption."""
    if rng.random() > p_variant:
        return f"{core} {form}"
    op = rng.choice(
        ["form", "upper", "noform", "punct", "typo", "lead", "reorder", "abbr"],
        p=[0.20, 0.15, 0.20, 0.10, 0.15, 0.08, 0.06, 0.06],
    )
    words = core.split()
    if op == "form":
        return f"{core} {rng.choice(FORMS)}"
    if op == "upper":
        return f"{core} {form}".upper()
    if op == "noform":
        return core
    if op == "punct":
        return f"{core}, {form}."
    if op == "typo":
        return f"{_typo(core, rng)} {form}"
    if op == "lead":
        return f"Oy {core} Ab"
    if op == "reorder":
        return f"{' '.join(reversed(words))} {form}"
    return f"{words[0][:3]}. {' '.join(words[1:])} {form}"  # abbreviated first word


@st.cache_data(show_spinner=False)
def generate_demo_data(level: str, n_rows: int = 12_000, n_suppliers: int = 140) -> pd.DataFrame:
    """Simulate public-procurement awards with known supplier identities and injected defects."""
    rng = np.random.default_rng(SEED)
    p = MESSINESS[level]

    # Supplier master, including "hard negatives": distinct suppliers with near-identical names.
    combos = [(s, sec) for s in STEMS for sec in SECTOR_DIVISIONS]
    masters = []
    for k in rng.choice(len(combos), size=n_suppliers, replace=False):
        stem, sec = combos[k]
        masters.append(dict(core=f"{stem} {sec}", form=str(rng.choice(FORMS)),
                            division=str(rng.choice(SECTOR_DIVISIONS[sec]))))
    for k in rng.choice(n_suppliers, size=int(0.12 * n_suppliers), replace=False):
        base = masters[int(k)]
        masters.append(dict(base, core=f"{base['core']} {rng.choice(['Services', 'Group', 'Solutions'])}"))
    m = len(masters)

    weights = rng.pareto(1.3, size=m) + 1.0  # heavy-tailed spend concentration
    weights /= weights.sum()
    sid = rng.choice(m, size=n_rows, p=weights)
    mu = rng.normal(10.2, 1.0, size=m)
    amount = np.round(np.exp(mu[sid] + rng.normal(0, 0.6, size=n_rows)), 2)
    dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, size=n_rows), unit="D")
    division = np.array([masters[i]["division"] for i in sid])
    cpv = np.char.add(division.astype("U2"), np.char.zfill(rng.integers(0, 10**6, n_rows).astype(str), 6))

    df = pd.DataFrame({
        "supplier_name": [render_name(masters[i]["core"], masters[i]["form"], rng, p["name"]) for i in sid],
        "contract_value": amount,
        "award_date": dates,
        "cpv_code": cpv,
        "true_supplier_id": sid,
        "_inj_duplicate": False,
        "_inj_outlier": False,
    })

    # Outliers: amounts inflated by 100x-1600x (typical unit / decimal-place errors).
    out_idx = rng.choice(n_rows, size=int(p["out"] * n_rows), replace=False)
    df.loc[out_idx, "contract_value"] = (df.loc[out_idx, "contract_value"] * 10 ** rng.uniform(2, 3.2, len(out_idx))).round(2)
    df.loc[out_idx, "_inj_outlier"] = True

    # Duplicates: the same award re-entered, often under a different spelling of the supplier.
    dup = df.loc[rng.choice(n_rows, size=int(p["dup"] * n_rows), replace=False)].copy()
    dup["supplier_name"] = [render_name(masters[i]["core"], masters[i]["form"], rng, 0.7) for i in dup["true_supplier_id"]]
    dup["_inj_duplicate"] = True
    df = pd.concat([df, dup], ignore_index=True)

    # Missing / invalid values.
    n = len(df)
    df.loc[rng.random(n) < p["no_cat"], "cpv_code"] = None
    bad = rng.random(n) < p["bad_cat"]
    df.loc[bad, "cpv_code"] = rng.choice(["N/A", "7200000", "72xx0000", "0"], size=int(bad.sum()))
    df.loc[rng.random(n) < p["no_amt"], "contract_value"] = np.nan
    neg = rng.random(n) < p["neg"]
    df.loc[neg, "contract_value"] = -df.loc[neg, "contract_value"].abs()
    df.loc[rng.random(n) < p["no_sup"], "supplier_name"] = None

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "notice_id", [f"N{i:06d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------- #
# Standardisation and supplier resolution
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ColumnMap:
    supplier: str
    amount: str
    date: str | None
    category: str | None


def _parse_amount(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    has_comma, has_dot = t.str.contains(",").any(), t.str.contains(r"\.").any()
    t = t.str.replace(",", ".", regex=False) if has_comma and not has_dot else t.str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce")


def standardise(raw: pd.DataFrame, cols: ColumnMap) -> pd.DataFrame:
    """Map an arbitrary table to the internal schema. Truth columns (demo only) are kept."""
    out = pd.DataFrame(index=raw.index)
    out["supplier_raw"] = raw[cols.supplier].astype(object).where(raw[cols.supplier].notna(), None)
    out["supplier_raw"] = out["supplier_raw"].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else None)
    out["amount"] = _parse_amount(raw[cols.amount])
    out["date"] = pd.to_datetime(raw[cols.date], errors="coerce", format="mixed") if cols.date else pd.NaT
    if cols.category:
        cat = raw[cols.category].astype(object).map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else None)
    else:
        cat = pd.Series(None, index=raw.index, dtype=object)
    out["cpv_raw"] = cat
    for extra in ("true_supplier_id", "_inj_duplicate", "_inj_outlier"):
        if extra in raw.columns:
            out[extra] = raw[extra]
    return out.reset_index(drop=True)


def normalise_name(name: str | None) -> str:
    """Casefold, strip diacritics and punctuation, drop legal-form tokens at either end."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    tokens = re.sub(r"[^\w\s]", " ", s).split()
    core = list(tokens)
    while len(core) > 1 and core[-1] in LEGAL_FORMS:
        core.pop()
    while len(core) > 1 and core[0] in LEGAL_FORMS:
        core.pop(0)
    return " ".join(core)


@st.cache_data(show_spinner="Comparing supplier names...")
def linkage_matrix(keys: tuple[str, ...]) -> np.ndarray | None:
    """Average-linkage (UPGMA) tree over 1 - token_sort_ratio/100 for all unique keys."""
    if len(keys) < 2:
        return None
    sim = process.cdist(keys, keys, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1)
    dist = 100.0 - sim.astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method="average")


def cut_tree(Z: np.ndarray | None, n_keys: int, threshold: int) -> np.ndarray:
    """Cluster labels such that average pairwise similarity inside a merge is >= threshold."""
    if Z is None:
        return np.ones(n_keys, dtype=int)
    return fcluster(Z, t=100 - threshold, criterion="distance")


def _pairs(counts: pd.Series) -> int:
    c = counts.to_numpy(dtype=np.int64)
    return int((c * (c - 1) // 2).sum())


def pairwise_scores(pred: pd.Series, true: pd.Series) -> dict[str, float]:
    """Pairwise precision / recall / F1 of a clustering against known identities."""
    d = pd.DataFrame({"p": pred.to_numpy(), "t": true.to_numpy()})
    tp, pp, tt = _pairs(d.groupby(["p", "t"]).size()), _pairs(d.groupby("p").size()), _pairs(d.groupby("t").size())
    precision = tp / pp if pp else 1.0
    recall = tp / tt if tt else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def resolve_suppliers(df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Attach `key`, `cluster` and canonical `supplier` columns."""
    out = df.copy()
    out["key"] = out["supplier_raw"].map(normalise_name)
    keys = sorted(out.loc[out["key"] != "", "key"].unique())
    if len(keys) > MAX_UNIQUE_KEYS:
        st.error(f"{len(keys):,} distinct supplier names is above the {MAX_UNIQUE_KEYS:,} limit of this "
                 "in-memory version. Filter the file to a smaller scope and try again.")
        st.stop()
    Z = linkage_matrix(tuple(keys))
    labels = cut_tree(Z, len(keys), threshold)
    out["cluster"] = out["key"].map(dict(zip(keys, labels))).fillna(-1).astype(int)

    # Canonical name = most frequent original spelling inside the cluster.
    canon = (out[out["cluster"] >= 0].groupby(["cluster", "supplier_raw"]).size().reset_index(name="n")
             .sort_values(["cluster", "n", "supplier_raw"], ascending=[True, False, True])
             .drop_duplicates("cluster").set_index("cluster")["supplier_raw"])
    out["supplier"] = out["cluster"].map(canon).fillna("(missing supplier)")
    return out, Z, keys


def threshold_sweep(df: pd.DataFrame, Z: np.ndarray | None, keys: list[str]) -> pd.DataFrame:
    """Evaluate matching quality over a grid of thresholds (demo data only)."""
    names = df.loc[df["key"] != "", ["key", "supplier_raw", "true_supplier_id"]].drop_duplicates(["supplier_raw"])
    rows = []
    for t in range(60, 100, 2):
        mapping = dict(zip(keys, cut_tree(Z, len(keys), t)))
        rows.append({"threshold": t, **pairwise_scores(names["key"].map(mapping), names["true_supplier_id"])})
    return pd.DataFrame(rows)


def review_table(df: pd.DataFrame) -> pd.DataFrame:
    """Merged clusters ranked by their weakest internal link, i.e. what a human should check first."""
    rows = []
    for _, g in df[df["cluster"] >= 0].groupby("cluster"):
        keys = g["key"].unique()
        if len(keys) < 2:
            continue
        sim = process.cdist(keys, keys, scorer=fuzz.token_sort_ratio)
        rows.append({
            "Canonical supplier": g["supplier"].iat[0],
            "Weakest link": float(sim[np.triu_indices(len(keys), 1)].min()),
            "Spellings": g["supplier_raw"].nunique(),
            "Rows": len(g),
            "Spend": g["amount"].where(g["amount"] > 0).sum(),
            "Examples": " | ".join(g["supplier_raw"].value_counts().index[:4]),
        })
    return pd.DataFrame(rows).sort_values("Weakest link").reset_index(drop=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Data quality checks
# --------------------------------------------------------------------------- #

def _robust_z(x: pd.Series) -> pd.Series:
    """Modified z-score: 0.6745 * (x - median) / MAD."""
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    valid_cpv = out["cpv_raw"].fillna("").str.match(r"^\d{8}(-\d)?$")
    out["division"] = np.where(valid_cpv, out["cpv_raw"].fillna("").str[:2], None)
    out["division_name"] = out["division"].map(lambda d: CPV_DIVISIONS.get(d, f"CPV {d}") if d else "Uncategorised")

    out["f_missing_supplier"] = out["supplier_raw"].isna()
    out["f_missing_amount"] = out["amount"].isna()
    out["f_missing_date"] = out["date"].isna()
    out["f_missing_category"] = out["cpv_raw"].isna()
    out["f_bad_category"] = out["cpv_raw"].notna() & ~valid_cpv
    out["f_nonpositive"] = out["amount"] <= 0
    out["f_variant"] = out["supplier_raw"].notna() & (out["supplier_raw"] != out["supplier"])

    # Duplicates are looked for on the *resolved* supplier, so re-typed names still match.
    complete = out["amount"].notna() & out["date"].notna() & (out["cluster"] >= 0)
    out["f_duplicate"] = complete & out.duplicated(["cluster", "date", "amount", "cpv_raw"], keep="first")
    out["f_duplicate_rawname"] = complete & out.duplicated(["supplier_raw", "date", "amount", "cpv_raw"], keep="first")

    # Outliers: modified z-score on log10(amount) within CPV division (global if the group is small).
    log_amt = np.log10(out["amount"].where(out["amount"] > 0))
    grp = out["division"].fillna("none")
    grp = grp.where(grp.groupby(grp).transform("size") >= 30, "all")
    out["z"] = log_amt.groupby(grp).transform(_robust_z)
    out["f_outlier"] = out["z"].abs() > OUTLIER_CUTOFF
    return out


@dataclass(frozen=True)
class Issue:
    label: str
    flag: str
    action: str
    needs: str | None = None  # column that must exist for the check to make sense


ISSUES = [
    Issue("Likely duplicate records", "f_duplicate",
          "Confirm in the source system and remove the repeated entries so spend is not double counted."),
    Issue("Amount far outside its category norm", "f_outlier",
          "Check the source documents; these are often unit or decimal-place errors."),
    Issue("Non-positive amount", "f_nonpositive", "Confirm whether these are credit notes or entry errors."),
    Issue("Supplier spelled in several ways", "f_variant",
          "Standardise names against a supplier master list; review the merges below first."),
    Issue("Missing category code", "f_missing_category",
          "Classify these records; uncategorised spend cannot be analysed by category.", "category"),
    Issue("Malformed category code", "f_bad_category",
          "Correct the code format (8 digits, optional check digit).", "category"),
    Issue("Missing amount", "f_missing_amount", "Recover the value from the contract or invoice."),
    Issue("Missing supplier", "f_missing_supplier", "Add the supplier before this spend is reported."),
    Issue("Missing date", "f_missing_date", "Add the award date so records can be placed in a period.", "date"),
]


def issue_table(df: pd.DataFrame) -> pd.DataFrame:
    total_spend = df["amount"].where(df["amount"] > 0).sum()
    has = {"category": df["cpv_raw"].notna().any(), "date": df["date"].notna().any()}
    rows = []
    for issue in ISSUES:
        if issue.needs and not has[issue.needs]:
            continue
        mask = df[issue.flag]
        spend = df.loc[mask, "amount"].where(df["amount"] > 0).sum()
        row_share, spend_share = mask.mean() * 100, (spend / total_spend * 100 if total_spend else 0.0)
        worst = max(row_share, spend_share)
        rows.append({
            "Issue": issue.label, "Rows": int(mask.sum()), "Share of rows": row_share,
            "Spend affected": spend, "Share of spend": spend_share,
            "Severity": "High" if worst >= 5 else "Medium" if worst >= 1 else "Low",
            "Suggested action": issue.action,
        })
    return pd.DataFrame(rows).sort_values(["Share of spend", "Share of rows"], ascending=False).reset_index(drop=True)


def dimension_scores(df: pd.DataFrame) -> dict[str, float]:
    """Row-based quality dimensions, each in [0, 100]. Dimensions without data are omitted."""
    completeness_cols = ["f_missing_supplier", "f_missing_amount"]
    if df["date"].notna().any():
        completeness_cols.append("f_missing_date")
    if df["cpv_raw"].notna().any():
        completeness_cols.append("f_missing_category")
    scores = {
        "Completeness": 100 * (1 - df[completeness_cols].mean().mean()),
        "Validity": 100 * (1 - (df["f_bad_category"] | df["f_nonpositive"]).mean()),
        "Uniqueness": 100 * (1 - df["f_duplicate"].mean()),
        "Plausibility": 100 * (1 - df["f_outlier"].mean()),
        "Name consistency": 100 * (1 - df["f_variant"].mean()),
    }
    return scores


def top10_share(df: pd.DataFrame, col: str) -> float:
    d = df[(df["amount"] > 0) & df[col].notna()]
    total = d["amount"].sum()
    return d.groupby(col)["amount"].sum().nlargest(10).sum() / total * 100 if total else 0.0


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #

def fmt_money(v: float, currency: str = "EUR") -> str:
    a = abs(v)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{currency} {v / div:,.1f}{suffix}"
    return f"{currency} {v:,.0f}"


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="simple_white", height=height, colorway=COLORWAY,
        font=dict(family="Public Sans, sans-serif", color=INK, size=13),
        margin=dict(l=8, r=8, t=48, b=8), title=dict(font=dict(size=15), x=0),
        legend=dict(title=None, orientation="h", y=-0.18),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E3E9ED")
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
        .stApp, [data-testid="stMarkdownContainer"], [data-testid="stMetric"] {font-family:'Public Sans', sans-serif;}
        h1, h2, h3 {font-family:'Newsreader', Georgia, serif !important; font-weight:600 !important; letter-spacing:-0.01em;}
        .block-container {padding-top:2.2rem; max-width:1240px;}
        [data-testid="stMetric"] {background:#FFFFFF; border:1px solid #DCE4E8; border-left:4px solid #0F6E7C;
                                  border-radius:6px; padding:0.7rem 0.95rem;}
        [data-testid="stMetricLabel"] p {font-size:0.82rem; color:#5A6B77;}
        .lede {color:#5A6B77; font-size:1.05rem; max-width:62ch; margin-top:-0.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def guess_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        for c in columns:
            if hint in c.lower():
                return c
    return None


def read_upload(file) -> pd.DataFrame:
    raw = file.getvalue()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file.")


# --------------------------------------------------------------------------- #
# Sidebar: data source and settings
# --------------------------------------------------------------------------- #

def sidebar() -> tuple[pd.DataFrame, str, int, bool, bool, bool]:
    with st.sidebar:
        st.subheader("Data")
        source = st.radio("Source", ["Demo benchmark (synthetic)", "Upload CSV"], label_visibility="collapsed")
        currency = "EUR"
        if source.startswith("Demo"):
            level = st.select_slider("Data messiness", ["Low", "Medium", "High"], value="Medium")
            raw = generate_demo_data(level)
            cols = ColumnMap("supplier_name", "contract_value", "award_date", "cpv_code")
            public = raw.loc[:, ~raw.columns.str.startswith(("true_", "_inj"))]
            st.download_button("Download this demo as CSV", public.to_csv(index=False).encode(),
                               "demo_awards.csv", "text/csv")
        else:
            file = st.file_uploader("CSV file", type=["csv"])
            if file is None:
                st.info("Upload a CSV with at least a supplier name column and an amount column.")
                st.stop()
            raw = read_upload(file)
            options = list(raw.columns)
            none = "(not available)"
            sup = st.selectbox("Supplier name", options, index=options.index(guess_column(options, ("supplier", "vendor", "winner", "contractor", "name")) or options[0]))
            amt = st.selectbox("Amount", options, index=options.index(guess_column(options, ("value", "amount", "price", "total")) or options[0]))
            dt = st.selectbox("Date", [none] + options, index=([none] + options).index(guess_column(options, ("date", "award")) or none))
            cat = st.selectbox("Category code (CPV)", [none] + options, index=([none] + options).index(guess_column(options, ("cpv", "category", "code")) or none))
            currency = st.text_input("Currency label", "EUR")
            cols = ColumnMap(sup, amt, None if dt == none else dt, None if cat == none else cat)

        st.subheader("Supplier matching")
        threshold = st.slider("Similarity threshold", 70, 98, DEFAULT_THRESHOLD,
                              help="Names are merged when their average similarity is at least this value. "
                                   "Higher is stricter: fewer false merges, more missed ones.")
        st.subheader("Analysis")
        excl_dup = st.toggle("Exclude likely duplicates", value=True)
        excl_out = st.toggle("Exclude extreme outliers", value=True)

    return standardise(raw, cols), currency, threshold, excl_dup, excl_out, source.startswith("Demo")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def tab_spend(df: pd.DataFrame, cur: str, excl_dup: bool, excl_out: bool) -> None:
    use = df["amount"] > 0
    if excl_dup:
        use &= ~df["f_duplicate"]
    if excl_out:
        use &= ~df["f_outlier"]
    d = df[use]
    if d.empty:
        st.warning("No rows with a positive amount to analyse.")
        return

    f1, f2 = st.columns([2, 1])
    divisions = f1.multiselect("Category", sorted(d["division_name"].unique()), placeholder="All categories")
    if divisions:
        d = d[d["division_name"].isin(divisions)]
    years = d["year"].dropna().astype(int)
    if not years.empty and years.min() < years.max():
        lo, hi = f2.slider("Award year", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
        d = d[d["year"].between(lo, hi)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Spend in view", fmt_money(d["amount"].sum(), cur))
    m2.metric("Suppliers (cleaned)", f"{d.loc[d['cluster'] >= 0, 'cluster'].nunique():,}")
    m3.metric("Contracts", f"{len(d):,}")

    left, right = st.columns(2)
    if d["year"].notna().any():
        top = d.groupby("division_name")["amount"].sum().nlargest(7).index
        by_year = (d.assign(cat=d["division_name"].where(d["division_name"].isin(top), "Other"))
                   .groupby(["year", "cat"], as_index=False)["amount"].sum())
        fig = px.bar(by_year, x="year", y="amount", color="cat", title=f"Spend by year and category ({cur})")
        fig.update_layout(barmode="stack", xaxis=dict(dtick=1))
        left.plotly_chart(style_fig(fig), width="stretch")
    else:
        by_cat = d.groupby("division_name", as_index=False)["amount"].sum().nlargest(10, "amount")
        left.plotly_chart(style_fig(px.bar(by_cat, x="amount", y="division_name", orientation="h",
                                           title=f"Spend by category ({cur})")), width="stretch")

    top_sup = (d[d["cluster"] >= 0].groupby("supplier", as_index=False)["amount"].sum()
               .nlargest(12, "amount").sort_values("amount"))
    fig = px.bar(top_sup, x="amount", y="supplier", orientation="h", title=f"Top suppliers, cleaned names ({cur})")
    fig.update_traces(marker_color=TEAL)
    right.plotly_chart(style_fig(fig), width="stretch")

    st.markdown("##### Supplier by year")
    if d["year"].notna().any():
        cube = (d[d["cluster"] >= 0].pivot_table(index="supplier", columns="year", values="amount",
                                                 aggfunc="sum", fill_value=0.0))
        cube.columns = [str(int(c)) for c in cube.columns]
        cube["Total"] = cube.sum(axis=1)
        cube = cube.sort_values("Total", ascending=False)
    else:
        cube = d[d["cluster"] >= 0].groupby("supplier")["amount"].sum().rename("Total").to_frame().sort_values("Total", ascending=False)
    st.dataframe(cube.reset_index(), hide_index=True, width="stretch", height=320,
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in cube.columns})
    st.download_button("Download spend cube (CSV)", cube.reset_index().to_csv(index=False).encode(),
                       "spend_cube.csv", "text/csv")


def tab_matching(df: pd.DataFrame, Z, keys, threshold: int, is_demo: bool) -> None:
    n_raw, n_clean = df["supplier_raw"].nunique(), df.loc[df["cluster"] >= 0, "cluster"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distinct spellings", f"{n_raw:,}")
    c2.metric("Suppliers after cleaning", f"{n_clean:,}", f"-{(1 - n_clean / max(n_raw, 1)) * 100:.0f}%", delta_color="off")
    raw_top, clean_top = top10_share(df, "supplier_raw"), top10_share(df, "supplier")
    c3.metric("Top-10 share, raw names", f"{raw_top:.1f}%")
    c4.metric("Top-10 share, cleaned", f"{clean_top:.1f}%", f"{clean_top - raw_top:+.1f} pts", delta_color="off")
    st.caption("Fragmented names make suppliers look smaller than they are. Consolidation shows the real concentration.")

    if is_demo:
        st.markdown("##### How good is the matching?")
        names = df.loc[df["key"] != "", ["key", "supplier_raw", "true_supplier_id"]].drop_duplicates("supplier_raw")
        now = pairwise_scores(df.loc[names.index, "cluster"], names["true_supplier_id"])
        s1, s2, s3 = st.columns(3)
        s1.metric("Precision", f"{now['precision']:.3f}", help="Of the name pairs we merged, the share that truly belong together.")
        s2.metric("Recall", f"{now['recall']:.3f}", help="Of the name pairs that truly belong together, the share we merged.")
        s3.metric("F1", f"{now['f1']:.3f}")
        sweep = threshold_sweep(df, Z, keys).melt("threshold", var_name="metric", value_name="score")
        fig = px.line(sweep, x="threshold", y="score", color="metric", markers=True,
                      title="Trade-off between false merges (precision) and missed merges (recall)")
        fig.add_vline(x=threshold, line_dash="dot", line_color=SLATE)
        fig.update_yaxes(range=[0, 1.02])
        st.plotly_chart(style_fig(fig, 340), width="stretch")
        st.caption("Scores are computed on distinct spellings against the simulator's hidden supplier identities.")

    st.markdown("##### Merges to review first")
    st.caption("Clusters sorted by their weakest internal similarity. A human check here is cheapest and most valuable.")
    review = review_table(df)
    if review.empty:
        st.info("No suppliers were merged at this threshold.")
    else:
        st.dataframe(review.head(60), hide_index=True, width="stretch", height=380, column_config={
            "Weakest link": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Spend": st.column_config.NumberColumn(format="%.0f"),
            "Examples": st.column_config.TextColumn(width="large"),
        })


def tab_quality(df: pd.DataFrame, cur: str, is_demo: bool) -> None:
    dims = dimension_scores(df)
    overall = float(np.mean(list(dims.values())))
    left, right = st.columns([1, 2])
    left.metric("Overall quality score", f"{overall:.0f} / 100")
    left.caption("Simple average of five row-based dimensions. The table on the right is spend-weighted.")
    fig = go.Figure(go.Bar(x=list(dims.values()), y=list(dims.keys()), orientation="h",
                           marker_color=[TEAL if v >= 95 else AMBER if v >= 85 else BRICK for v in dims.values()],
                           text=[f"{v:.1f}" for v in dims.values()], textposition="outside"))
    fig.update_xaxes(range=[0, 108], title=None)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Score by dimension")
    right.plotly_chart(style_fig(fig, 300), width="stretch")

    issues = issue_table(df)
    st.markdown("##### What is wrong, and how much spend it touches")
    st.dataframe(issues, hide_index=True, width="stretch", column_config={
        "Rows": st.column_config.NumberColumn(format="%d"),
        "Share of rows": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        "Spend affected": st.column_config.NumberColumn(format="%.0f", help=f"Sum of positive amounts, {cur}"),
        "Share of spend": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        "Suggested action": st.column_config.TextColumn(width="large"),
    })
    st.caption("Rows can carry more than one issue, so shares do not add up to 100%.")

    st.markdown("##### Fix first")
    top = issues[(issues["Rows"] > 0)].head(3)
    if top.empty:
        st.success("No issues found.")
    for i, r in enumerate(top.itertuples(index=False), start=1):
        st.markdown(f"{i}. **{r[0]}**: {r[1]:,} rows ({r[2]:.1f}%), touching {r[4]:.1f}% of spend "
                    f"({fmt_money(r[3], cur)}). {r[6]}")

    dup_clean, dup_raw = int(df["f_duplicate"].sum()), int(df["f_duplicate_rawname"].sum())
    st.info(f"Exact-name matching finds {dup_raw:,} duplicate records. After supplier consolidation it finds "
            f"{dup_clean:,}. Duplicates hide behind inconsistent supplier names.")

    if is_demo and "_inj_duplicate" in df:
        with st.expander("Detection check against the defects injected into the demo data"):
            rows = []
            for label, flag, truth in (("Duplicates", "f_duplicate", "_inj_duplicate"), ("Outliers", "f_outlier", "_inj_outlier")):
                tp = int((df[flag] & df[truth].astype(bool)).sum())
                rows.append({"Defect": label, "Injected": int(df[truth].sum()), "Flagged": int(df[flag].sum()),
                             "Precision": tp / max(int(df[flag].sum()), 1), "Recall": tp / max(int(df[truth].sum()), 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Precision": st.column_config.NumberColumn(format="%.2f"),
                "Recall": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Outlier precision is lower by design: the log-normal amounts have genuine extreme values "
                       "that the injection did not create. Duplicate recall is below 1 when a copy lost its amount or category.")


def tab_method() -> None:
    st.markdown(
        """
        **Supplier resolution.** Names are casefolded, stripped of diacritics, punctuation and legal-form tokens
        (Oy, Ab, Ltd, GmbH, ...). Similarity is the token-sort ratio, which ignores word order. Distinct names are
        grouped with average-linkage hierarchical clustering; cutting the tree at *1 - threshold/100* keeps every merge
        above the chosen average similarity. The canonical name is the most frequent spelling in the cluster.

        **Quality checks.** Completeness, validity, uniqueness, plausibility and naming consistency. Duplicates are
        searched on the resolved supplier, date, amount and category. Outliers use the modified z-score
        (median and MAD) of log10(amount) within each CPV division, flagged above 3.5.

        **Evaluation.** The demo data is simulated with hidden supplier identities, so matching is scored with pairwise
        precision, recall and F1, and the threshold sweep shows the trade-off explicitly.

        **Limits.** Matching is lexical, so unrelated trading names of the same company will not merge. Similar
        names of different companies can. Memory is quadratic in distinct names, hence the 4,000-name guardrail.
        Full details are in the README.
        """
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    inject_css()
    st.title("Spend Data Quality Workbench")
    st.markdown('<p class="lede">Clean up supplier names, measure what is wrong with the data, and see how much '
                "spend each problem touches.</p>", unsafe_allow_html=True)

    std, cur, threshold, excl_dup, excl_out, is_demo = sidebar()
    resolved, Z, keys = resolve_suppliers(std, threshold)
    df = add_quality_flags(resolved)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(df):,}")
    k2.metric("Total spend", fmt_money(df["amount"].where(df["amount"] > 0).sum(), cur))
    k3.metric("Suppliers after cleaning", f"{df.loc[df['cluster'] >= 0, 'cluster'].nunique():,}")
    k4.metric("Quality score", f"{np.mean(list(dimension_scores(df).values())):.0f} / 100")

    if is_demo:
        st.caption("Demo mode: this dataset is synthetic. It imitates public procurement awards and includes "
                   "injected defects with known ground truth.")

    t1, t2, t3, t4 = st.tabs(["Spend cube", "Supplier matching", "Data quality report", "Method"])
    with t1:
        tab_spend(df, cur, excl_dup, excl_out)
    with t2:
        tab_matching(df, Z, keys, threshold, is_demo)
    with t3:
        tab_quality(df, cur, is_demo)
    with t4:
        tab_method()


if __name__ == "__main__":
    main()
