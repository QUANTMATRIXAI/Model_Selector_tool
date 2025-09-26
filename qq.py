from pathlib import Path
import html

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE SETUP (QuantMatrix Style)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantMatrix • Model & Optimizer Suite", layout="wide")

# Inject Inter + brand styles inspired by the QuantMatrix style guide
st.markdown(
    r"""
    <style>
      /* Font */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }

      /* Brand tokens */
      :root{
        --qm-primary: #FFBD59; /* Yellow */
        --qm-primary-700: #E6AA50;
        --qm-green:   #41C185; /* Secondary */
        --qm-blue:    #458EE2; /* Secondary */
        --qm-bg:      #F5F5F5; /* Background */
        --qm-white:   #FFFFFF;
        --qm-text:    #333333; /* Body */
        --qm-muted:   #666666; /* Muted */
        --qm-soft:    #FFF2DF; /* Light Yellow */
        --qm-border:  #E8E8E8;
      }

      /* Page background */
      .main, .stApp, .block-container { background: var(--qm-bg) !important; }

      /* Headings */
      .qm-hero { font-size: 40px; font-weight: 700; letter-spacing: -0.3px; color: var(--qm-text); }
      .qm-sub   { font-size: 16px; color: var(--qm-muted); margin-top: -6px; }

      /* Cards */
      .qm-card {
        background: var(--qm-white);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        border: 1px solid var(--qm-border);
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        position: relative;
        transition: transform .12s ease, box-shadow .12s ease;
        min-height: 170px;
      }
      .qm-card:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(0,0,0,.08); }
      .qm-card .qm-eyebrow { font-size: 12px; font-weight: 600; color: var(--qm-muted); letter-spacing: .08em; text-transform: uppercase; }
      .qm-card .qm-title { font-size: 22px; font-weight: 700; margin: 2px 0 6px; color: var(--qm-text); }
      .qm-card .qm-copy  { font-size: 14px; color: var(--qm-muted); }
      .qm-chip { font-size: 12px; padding: 4px 8px; border-radius: 999px; background: var(--qm-soft); border: 1px solid var(--qm-primary); color: #8c5b00; font-weight: 600; }

      /* Accent bars per card */
      .qm-accent-yellow { border-top: 4px solid var(--qm-primary); }
      .qm-accent-green  { border-top: 4px solid var(--qm-green); }
      .qm-accent-blue   { border-top: 4px solid var(--qm-blue); }

      /* Buttons */
      .stButton>button {
        border-radius: 12px; border: 0; padding: 10px 14px; font-weight: 700;
        background: var(--qm-primary); color: #222; box-shadow: 0 4px 12px rgba(255,189,89,0.35);
      }
      .stButton>button:hover { filter: brightness(0.98); box-shadow: 0 6px 16px rgba(255,189,89,0.45); }
      .stButton>button:active { transform: translateY(1px); }

      /* Secondary button look */
      .qm-btn-secondary .stButton>button{ background: var(--qm-blue) !important; color: white !important; box-shadow: 0 4px 12px rgba(69,142,226,0.35); }

      /* Panels */
      .qm-panel { background: var(--qm-white); border: 1px solid var(--qm-border); border-radius: 16px; padding: 16px; }
      .qm-panel h3 { margin-top: 4px; }

      /* Tiny label */
      .qm-label { font-size: 12px; color: var(--qm-muted); }

      /* Journeys timeline */
      .journey-shell {
        background: var(--qm-white);
        border: 1px solid var(--qm-border);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 5px 16px rgba(0,0,0,0.04);
        margin-top: 12px;
      }
      .journey-header {
        display: flex;
        flex-wrap: wrap;
        gap: 12px 18px;
        margin-bottom: 14px;
        align-items: center;
      }
      .journey-pill {
        background: var(--qm-soft);
        color: #8c5b00;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
      }
      .journey-timeline {
        margin-top: 6px;
      }
      .journey-step {
        display: grid;
        grid-template-columns: 58px 1fr;
        column-gap: 16px;
        padding: 14px 0;
        border-bottom: 1px solid var(--qm-border);
      }
      .journey-step:last-child { border-bottom: none; }
      .journey-seq {
        font-size: 18px;
        font-weight: 700;
        color: var(--qm-primary);
      }
      .journey-main {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .journey-event {
        font-size: 15px;
        font-weight: 600;
        color: var(--qm-text);
      }
      .journey-page {
        font-size: 13px;
        color: var(--qm-muted);
        word-break: break-word;
      }
      .journey-meta {
        font-size: 12px;
        color: var(--qm-muted);
        display: flex;
        flex-wrap: wrap;
        gap: 10px 16px;
      }
      .journey-meta span {
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }
      .journey-meta strong { color: var(--qm-text); }
      .journey-empty {
        font-size: 14px;
        color: var(--qm-muted);
        padding: 14px 0;
      }

      /* Divider */
      hr.qm { border: 0; border-top: 1px solid var(--qm-border); margin: 10px 0 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# NAV / STATE
# ──────────────────────────────────────────────────────────────────────────────
SECTIONS = {
    "home": "Home",
    "consolidator": "Model Consolidator",
    "insights": "Insights",
    "journeys": "Journeys",
    "optimizer_config": "Optimizer Configurer",
    "optimizer": "Marketing Inputs",
    "contextual": "Contextual Inputs",
}

if "section" not in st.session_state:
    st.session_state.section = "home"

JOURNEYS_FILE = Path(__file__).with_name("TW Attribution_Logic Verification_Table (1).csv")


def _clean_text_series(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    for token in (" ", " ", "﻿"):
        s = s.str.replace(token, " ", regex=False)
    s = s.str.replace("Â", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "NULL": np.nan, "null": np.nan, "None": np.nan, "NONE": np.nan, "NaT": np.nan})
    return s


def _first_valid_value(series: pd.Series):
    for val in series:
        if pd.notna(val) and str(val).strip():
            return val
    return None


def _format_timedelta(td: pd.Timedelta | None) -> str:
    if td is None or pd.isna(td):
        return "--"
    total_seconds = int(td.total_seconds())
    if total_seconds <= 0:
        return "0s"
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds and not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts) if parts else "0s"


@st.cache_data(show_spinner=False)

def load_journey_data(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame(columns=[
            "journey_name", "event_sequence", "event_timestamp", "event_type",
            "page_location", "purchase_timestamp", "source", "ad_name",
            "event_time", "purchase_time", "event_type_norm"
        ])

    df = pd.read_csv(path_obj)
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "JourneyName": "journey_name",
        "event_sequence": "event_sequence",
        "event_timestamp": "event_timestamp",
        "event_type": "event_type",
        "page_location": "page_location",
        "purchase_timestamp": "purchase_timestamp",
        "source": "source",
        "ad_name": "ad_name",
    }
    df = df.rename(columns=rename_map)

    text_cols = ["journey_name", "event_timestamp", "event_type", "page_location", "purchase_timestamp", "source", "ad_name"]
    for col in text_cols:
        if col in df.columns:
            df[col] = _clean_text_series(df[col])

    if "event_sequence" in df.columns:
        df["event_sequence"] = pd.to_numeric(df["event_sequence"], errors="coerce")
    else:
        df["event_sequence"] = np.nan

    df = df.dropna(subset=["journey_name", "event_sequence"]).copy()
    df["journey_name"] = df["journey_name"].astype(str)
    df["event_sequence"] = df["event_sequence"].astype(int)

    if "event_type" in df.columns:
        df["event_type_norm"] = df["event_type"].fillna("").str.strip().str.lower()
    else:
        df["event_type_norm"] = ""

    if "event_timestamp" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    else:
        df["event_time"] = pd.NaT

    if "purchase_timestamp" in df.columns:
        df["purchase_time"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
    else:
        df["purchase_time"] = pd.NaT

    if "ad_name" in df.columns:
        has_valid_ad = df.groupby("journey_name")["ad_name"].transform(lambda s: s.notna().any())
        df = df[has_valid_ad]
    else:
        return df.iloc[0:0]

    df = df.sort_values(["journey_name", "event_sequence", "event_time"]).reset_index(drop=True)

    purchase_seq = (
        df[df["event_type_norm"] == "purchase"]
        .groupby("journey_name")["event_sequence"]
        .min()
    )
    if not purchase_seq.empty:
        df = df.merge(purchase_seq.rename("purchase_limit"), on="journey_name", how="left")
        df = df[(df["purchase_limit"].isna()) | (df["event_sequence"] <= df["purchase_limit"])].copy()
        df = df.drop(columns=["purchase_limit"])

    return df





def go(to: str):
    # Persist target section and force a rerun so the router picks it up immediately
    st.session_state.section = to
    try:
        st.rerun()
    except Exception:
        # Fallback for older Streamlit versions
        st.experimental_rerun()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MODEL CONSOLIDATOR (Winner or MAPE-weighted)
# ──────────────────────────────────────────────────────────────────────────────
# Core logic from earlier messages, compact and robust

def _normalize_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _resolve_col(df: pd.DataFrame, candidates) -> str | None:
    cand_norm = [_normalize_name(c) for c in candidates]
    mapping = {_normalize_name(c): c for c in df.columns}
    for c in mapping:
        if c in cand_norm:
            return mapping[c]
    for c in mapping:
        for want in cand_norm:
            if want in c:
                return mapping[c]
    return None


def detect_grouping_keys(df: pd.DataFrame):
    if "Model" not in df.columns:
        raise ValueError("Expected a 'Model' column.")
    model_idx = list(df.columns).index("Model")
    if model_idx == 0:
        raise ValueError("'Model' is the first column; need combination columns to the left.")
    return list(df.columns[:model_idx])


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            coerced = pd.to_numeric(out[c], errors="coerce")
            if coerced.notna().sum() >= max(3, int(0.1 * len(out))):
                out[c] = coerced
    return out


def apply_optional_filters(df: pd.DataFrame,
    use_r2_test: bool, r2_test_min: float,
    use_mape_test: bool, mape_test_max: float,
    use_neg_price_elast: bool,
    use_elast_range: bool, elast_lo: float, elast_hi: float,
    use_d1_pos: bool,
                           use_rpi_neg: bool) -> pd.DataFrame:
    f = df.copy()

    r2_test_col   = _resolve_col(f, ["R2 Test", "r2_test", "R2_Test"])
    mape_test_col = _resolve_col(f, ["MAPE Test", "mape_test", "MAPE_Test"])
    self_el_col   = _resolve_col(f, ["SelfElasticity", "self_elasticity", "Price_Elasticity"])

    if use_r2_test and r2_test_col in f.columns:
        f = f[pd.to_numeric(f[r2_test_col], errors="coerce") >= r2_test_min]
    if use_mape_test and mape_test_col in f.columns:
        f = f[pd.to_numeric(f[mape_test_col], errors="coerce") <= mape_test_max]

    if self_el_col in f.columns:
        if use_neg_price_elast:
            f = f[pd.to_numeric(f[self_el_col], errors="coerce") < 0]
        if use_elast_range:
            se = pd.to_numeric(f[self_el_col], errors="coerce")
            f = f[(se >= elast_lo) & (se <= elast_hi)]

    if use_d1_pos and "Beta_D1" in f.columns:
        f = f[pd.to_numeric(f["Beta_D1"], errors="coerce") > 0]

    if use_rpi_neg:
        rpi_cols = [c for c in f.columns if c.startswith("Beta_") and c.endswith("_RPI")]
        for c in rpi_cols:
            f = f[pd.to_numeric(f[c], errors="coerce") <= 0]

    if self_el_col in f.columns:
        f = f[f[self_el_col].notna()]

    return f


def average_across_folds(df: pd.DataFrame, grouping_keys: list[str]):
    base_avg_cols = [
        "R2 Train","R2 Test","MAPE Train","MAPE Test","MSE Train","MSE Test",
        "SelfElasticity","CSF","MCV","PPU_at_Elasticity","B0 (Original)","Consumer_Surplus_Ratio"
    ]
    avg_cols = [c for c in base_avg_cols if c in df.columns]
    beta_cols = [c for c in df.columns if c.startswith("Beta_")]
    avg_cols.extend(beta_cols)

    exclude = set(grouping_keys + ["Model","Fold","ElasticityFlag","Contribution"])
    feature_cols = []
    for c in df.columns:
        if c in avg_cols or c in exclude or c.startswith("Beta_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    avg_cols.extend(feature_cols)

    grp = grouping_keys + ["Model"]
    averaged = df.groupby(grp, dropna=False)[avg_cols].mean(numeric_only=True).reset_index()

    # also carry Mean_<feature> for equation display
    for c in feature_cols:
            averaged[f"Mean_{c}"] = averaged[c]

    return averaged, beta_cols, feature_cols


def build_weighted_ensemble(group_df: pd.DataFrame, weight_col: str | None):
    res = {}

    if weight_col is None or weight_col not in group_df.columns:
        w = np.ones(len(group_df), dtype=float)
    else:
        m = pd.to_numeric(group_df[weight_col], errors="coerce")
        if m.isna().all():
            w = np.ones(len(group_df), dtype=float)
        else:
            best = m.min()
            w = np.exp(-0.5 * (m - best))
    wsum = w.sum()
    if wsum == 0 or np.isnan(wsum):
        w = np.ones(len(group_df), dtype=float)
        wsum = w.sum()

    def wavg(col):
        return np.average(group_df[col], weights=w) if col in group_df.columns else np.nan

    for col, outn in [
        ("SelfElasticity","Weighted_Elasticity"),
        ("CSF","Weighted_CSF"),
        ("MCV","Weighted_MCV"),
        ("Consumer_Surplus_Ratio","Weighted_CSR"),
        ("MAPE Test","Weighted_MAPE_Test"),
        ("MAPE Train","Weighted_MAPE_Train"),
        ("R2 Test","Weighted_R2_Test"),
        ("R2 Train","Weighted_R2_Train"),
        ("B0 (Original)","Weighted_B0"),
    ]:
        if col in group_df.columns:
            res[outn] = wavg(col)

    for c in [c for c in group_df.columns if c.startswith("Beta_")]:
        if group_df[c].notna().any():
            res[f"Weighted_{c}"] = np.average(group_df[c].fillna(0), weights=w)

    for c in [c for c in group_df.columns if c.startswith("Mean_")]:
        res[c] = group_df[c].mean()

    res["Models_Used"] = int(len(group_df))
    best_idx = group_df[weight_col].astype(float).idxmin() if weight_col and weight_col in group_df.columns else group_df.index[0]
    res["Best_Model"] = str(group_df.loc[best_idx, "Model"]) if "Model" in group_df.columns else ""
    res["Best_MAPE"] = float(group_df[weight_col].min()) if weight_col and weight_col in group_df.columns else np.nan
    if "R2 Test" in group_df.columns:
        res["Avg_R2_Test"] = float(group_df["R2 Test"].mean())
    res["Weight_Concentration"] = float((w / wsum).max())

    if "Model" in group_df.columns:
        mixed = group_df["Model"].astype(str).str.startswith("MixedLM").values
        res["Mixed_Effects_Weight"] = float((w[mixed] / wsum).sum()) if mixed.any() else 0.0

    has_b0 = ("Weighted_B0" in res) and pd.notna(res["Weighted_B0"])
    beta_feats = {k.replace("Weighted_Beta_", "") for k in res if k.startswith("Weighted_Beta_")}
    mean_feats = {k.replace("Mean_", "") for k in res if k.startswith("Mean_")}
    res["Equation_Complete"] = bool(has_b0 and beta_feats.issubset(mean_feats) and len(beta_feats) > 0)
    if res["Equation_Complete"]:
        yhat = res["Weighted_B0"]
        for feat in beta_feats:
            yhat += res[f"Weighted_Beta_{feat}"] * res[f"Mean_{feat}"]
        res["Y_Pred_at_Mean"] = float(yhat)

    return res


def pick_best_single(group_df: pd.DataFrame, mape_col: str | None, r2_col: str | None) -> dict:
    if mape_col and mape_col in group_df.columns and group_df[mape_col].notna().any():
        idx = group_df[mape_col].astype(float).idxmin()
    elif r2_col and r2_col in group_df.columns and group_df[r2_col].notna().any():
        idx = group_df[r2_col].astype(float).idxmax()
    else:
        idx = group_df.index[0]

    row = group_df.loc[idx].to_dict()
    out = {f"Selected_{k}": v for k, v in row.items() if k not in ["Model"]}
    out["Selected_Model"] = str(group_df.loc[idx, "Model"]) if "Model" in group_df.columns else ""
    return out


def extract_candidate_variables(raw_df: pd.DataFrame | None, consolidated_df: pd.DataFrame | None):
    """Return candidate feature names derived from Beta_*/Mean_* (raw) and Weighted_Beta_* (consolidated)."""
    beta_vars, mean_vars, weighted_beta_vars = set(), set(), set()
    if raw_df is not None:
        for c in raw_df.columns:
            if isinstance(c, str) and c.startswith("Beta_"):
                beta_vars.add(c.replace("Beta_", "", 1))
            if isinstance(c, str) and c.startswith("Mean_"):
                mean_vars.add(c.replace("Mean_", "", 1))
    if consolidated_df is not None:
        for c in consolidated_df.columns:
            if isinstance(c, str) and c.startswith("Weighted_Beta_"):
                weighted_beta_vars.add(c.replace("Weighted_Beta_", "", 1))
            if isinstance(c, str) and c.startswith("Mean_"):
                mean_vars.add(c.replace("Mean_", "", 1))
    all_features = sorted(beta_vars | weighted_beta_vars | mean_vars)
    return {
        "beta_vars": sorted(beta_vars),
        "weighted_beta_vars": sorted(weighted_beta_vars),
        "mean_vars": sorted(mean_vars),
        "all_features": all_features,
    }


def suggest_media_variables(names: list[str]) -> list[str]:
    """Heuristic suggestion for media spend variables."""
    patterns = [
        "spend", "impression", "reach", "click", "ctr", "cpc", "cpa",
        "meta", "facebook", "google", "ad", "ads", "youtube", "search",
        "display", "campaign", "budget", "tv", "radio", "ooh", "grps", "trp"
    ]
    out = [n for n in names if any(p in n.lower() for p in patterns)]
    # de-dup & stable order
    seen, dedup = set(), []
    for n in out:
        if n not in seen:
            dedup.append(n); seen.add(n)
    return dedup


def parse_unit_cost_file(df_like: pd.DataFrame) -> dict:
    """Parse an uploaded unit-cost mapping DataFrame into {variable: unit_cost}.
    Accepts column names like ['variable'|'var'|'feature'] and ['unit_cost'|'cost'|'cpc'|'cpm'].
    """
    if df_like is None or df_like.empty:
        return {}
    cols = {c.lower().strip(): c for c in df_like.columns}
    var_col = None
    cost_col = None
    for k in ["variable", "var", "feature", "name"]: 
        if k in cols: var_col = cols[k]; break
    for k in ["unit_cost", "cost", "cpc", "cpm"]: 
        if k in cols: cost_col = cols[k]; break
    if var_col is None or cost_col is None:
        # fallback: first two columns
        if len(df_like.columns) >= 2:
            var_col, cost_col = df_like.columns[:2]
        else:
            return {}
    mapping = {}
    for _, r in df_like[[var_col, cost_col]].dropna().iterrows():
        try:
            mapping[str(r[var_col])] = float(r[cost_col])
        except Exception:
            continue
    return mapping


def show_consolidator():
    st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
    st.markdown("<span class='qm-eyebrow'>Section 1</span>", unsafe_allow_html=True)
    st.markdown("<h2>Model Consolidator</h2>", unsafe_allow_html=True)
    st.caption("Upload → Optional filters → Winner or MAPE-weighted → 1 row per combination. Saved to session state for later pages.")

    uploaded = st.file_uploader("Upload MODEL_RESULTS (CSV or Parquet). Must contain a 'Model' column.", type=["csv","parquet"], key="u_model_results")

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_parquet(uploaded)
        df = coerce_numeric(df)
        try:
            auto_keys = detect_grouping_keys(df)
        except Exception as e:
            st.error(str(e))
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.caption(f"Detected combination keys (left of 'Model'): {auto_keys}")
        keys = st.multiselect("Override grouping keys (optional).", options=list(df.columns), default=auto_keys, key="keys_override")
        grouping_keys = keys if keys else auto_keys
        for k in grouping_keys:
            df[k] = df[k].astype(str).str.strip().fillna("")

        with st.expander("Optional Filters", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                use_r2_test   = st.checkbox("R² Test ≥", value=True, key="f_use_r2")
                r2_test_min   = st.number_input("Min R² Test", value=0.5, step=0.05, format="%.2f", key="f_r2_min")
                use_mape_test = st.checkbox("MAPE Test ≤", value=True, key="f_use_mape")
                mape_test_max = st.number_input("Max MAPE Test (%)", value=30.0, step=1.0, format="%.1f", key="f_mape_max")
            with c2:
                use_neg_price = st.checkbox("Price Elasticity < 0", value=True, key="f_neg_price")
                use_el_range  = st.checkbox("Bound Self-Elasticity", value=True, key="f_el_range")
                elast_lo, elast_hi = st.slider("Elasticity range", -10.0, 0.0, (-5.0, -0.1), key="f_el_bounds")
            with c3:
                use_d1_pos = st.checkbox("Beta_D1 > 0 (if present)", value=("Beta_D1" in df.columns), key="f_d1_pos")
                use_rpi_neg = st.checkbox("RPI betas ≤ 0 (if present)", value=True, key="f_rpi_neg")

        mode = st.radio("Output mode (always 1 row per combination):",
                        ["Best single model (winner)", "Weighted ensemble (MAPE-based)"], index=0, key="mode_sel")

        if st.button("🚀 Consolidate Models", type="primary", key="btn_consolidate"):
            filtered = apply_optional_filters(
                df,
                st.session_state.f_use_r2, st.session_state.f_r2_min,
                st.session_state.f_use_mape, st.session_state.f_mape_max,
                st.session_state.f_neg_price,
                st.session_state.f_el_range, st.session_state.f_el_bounds[0], st.session_state.f_el_bounds[1],
                st.session_state.f_d1_pos,
                st.session_state.f_rpi_neg,
            )
            if filtered.empty:
                st.warning("No rows after filtering. Relax or disable filters.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            averaged, beta_cols, feat_cols = average_across_folds(filtered, grouping_keys)
            mape_test_col = _resolve_col(averaged, ["MAPE Test", "mape_test", "MAPE_Test"])
            r2_test_col   = _resolve_col(averaged, ["R2 Test", "r2_test", "R2_Test"])

            results = []
            for combo_vals, g in averaged.groupby(grouping_keys, dropna=False):
                if st.session_state.mode_sel.startswith("Best"):
                    out = pick_best_single(g, mape_test_col, r2_test_col)
                else:
                    out = build_weighted_ensemble(g, weight_col=mape_test_col)
                if out:
                    row = {k: v for k, v in zip(grouping_keys, combo_vals)} if isinstance(combo_vals, tuple) else {grouping_keys[0]: combo_vals}
                    row.update(out)
                    results.append(row)

            out_df = pd.DataFrame(results)
            if out_df.empty:
                st.warning("No combinations produced an output.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # Validate uniqueness: exactly one row per combo
            if not out_df.groupby(grouping_keys).size().eq(1).all():
                st.error("Internal error: duplicate rows for a combination.")

            # Save to session for the next sections
            st.session_state["saved_models"] = out_df
            st.session_state["raw_model_results"] = df
            st.session_state["candidate_vars"] = extract_candidate_variables(df, out_df)
            st.session_state["grouping_keys"] = grouping_keys

            st.success(f"✅ Consolidated {len(out_df)} combinations — saved to session as 'saved_models'.")
            st.dataframe(out_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Results",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="consolidated_models.csv",
                mime="text/csv",
            )

            st.info("Tip: View Insights next — contribution shares from the consolidated model.")
            cnav1, cnav2, cnav3 = st.columns(3)
            with cnav1:
                if st.button("Go to Insights →", key="go_insights", type="secondary"):
                    go("insights")
            with cnav2:
                if st.button("Go to Optimizer Configurer →", key="go_cfg", type="secondary"):
                    go("optimizer_config")
            with cnav3:
                if st.button("Go to Marketing Inputs →", key="go_opt_after_cons", type="secondary"):
                    go("optimizer")

    colA, colB = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"):
            go("home")
    st.markdown("</div>", unsafe_allow_html=True)


##############color pallete 

# ======== Palette + Chart Helpers (add once near top of your file) ========

def _qm_color_map(categories):
    # Brand-first palette; "Others" always neutral grey.
    base_seq = [
        "#458EE2", "#41C185", "#FFBD59", "#7C3AED",
        "#EF4444", "#059669", "#F59E0B", "#1F2937",
        "#10B981", "#F43F5E", "#6366F1", "#0EA5E9"
    ]
    cmap = {}
    i = 0
    for c in categories:
        if c == "Others":
            cmap[c] = "#9CA3AF"
        elif c not in cmap:
            cmap[c] = base_seq[i % len(base_seq)]
            i += 1
    return cmap

def _apply_qm_layout(fig, title):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, y=0.98),
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, system-ui, -apple-system, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        uniformtext_minsize=12,
        uniformtext_mode="hide",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INSIGHTS (Contribution Shares) — SIMPLE & SPACE-EFFICIENT
# ──────────────────────────────────────────────────────────────────────────────
def show_insights():
    """
    Simple Contribution Explorer:
      Effect_i = Beta_i * Mean_i
      SignedShare_i = Effect_i / sum_j Effect_j   (fallback to magnitude-norm only if denom≈0)

    Controls: Show (All/Positive/Negative), Name contains, Remove variables,
              Top-N + Others, and a toggle to view a scrollable FULL GRID
              (rows = all combinations, cols = variables) with in-table heatmap coloring.
    """
    import re
    import hashlib
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    # ---- Minimal style (Inter + brand colors)
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .qm-eyebrow { color:#6B7280; font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
      .qm-panel { padding: 12px 16px; border: 1px solid #E5E7EB; border-radius:16px; background:#FFFFFF; }
      .title-h2 { font-weight:700; margin: 0 0 8px 0; }
      .section-title { font-weight:700; font-size:16px; margin: 12px 0 6px; }
    </style>
    """, unsafe_allow_html=True)

    BRAND_GREEN = "#41C185"
    BRAND_RED   = "#EF4444"

    # ----------------- tiny helpers -----------------
    def _tof(x, default=None):
        try: return float(x)
        except Exception: return default

    def _pretty(v: str) -> str:
        if not isinstance(v, str): return str(v)
        s = re.sub(r"_meta(_impression)?$", "", v, flags=re.I)
        s = re.sub(r"_OUTCOME_.*$", "", s, flags=re.I)
        s = re.sub(r"_impression$", "", s, flags=re.I)
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s[:48] + ("…" if len(s) > 48 else "")

    def _unique_key(prefix: str, *parts) -> str:
        h = hashlib.sha1(("|".join(map(str, parts))).encode()).hexdigest()[:8]
        return f"{prefix}_{h}"

    def _dimension_cols_for(df: pd.DataFrame, preset):
        if preset: return preset
        blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used",
                            "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
        dims = [c for c in df.columns if not str(c).startswith(blacklist_prefix)]
        return dims[:3]

    def _find_pairs_from_columns(columns):
        """Return dict[var] = {'beta_col': ..., 'mean_col': ...} from the DF columns."""
        beta_pref = ["Weighted_Beta_", "Selected_Beta_", "Beta_"]
        mean_pref = ["Mean_", "Selected_Mean_"]
        pairs = {}
        # Discover variables
        for c in columns:
            if not (isinstance(c, str) and "Beta_" in c):
                continue
            var = c
            for p in beta_pref:
                if var.startswith(p):
                    var = var[len(p):]
                    break
            # choose best beta col
            beta_col = None
            for p in ("Weighted_Beta_", "Selected_Beta_", "Beta_"):
                cand = f"{p}{var}"
                if cand in columns:
                    beta_col = cand; break
            if not beta_col: 
                continue
            # pick matching mean col
            mean_col = None
            for mp in mean_pref:
                cand = f"{mp}{var}"
                if cand in columns: 
                    mean_col = cand; break
            if mean_col is None and f"Mean_{var}" in columns:
                mean_col = f"Mean_{var}"
            pairs[var] = {"beta_col": beta_col, "mean_col": mean_col}
        return pairs

    # ----------------- UI scaffolding -----------------
    st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
    st.markdown("<span class='qm-eyebrow'>Section 2</span>", unsafe_allow_html=True)
    st.markdown("<h2 class='title-h2'>Insights — Contribution Shares</h2>", unsafe_allow_html=True)

    saved_models = st.session_state.get("saved_models")
    grouping_keys = st.session_state.get("grouping_keys", [])
    if saved_models is None or saved_models.empty:
        st.warning("No consolidated models in session. Run Section 1 first.")
        if st.button("← Back to Home"):
            go("home")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Detect dimensions and pairs
    dimension_cols_all = _dimension_cols_for(saved_models, grouping_keys)
    pairs_all = _find_pairs_from_columns(saved_models.columns)

    # ----------------- Single-combination selector -----------------
    labels = [" | ".join([f"{k}:{saved_models.iloc[i][k]}" for k in dimension_cols_all]) for i in range(len(saved_models))]
    last_idx = st.session_state.get("insights_combo_idx", 0)
    combo_idx = st.selectbox(
        "Combination",
        options=list(range(len(saved_models))),
        index=min(last_idx, len(saved_models)-1),
        format_func=lambda i: labels[i],
        key="combo_select_insights"
    )
    st.session_state["insights_combo_idx"] = combo_idx
    row = saved_models.iloc[combo_idx]

    # ---------- Build base table: Variable, Beta, Mean, Effect
    parts = []
    seen = set()
    for c in row.index:
        if not (isinstance(c, str) and "Beta_" in c):
            continue
        v = c.split("Beta_")[-1]
        if v in seen:
            continue
        beta = None
        for key in (f"Weighted_Beta_{v}", f"Selected_Beta_{v}", f"Beta_{v}"):
            if key in row and pd.notna(row[key]):
                beta = _tof(row[key]); 
                if beta is not None: break
        if beta is None:
            continue
        mean = None
        for key in (f"Mean_{v}", f"Selected_Mean_{v}"):
            if key in row and pd.notna(row[key]):
                mean = _tof(row[key]); 
                if mean is not None: break
        if mean is None and v.lower() in ("intercept", "const"):
            mean = 1.0
        if mean is None:
            mean = 0.0
        parts.append({"Variable": v, "Beta": beta, "Mean": mean, "Effect": beta * mean})
        seen.add(v)

    base = pd.DataFrame(parts)
    if base.empty:
        st.info("No usable Beta × Mean pairs for this combination.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---------- Compact controls row
    col1, col2, col3, col4 = st.columns([1, 1.2, 1.2, 1])
    with col1:
        show_mode = st.selectbox("Show", ["All", "Positive", "Negative"], index=0)
    with col2:
        name_filter = st.text_input("Name contains", value="").strip()
    with col3:
        top_n = st.number_input("Top-N", min_value=1, value=10, step=1)
    with col4:
        group_others = st.checkbox("Group Others", value=True)

    with st.expander("Remove variables", expanded=False):
        exclude_vars = st.multiselect("Select variables to remove", base["Variable"].tolist(), default=[])

    # ---------- Apply filters to the SINGLE view ----------
    df = base.copy()
    if show_mode == "Positive":
        df = df[df["Effect"] > 0]
    elif show_mode == "Negative":
        df = df[df["Effect"] < 0]
    if name_filter:
        df = df[df["Variable"].str.contains(name_filter, case=False, na=False)]
    if exclude_vars:
        df = df[~df["Variable"].isin(exclude_vars)]

    # ---------- Normalize (signed; fallback to magnitude if denom≈0) ----------
    denom_signed = float(df["Effect"].sum()) if not df.empty else 0.0
    if abs(denom_signed) < 1e-12:
        st.info("Sum of effects is ~0 under current filters; showing magnitude-normalized view for readability.")
        denom_abs = float(np.abs(df["Effect"]).sum()) if not df.empty else 0.0
        if denom_abs == 0.0:
            st.info("No non-zero effects remain. Adjust filters.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        df["AbsShare"] = np.abs(df["Effect"]) / denom_abs
        df["SignedShare"] = np.sign(df["Effect"]) * df["AbsShare"]
    else:
        df["SignedShare"] = df["Effect"] / denom_signed
        df["AbsShare"]   = np.abs(df["SignedShare"])

    # ---------- Top-N + Others ----------
    df = df.sort_values("AbsShare", ascending=False).reset_index(drop=True)
    def topn_with_others(dfi: pd.DataFrame, n: int, group_rest: bool) -> pd.DataFrame:
        if dfi.empty or not group_rest or n >= len(dfi): return dfi
        top = dfi.head(int(n)).copy()
        rest = dfi.iloc[int(n):]
        if not rest.empty:
            others = pd.DataFrame([{
                "Variable": "Others",
                "Beta": np.nan, "Mean": np.nan,
                "Effect": rest["Effect"].sum(),
                "SignedShare": rest["SignedShare"].sum(),
                "AbsShare": np.abs(rest["SignedShare"].sum())
            }])
            return pd.concat([top, others], ignore_index=True)
        return top

    display = topn_with_others(df, int(top_n), group_others)

    # =================== BUSINESS-CLEAN: TABLE + SIGNED DIVERGING BAR (RAW NAMES) ===================
    if not display.empty:
        df_view = display.copy()
        df_view["Direction"] = np.where(df_view["SignedShare"] >= 0, "Positive", "Negative")
        df_view["ShareLabel"] = df_view["SignedShare"].map(lambda x: f"<b>{abs(x):.1%}</b>")
        df_view = df_view.sort_values("AbsShare", ascending=True)

        # ------- layout: table left, bar right -------
        col_table, col_bar = st.columns([1.1, 1.4])

        # TABLE (left) — keep RAW variable names
        with col_table:
            tbl = df_view.copy().sort_values("AbsShare", ascending=False)
            tbl["Share (%)"] = (tbl["AbsShare"] * 100).round(1)
            tbl = tbl[["Variable", "Beta", "Mean", "Effect", "Direction", "Share (%)"]]
            st.markdown("<div class='section-title'>Contribution Table</div>", unsafe_allow_html=True)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            dl_key = f"download_contrib_table_{combo_idx}"
            st.download_button(
                "📥 Download Contributions",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"contribution_shares_combo{combo_idx}.csv",
                mime="text/csv",
                key=dl_key
            )

        # BAR (right) — signed, negative axis, RAW names on Y
        with col_bar:
            fig = px.bar(
                df_view,
                y="Variable", x="SignedShare",
                orientation="h",
                color="Direction",
                color_discrete_map={"Positive": BRAND_GREEN, "Negative": BRAND_RED},
                text="ShareLabel",
            )
            fig.update_traces(
                textposition="outside",
                textfont=dict(size=12),
                marker=dict(line=dict(color="white", width=1.0)),
                hovertemplate="<b>%{y}</b><br>Direction: %{marker.color}<br>Share of total: %{text}<br>Effect: %{customdata:.4f}<extra></extra>",
                customdata=df_view["Effect"].values,
            )
            max_abs = float(df_view["AbsShare"].max()) if not df_view.empty else 0.5
            pad = 0.12
            fig.update_xaxes(
                range=[-(max_abs * (1 + pad)), max_abs * (1 + pad)],
                tickformat=".0%",
                title_text="Share of total (signed)",
                zeroline=True, zerolinewidth=1.6, zerolinecolor="#9CA3AF",
                gridcolor="#E5E7EB"
            )
            fig.update_yaxes(
                title_text="Variable",
                categoryorder="array",
                categoryarray=df_view["Variable"].tolist()
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=12, r=12, t=54, b=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0,
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb", borderwidth=1)
            )
            st.plotly_chart(fig, use_container_width=True)


    # =================== TOGGLE: FULL GRID (follows filters) ===================
    st.markdown("<div class='section-title'>Explore all combinations</div>", unsafe_allow_html=True)
    show_grid = st.toggle("Show full grid (scrollable, heatmap-colored table)", value=False, key="toggle_full_grid")

    if show_grid:
        st.caption("Values are signed shares per combination; coloring is centered at 0. Top-N + Others, name filter, and removals apply here too.")

        # --- helper to compute per-row signed shares (fallback to magnitude when denom≈0)
        def compute_row_shares(r: pd.Series, pairs: dict, keep_vars: list) -> dict:
            effects = {}
            for var, cols in pairs.items():
                if var not in keep_vars:
                    continue
                b = _tof(r.get(cols["beta_col"]), None)
                m = _tof(r.get(cols["mean_col"]), None) if cols["mean_col"] is not None else None
                if b is None or m is None:
                    continue
                effects[var] = b * m

            # apply Show filter per row (sign-based masking)
            if show_mode == "Positive":
                effects = {k: v for k, v in effects.items() if v > 0}
            elif show_mode == "Negative":
                effects = {k: v for k, v in effects.items() if v < 0}

            if not effects:
                return {}

            denom_s = sum(effects.values())
            if abs(denom_s) >= 1e-12:
                # signed normalization
                return {k: (v / denom_s) for k, v in effects.items()}

            # fallback: magnitude normalization but retain sign
            denom_a = sum(abs(v) for v in effects.values())
            if denom_a == 0:
                return {k: 0.0 for k in effects.keys()}
            return {k: (np.sign(v) * (abs(v) / denom_a)) for k, v in effects.items()}

        # --- variable universe after filters (name + removals)
        candidate_vars = [v for v in pairs_all.keys()]
        if name_filter:
            import re as _re
            candidate_vars = [v for v in candidate_vars if _re.search(name_filter, v, flags=_re.I)]
        if exclude_vars:
            candidate_vars = [v for v in candidate_vars if v not in exclude_vars]

        if not candidate_vars:
            st.info("No variables match the current filters for the grid.")
            st.stop()

        # --- build preliminary grid for all combinations (to choose Top-N globally)
        dim_cols = dimension_cols_all
        # Prefer Brand if present; else fall back to full combo label
        if "Brand" in saved_models.columns:
            combo_labels = saved_models["Brand"].astype(str).fillna("--")
        else:
            combo_labels = saved_models[dim_cols].astype(str).apply(
                lambda r: " | ".join([f"{c}:{r[c]}" for c in dim_cols]), axis=1
            )

        prelim_rows = []
        for _, r in saved_models.iterrows():
            shares = compute_row_shares(r, pairs_all, candidate_vars)
            prelim_rows.append(shares)
        prelim_df = pd.DataFrame(prelim_rows).fillna(0.0)
        prelim_df.index = combo_labels  # index == Brand (or combo fallback)

        # --- choose Top-N variables by avg |share| across all rows (post-filters)
        mean_abs = prelim_df.abs().mean(axis=0)
        top_vars = mean_abs.sort_values(ascending=False).head(int(top_n)).index.tolist()
        other_vars = [v for v in candidate_vars if v not in top_vars]

        # --- final grid with Top-N (+ optional Others)
        grid = pd.DataFrame(index=prelim_df.index)
        for v in top_vars:
            grid[v] = prelim_df.get(v, 0.0)
        if group_others and other_vars:
            grid["Others"] = prelim_df[other_vars].sum(axis=1)

        # --- move Brand out of index; hide index in UI; keep full column names
        grid_out = grid.copy()
        grid_out.insert(0, "Brand", grid_out.index.astype(str))
        grid_out.index = pd.RangeIndex(len(grid_out))

        # =================== RENDER: prefer Ag-Grid (pinned Brand + horizontal scroll), fallback to st.dataframe ===================
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

            # Bold headers
            st.markdown("""
            <style>
            .ag-theme-balham .ag-header-cell-label { font-weight: 700; }
            </style>
            """, unsafe_allow_html=True)

            num_cols = [c for c in grid_out.columns if c != "Brand"]
            vmax = float(np.nanmax(np.abs(grid_out[num_cols].values))) if num_cols else 1.0
            if vmax == 0: vmax = 1.0

            # Heatmap cell style (diverging around 0)
            js_heatmap = JsCode("""
            function(params){
            if (params.value === null || params.value === undefined) return {};
            var vmax = (params.context && params.context.vmax) ? params.context.vmax : 1.0;
            if (vmax === 0) return {};
            var v = params.value / vmax;
            if (v > 1) v = 1;
            if (v < -1) v = -1;
            function mix(w,c,t){return [
                Math.round(w[0]+(c[0]-w[0])*t),
                Math.round(w[1]+(c[1]-w[1])*t),
                Math.round(w[2]+(c[2]-w[2])*t)
            ];}
            var white=[255,255,255], red=[239,68,68], blue=[37,99,235];
            var t = Math.abs(v);
            var rgb = (v>=0)? mix(white,red,t) : mix(white,blue,t);
            return { backgroundColor: 'rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')' };
            }
            """)

            # Percent formatter
            js_percent = JsCode("""
            function(params){
            if (params.value === null || params.value === undefined) return '';
            return (params.value * 100).toFixed(2) + '%';
            }
            """)

            # Compute reasonable column widths from header text → ensures horizontal scroll & full names
            def _col_width(header: str, min_w=120, max_w=420, px_per_char=9):
                return max(min_w, min(max_w, int(len(header) * px_per_char)))

            gb = GridOptionsBuilder.from_dataframe(grid_out, enableValue=False, enableRowGroup=False, enablePivot=False)
            gb.configure_default_column(resizable=True, sortable=True, filter=True)
            gb.configure_grid_options(
                domLayout="normal",              # do NOT size to fit; enables horizontal scroll
                suppressMenuHide=False,
                ensureDomOrder=True,
                wrapHeaderText=True,             # wrap long header names
                autoHeaderHeight=True,           # grow header row to fit wrapped text
                suppressColumnVirtualisation=False
            )

            # Brand pinned + bold, width based on content
            brand_width = _col_width("Brand", min_w=150, max_w=260, px_per_char=10)
            gb.configure_column(
                "Brand",
                pinned="left",
                lockPinned=True,
                cellStyle={"fontWeight": "700", "color": "#111827"},
                headerName="Brand",
                width=brand_width,
                tooltipField="Brand"
            )

            # Numeric columns: percent + heatmap + width from header length
            for c in num_cols:
                header = c  # keep full header text
                width = _col_width(header)
                gb.configure_column(
                    c,
                    headerName=header,
                    width=width,
                    type=["numericColumn"],
                    valueFormatter=js_percent,
                    cellStyle=js_heatmap,
                    tooltipField=c
                )

            gridOptions = gb.build()
            gridOptions["context"] = {"vmax": vmax}

            AgGrid(
                grid_out,
                gridOptions=gridOptions,
                theme="balham",
                height=440,
                fit_columns_on_grid_load=False,   # keep natural widths (horizontal scrollbar visible)
                allow_unsafe_jscode=True
            )

        except ImportError:
            # Fallback: styled dataframe (no pinning; Streamlit will horizontal-scroll when wide)
            st.info("For a pinned Brand column and perfect horizontal scrolling, install: `pip install streamlit-aggrid`")
            num_cols = [c for c in grid_out.columns if c != "Brand"]
            vmax = float(np.nanmax(np.abs(grid_out[num_cols].values))) if num_cols else 1.0
            if vmax == 0: vmax = 1.0
            styled = (
                grid_out.style
                .format({c: "{:.2%}" for c in num_cols})
                .set_properties(subset=pd.IndexSlice[:, ["Brand"]], **{"font-weight": "700", "color": "#111827"})
                .background_gradient(cmap="RdBu", vmin=-vmax, vmax=+vmax, subset=pd.IndexSlice[:, num_cols])
                .set_table_styles([
                    {"selector": "th.col_heading",
                    "props": [("font-weight","700"), ("font-size","13.5px"),
                            ("background-color","#F9FAFB"), ("color","#111827")]}
                ])
            )
            st.dataframe(styled, use_container_width=True, height=440, hide_index=True)

        # --- download (with Brand as a real column)
        dl_grid_key = _unique_key("download_full_grid", grid_out.shape, show_mode, name_filter, top_n, group_others)
        st.download_button(
            "📥 Download full grid (signed shares)",
            data=grid_out.to_csv(index=False).encode("utf-8"),
            file_name="contribution_grid_signed.csv",
            mime="text/csv",
            key=dl_grid_key
        )

        # =================== PORTFOLIO — Σ(β×x) contributions (respects filters) ===================
        import re
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import streamlit as st
        import hashlib

        BRAND_GREEN = "#10B981"  # emerald-500
        BRAND_RED   = "#EF4444"  # red-500
        BRAND_BLUE  = "#2563EB"  # blue-600 (for Actuals)
        BRAND_GREEN_SOFT = "#6EE7B7"  # emerald-300 (for Other split)

        def _unique_key(prefix: str, *parts) -> str:
            h = hashlib.sha1(("|".join(map(str, parts))).encode()).hexdigest()[:8]
            return f"{prefix}_{h}"

        def _tof(x, default=None):
            try: return float(x)
            except Exception: return default

        def _find_pairs_from_columns(columns):
            beta_pref = ["Weighted_Beta_", "Selected_Beta_", "Beta_"]
            mean_pref = ["Mean_", "Selected_Mean_"]
            pairs = {}
            for c in columns:
                if not (isinstance(c, str) and "Beta_" in c): 
                    continue
                var = c
                for p in beta_pref:
                    if var.startswith(p):
                        var = var[len(p):]
                        break
                beta_col = next((f"{p}{var}" for p in ("Weighted_Beta_","Selected_Beta_","Beta_") if f"{p}{var}" in columns), None)
                if not beta_col: 
                    continue
                mean_col = next((f"{m}{var}" for m in ("Mean_","Selected_Mean_") if f"{m}{var}" in columns), None)
                pairs[var] = {"beta_col": beta_col, "mean_col": mean_col}
            return pairs

        def _pretty(v: str) -> str:
            if not isinstance(v, str): return str(v)
            s = re.sub(r"_meta(_impression)?$", "", v, flags=re.I)
            s = re.sub(r"_OUTCOME_.*$", "", s, flags=re.I)
            s = re.sub(r"_impression$", "", s, flags=re.I)
            s = s.replace("_", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s[:48] + ("…" if len(s) > 48 else "")

        st.markdown("### Portfolio — Contribution")

        # 0) Source DF holding all combinations (products/brands)
        if "saved_models" not in st.session_state or st.session_state["saved_models"] is None or st.session_state["saved_models"].empty:
            st.info("No portfolio data found in st.session_state['saved_models'].")
        else:
            df_all: pd.DataFrame = st.session_state["saved_models"].copy()
            pairs_all = _find_pairs_from_columns(df_all.columns)
            if not pairs_all:
                st.warning("No Beta/Mean columns found to build portfolio effects.")
            else:
                # 1) Start from the current filter context used in the single view
                #    Assumes you have variables: show_mode, name_filter, exclude_vars, top_n, group_others defined above.
                candidate_vars = list(pairs_all.keys())
                if 'name_filter' in locals() and name_filter:
                    candidate_vars = [v for v in candidate_vars if re.search(name_filter, v, flags=re.I)]
                if 'exclude_vars' in locals() and exclude_vars:
                    candidate_vars = [v for v in candidate_vars if v not in exclude_vars]
                if not candidate_vars:
                    st.info("No variables left after filters to compute portfolio.")
                else:
                    # 2) Aggregate raw effects across ALL rows (brands/products), respecting Show mode per brand row
                    effects_sum = {v: 0.0 for v in candidate_vars}
                    for _, r in df_all.iterrows():
                        # per-row effect build (optionally mask by sign for 'Show')
                        row_effects = {}
                        for v in candidate_vars:
                            cols = pairs_all[v]
                            b = _tof(r.get(cols["beta_col"]), None)
                            m = _tof(r.get(cols["mean_col"]), None) if cols["mean_col"] is not None else None
                            if b is None or m is None:
                                continue
                            e = b * m
                            # Apply Show filter at row level
                            if 'show_mode' in locals():
                                if show_mode == "Positive" and e <= 0: 
                                    continue
                                if show_mode == "Negative" and e >= 0: 
                                    continue
                            row_effects[v] = e
                        # Sum into portfolio
                        for v, e in row_effects.items():
                            effects_sum[v] += e

                    # 3) Build portfolio table, compute signed shares over remaining variables
                    port = pd.DataFrame({
                        "Variable": list(effects_sum.keys()),
                        "Portfolio Effect": list(effects_sum.values())
                    })
                    # Drop zero-only variables after filters
                    port = port[port["Portfolio Effect"].ne(0.0)]

                    if port.empty:
                        st.info("All portfolio effects are zero under current filters.")
                    else:
                        total_effect = float(port["Portfolio Effect"].sum())
                        if abs(total_effect) < 1e-12:
                            # fallback to magnitude-normalized shares for readability; keep sign on bar
                            denom = float(port["Portfolio Effect"].abs().sum())
                            port["AbsShare"] = port["Portfolio Effect"].abs() / denom if denom > 0 else np.nan
                            port["SignedShare"] = np.sign(port["Portfolio Effect"]) * port["AbsShare"]
                        else:
                            port["SignedShare"] = port["Portfolio Effect"] / total_effect
                            port["AbsShare"] = port["SignedShare"].abs()

                        # 4) Top-N + Others (same behavior as single view)
                        port = port.sort_values("AbsShare", ascending=False).reset_index(drop=True)
                        if 'group_others' in locals() and group_others and 'top_n' in locals() and len(port) > int(top_n):
                            N = int(top_n)
                            top = port.head(N).copy()
                            rest = port.iloc[N:]
                            others = pd.DataFrame([{
                                "Variable": "Others",
                                "Portfolio Effect": rest["Portfolio Effect"].sum(),
                                "SignedShare": rest["SignedShare"].sum(),
                                "AbsShare": abs(rest["SignedShare"].sum())
                            }])
                            port = pd.concat([top, others], ignore_index=True)

                        # 5) Split Contribution into Own vs Other by variable (stacked), unlinked stay single
                        plot_df = port.copy().reset_index(drop=True)
                        _rows = int(plot_df.shape[0]) if isinstance(plot_df, pd.DataFrame) else 10
                        _height = int(np.clip(30 * max(_rows, 6) + 180, 520, 1100))

                        # Prepare brand mappings if available
                        def _strip_media_tokens(s: str) -> str:
                            try:
                                import re as _re
                                s2 = str(s)
                                # Remove common media tokens to improve brand matching
                                s2 = _re.sub(r"impression_link_clicks?|impressions?|impr|imps|clicks?|google|meta|category|ad|grp|cpm|cpc", " ", s2, flags=_re.I)
                                return s2
                            except Exception:
                                return str(s)

                        def _norm_name(s: str) -> str:
                            try:
                                import re as _re
                                return _re.sub(r"[^a-z0-9]+", "", str(s).lower())
                            except Exception:
                                return str(s).lower()

                        brands_list = sorted(df_all["Brand"].astype(str).unique()) if "Brand" in df_all.columns else []
                        brand_norm_map = {b: _norm_name(b) for b in brands_list}
                        def map_var_to_brand(var_name: str):
                            pretty = _pretty(var_name)
                            pretty = _strip_media_tokens(pretty)
                            vn = _norm_name(pretty)
                            best = None
                            best_len = 0
                            for b, bn in brand_norm_map.items():
                                if bn and bn in vn and len(bn) > best_len:
                                    best = b; best_len = len(bn)
                            return best

                        # Compute own/other totals per variable across brands
                        own_signed_list = []
                        other_signed_list = []
                        unlinked_signed_list = []
                        own_effect_list = []
                        other_effect_list = []
                        unlinked_effect_list = []
                        labels_val = []
                        for _, vv in plot_df.iterrows():
                            v = vv["Variable"]
                            total_signed_share = float(vv["SignedShare"]) if pd.notna(vv["SignedShare"]) else 0.0
                            total_effect_val = _tof(vv.get("Portfolio Effect"), 0.0)
                            mapped_brand = map_var_to_brand(str(v)) if brands_list else None
                            if not mapped_brand:
                                own_signed_list.append(0.0)
                                other_signed_list.append(0.0)
                                unlinked_signed_list.append(total_signed_share)
                                own_effect_list.append(0.0)
                                other_effect_list.append(0.0)
                                unlinked_effect_list.append(total_effect_val)
                                labels_val.append(f"{abs(total_signed_share):.1%}")
                                continue
                            own_tot = 0.0; other_tot = 0.0
                            for _, rr in df_all.iterrows():
                                cols = pairs_all.get(v, None)
                                if not cols:
                                    continue
                                bcoef = _tof(rr.get(cols["beta_col"]), None)
                                mval = _tof(rr.get(cols["mean_col"]), None) if cols["mean_col"] is not None else None
                                if bcoef is None or mval is None:
                                    continue
                                eff_val = bcoef * mval
                                if 'show_mode' in locals():
                                    if show_mode == "Positive" and eff_val <= 0:
                                        continue
                                    if show_mode == "Negative" and eff_val >= 0:
                                        continue
                                br = str(rr.get("Brand", ""))
                                if br == mapped_brand:
                                    own_tot += eff_val
                                else:
                                    other_tot += eff_val
                            denom = abs(own_tot) + abs(other_tot)
                            if denom <= 0:
                                own_signed = 0.0
                                other_signed = 0.0
                                unlinked_signed = total_signed_share
                            else:
                                sign = 1.0 if total_signed_share >= 0 else -1.0
                                own_signed = sign * (abs(total_signed_share) * (abs(own_tot) / denom))
                                other_signed = sign * (abs(total_signed_share) * (abs(other_tot) / denom))
                                unlinked_signed = total_signed_share - (own_signed + other_signed)
                            own_signed_list.append(own_signed)
                            other_signed_list.append(other_signed)
                            unlinked_signed_list.append(unlinked_signed)
                            own_effect_list.append(own_tot)
                            other_effect_list.append(other_tot)
                            unlinked_effect_list.append(total_effect_val - (own_tot + other_tot))
                            labels_val.append(f"{abs(total_signed_share):.1%}")

                        plot_df["OwnSigned"] = own_signed_list
                        plot_df["OtherSigned"] = other_signed_list
                        plot_df["UnlinkedSigned"] = unlinked_signed_list
                        plot_df["OwnEffect"] = own_effect_list
                        plot_df["OtherEffect"] = other_effect_list
                        plot_df["UnlinkedEffect"] = unlinked_effect_list
                        # Merge Unlinked into Other so legend shows only Own vs Other
                        plot_df["OtherMerged"] = plot_df["OtherSigned"] + plot_df["UnlinkedSigned"]
                        plot_df["OtherEffectMerged"] = plot_df["OtherEffect"] + plot_df["UnlinkedEffect"]
                        plot_df["Label"] = labels_val

                        own_share_map = dict(zip(plot_df["Variable"], plot_df["OwnSigned"]))
                        other_share_map = dict(zip(plot_df["Variable"], plot_df["OtherMerged"]))
                        own_effect_map = dict(zip(plot_df["Variable"], plot_df["OwnEffect"]))
                        other_effect_map = dict(zip(plot_df["Variable"], plot_df["OtherEffectMerged"]))
                        port["OwnSigned"] = port["Variable"].map(own_share_map).fillna(0.0)
                        port["OtherSigned"] = port["Variable"].map(other_share_map).fillna(0.0)
                        port["OwnEffect"] = port["Variable"].map(own_effect_map).fillna(0.0)
                        port["OtherEffect"] = port["Variable"].map(other_effect_map).fillna(0.0)
                        port["OwnAbsShare"] = port["OwnSigned"].abs()
                        port["OtherAbsShare"] = port["OtherSigned"].abs()

                        # Optional: base table with split columns (kept hidden but ready for download)
                        disp = port.copy()
                        disp["Direction"] = np.where(disp["Portfolio Effect"] >= 0, "Positive", "Negative")
                        disp["Share (%)"] = (disp["AbsShare"] * 100).round(1)
                        disp["Own Share (%)"] = (disp["OwnAbsShare"] * 100).round(1)
                        disp["Other Share (%)"] = (disp["OtherAbsShare"] * 100).round(1)
                        disp = disp[["Variable", "Direction", "Portfolio Effect", "OwnEffect", "OtherEffect", "Share (%)", "Own Share (%)", "Other Share (%)", "SignedShare"]]

                        # st.dataframe(disp.sort_values("Share (%)", ascending=False), use_container_width=True, hide_index=True)
                        # st.download_button(
                        #     "📥 Download Portfolio Contributions (Σβx)",
                        #     data=disp.to_csv(index=False).encode("utf-8"),
                        #     file_name="portfolio_contributions_sum_betax.csv",
                        #     mime="text/csv",
                        #     key=_unique_key("dl_portfolio_sum_betax", len(disp))
                        # )

                        # Sort by total signed share
                        plot_df = plot_df.sort_values("SignedShare", ascending=True)
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_bar(y=plot_df["Variable"], x=plot_df["OwnSigned"], name="Own", orientation="h",
                                    marker=dict(color=BRAND_GREEN, line=dict(color="white", width=1.2)),
                                    text=[f"{abs(x):.1%}" for x in plot_df["OwnSigned"]], textposition="outside", cliponaxis=False)
                        fig.add_bar(y=plot_df["Variable"], x=plot_df["OtherMerged"], name="Other", orientation="h",
                                    marker=dict(color="#9CA3AF", line=dict(color="white", width=1.2)),
                                    text=[f"{abs(x):.1%}" for x in plot_df["OtherMerged"]], textposition="outside", cliponaxis=False)
                        fig.update_layout(barmode="stack")
                        # Intelligent asymmetric scaling
                        neg_min0 = float(min(0.0, plot_df["SignedShare"].min())) if not plot_df.empty else 0.0
                        pos_max0 = float(max(0.0, plot_df["SignedShare"].clip(lower=0).max())) if not plot_df.empty else 0.0
                        pad0 = 0.08
                        left0 = (neg_min0 * (1 + pad0)) if neg_min0 < 0 else -0.05
                        right0 = (pos_max0 * (1 + pad0)) if pos_max0 > 0 else 0.05
                        left0 = max(left0, -0.6)
                        right0 = min(right0, 0.85)
                        if right0 <= 0.05: right0 = 0.05
                        if left0 >= -0.05: left0 = -0.05
                        span0 = right0 - left0
                        dt0 = 0.1 if span0 > 0.6 else 0.05
                        fig.update_xaxes(
                            range=[left0, right0],
                            tickformat=".0%",
                            dtick=dt0,
                            title_text="Portfolio share (signed, from Σβx)",
                            zeroline=True, zerolinewidth=1.6, zerolinecolor="#9CA3AF",
                            gridcolor="#E5E7EB",
                            tickfont=dict(size=12)
                        )
                        fig.update_yaxes(title_text="", categoryorder="array", categoryarray=plot_df["Variable"].tolist(), tickfont=dict(size=12))
                        fig.update_layout(
                            template="plotly_white",
                            margin=dict(l=200, r=20, t=64, b=10),
                            showlegend=True,
                            bargap=0.25,
                            hoverlabel=dict(font=dict(size=12)),
                            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0,
                                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=12)),
                            title=dict(text="Portfolio Contributions (Σβ×x) — Own vs Other by Variable", x=0.01, y=0.98,
                                    font=dict(size=18, family="Inter, sans-serif", color="#111827"))
                        )
                        # Optional: Upload a variable-level actuals file to compare Actual Share vs Contribution Share
                        st.markdown("#### Optional: Compare with Actual Totals (one chart)")
                        vol_file2 = st.file_uploader(
                            "Upload actuals file (CSV/XLSX). EITHER long [Variable, Value] OR wide with variable columns",
                            type=["csv", "xlsx", "xls"],
                            key="upload_portfolio_volume"
                        )
                        rendered_combined = False
                        vol_df2 = None
                        if vol_file2 is not None:
                            try:
                                vol_df2 = pd.read_excel(vol_file2) if vol_file2.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(vol_file2)
                            except Exception as e:
                                st.error(f"Could not read the file: {e}")
                                vol_df2 = None

                        if isinstance(vol_df2, pd.DataFrame) and not vol_df2.empty:
                            # Accept both long format (Variable, Value) and wide format (one column per variable)
                            if {"Variable", "Value"}.issubset(set(vol_df2.columns)):
                                tmp = vol_df2[["Variable", "Value"]].copy()
                                tmp["Variable"] = tmp["Variable"].astype(str)
                                tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").fillna(0.0)
                                vol_grouped2 = (
                                    tmp.groupby("Variable", as_index=False)["Value"].sum().rename(columns={"Value": "Total_Actual"})
                                )
                            else:
                                # Wide format: match variable columns and sum across rows per variable
                                wide = vol_df2.copy()
                                wide_cols = [c for c in wide.columns if c in set(port["Variable"]) ]
                                if wide_cols:
                                    # Melt to long or directly compute totals
                                    totals = []
                                    for c in wide_cols:
                                        try:
                                            col_vals = pd.to_numeric(wide[c], errors="coerce").fillna(0.0)
                                            totals.append({"Variable": c, "Total_Actual": float(col_vals.sum())})
                                        except Exception:
                                            totals.append({"Variable": c, "Total_Actual": 0.0})
                                    vol_grouped2 = pd.DataFrame(totals)
                                else:
                                    vol_grouped2 = pd.DataFrame(columns=["Variable","Total_Actual"])  # no matching vars
                                # Align with variables present in portfolio contributions
                                vol_grouped2 = vol_grouped2[vol_grouped2["Variable"].isin(port["Variable"])].copy()
                                if not vol_grouped2.empty:
                                    grand_total2 = float(vol_grouped2["Total_Actual"].sum())
                                    vol_grouped2["Actual_Share"] = vol_grouped2["Total_Actual"] / grand_total2 if grand_total2 > 0 else 0.0

                                    # Build combined view: Variable, Contribution Share (Abs), Volume Share
                                    combined2 = pd.merge(
                                        port[["Variable", "SignedShare", "AbsShare", "Portfolio Effect", "OwnSigned", "OtherSigned", "OwnEffect", "OtherEffect"]],
                                        vol_grouped2[["Variable", "Actual_Share", "Total_Actual"]],
                                        on="Variable",
                                        how="inner"
                                    )
                                    if combined2.empty:
                                        st.info("No overlap between portfolio contributions and uploaded actuals.")
                                    else:
                                        combined2["OwnAbsShare"] = combined2["OwnSigned"].fillna(0.0).abs()
                                        combined2["OtherAbsShare"] = combined2["OtherSigned"].fillna(0.0).abs()

                                        import plotly.graph_objects as go
                                        fig2 = go.Figure()

                                        def _format_share_effect(share_val, effect_val):
                                            sv = 0.0 if pd.isna(share_val) else float(share_val)
                                            ev = 0.0 if pd.isna(effect_val) else float(effect_val)
                                            return f"{abs(sv):.1%} | {abs(ev):,.0f}"

                                        own_x = combined2["OwnSigned"].fillna(0.0)
                                        other_x = combined2["OtherSigned"].fillna(0.0)
                                        actual_x = combined2["Actual_Share"].fillna(0.0)
                                        total_actual_vals = combined2["Total_Actual"].fillna(0.0)
                                        own_text = [_format_share_effect(s, e) for s, e in zip(combined2["OwnSigned"], combined2["OwnEffect"])]
                                        other_text = [_format_share_effect(s, e) for s, e in zip(combined2["OtherSigned"], combined2["OtherEffect"])]
                                        actual_text = [f"{share:.1%} | {total:,.0f}" for share, total in zip(actual_x, total_actual_vals)]

                                        fig2.add_bar(
                                            y=combined2["Variable"], x=own_x, orientation="h",
                                            name="Own Contribution Share",
                                            marker=dict(color=BRAND_GREEN, line=dict(color="white", width=1.2)),
                                            text=own_text,
                                            textposition="outside",
                                            cliponaxis=False,
                                            offsetgroup="contrib"
                                        )
                                        fig2.add_bar(
                                            y=combined2["Variable"], x=other_x, orientation="h",
                                            name="Other Contribution Share",
                                            marker=dict(color=BRAND_GREEN_SOFT, line=dict(color="white", width=1.2)),
                                            text=other_text,
                                            textposition="outside",
                                            cliponaxis=False,
                                            offsetgroup="contrib"
                                        )
                                        fig2.add_bar(
                                            y=combined2["Variable"], x=actual_x, orientation="h",
                                            name="Actual Share",
                                            marker=dict(color=BRAND_BLUE, line=dict(color="white", width=1.2)),
                                            text=actual_text,
                                            textposition="outside",
                                            cliponaxis=False,
                                            offsetgroup="actual"
                                        )

                                        signed_vals = combined2["SignedShare"].fillna(0.0)
                                        try:
                                            neg_min = float(min(0.0, signed_vals.min())) if not signed_vals.empty else 0.0
                                        except Exception:
                                            neg_min = 0.0
                                        try:
                                            pos_max = float(max(
                                                (actual_x.max() if not actual_x.empty else 0.0),
                                                (signed_vals.clip(lower=0).max() if not signed_vals.empty else 0.0)
                                            ))
                                        except Exception:
                                            pos_max = 0.0
                                        pad2 = 0.08
                                        left = (neg_min * (1 + pad2)) if neg_min < 0 else -0.05
                                        right = (pos_max * (1 + pad2)) if pos_max > 0 else 0.05
                                        left = max(left, -0.6)
                                        right = min(right, 0.85)
                                        if right <= 0.05:
                                            right = 0.05
                                        if left >= -0.05:
                                            left = -0.05
                                        span = right - left
                                        dt = 0.1 if span > 0.6 else 0.05
                                        fig2.update_xaxes(
                                            range=[left, right],
                                            tickformat=".0%",
                                            dtick=dt,
                                            title_text="Share (signed for Contribution; positive for Actuals)",
                                            zeroline=True, zerolinewidth=1.6, zerolinecolor="#9CA3AF",
                                            gridcolor="#E5E7EB",
                                            tickfont=dict(size=12)
                                        )
                                        order_col = "Actual_Share" if "Actual_Share" in combined2.columns else "AbsShare"
                                        fig2.update_yaxes(
                                            title_text="",
                                            categoryorder="array",
                                            categoryarray=combined2.sort_values(order_col)["Variable"].tolist(),
                                            tickfont=dict(size=12)
                                        )
                                        fig2.update_layout(
                                            barmode="relative",
                                            template="plotly_white",
                                            margin=dict(l=220, r=20, t=64, b=10),
                                            bargap=0.25,
                                            hoverlabel=dict(font=dict(size=12)),
                                            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0,
                                                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb", borderwidth=1, font=dict(size=12)),
                                            title=dict(text="Portfolio β×x Contribution vs Actual Shares (Own vs Other)", x=0.01, y=0.98,
                                                    font=dict(size=18, family="Inter, sans-serif", color="#111827"))
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                        rendered_combined = True

                                else:
                                    st.info("Uploaded volume file has no variables matching the portfolio chart.")
                                # No specific error here; handled by empty/mismatch info above

                        # Only render base contribution chart if combined chart not shown
                        if not rendered_combined:
                            st.plotly_chart(fig, use_container_width=True)

                        # Removed secondary own/other chart; the main chart now shows the split (Own/Other/Unlinked)

    # Navigation
        cgo1, cgo2 = st.columns(2)
        with cgo1:
            if st.button("Go to Marketing Inputs →", key="btn_go_marketing", type="secondary"):
                go("optimizer")
        with cgo2:
            if st.button("Go to Contextual Inputs →", key="btn_go_context", type="secondary"):
                go("contextual")



    colA, colB = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"):
            go("home")
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Journeys · TripleWhale Paths
#  - Purchase item from sales_file.csv (Order name == journey_name)
#  - Ad → Product mapping from product_title_mapped.xlsx (shown inline beside ad badge)
#  - Right-side compact filters; left big scrollable timeline; purchase anchored
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import html
import re

# Files (same folder as app.py)
SALES_FILE = Path(__file__).with_name("sales_file.csv")
AD_PRODUCT_MAP_FILE = Path(__file__).with_name("product_title_mapped.xlsx")

# ------------------------------- helpers -------------------------------------
def _first_valid_value(s: pd.Series):
    for v in s:
        if pd.notna(v) and str(v).strip():
            return v
    return pd.NA

def _format_timedelta(td) -> str:
    if pd.isna(td):
        return "--"
    secs = int(td.total_seconds())
    if secs < 60: return f"{secs}s"
    mins, s = divmod(secs, 60)
    if mins < 60: return f"{mins}m {s}s"
    hrs, m = divmod(mins, 60)
    if hrs < 24: return f"{hrs}h {m}m"
    days, h = divmod(hrs, 24)
    return f"{days}d {h}h"

def _shorten(x: object, limit: int = 48) -> str:
    if not isinstance(x, str): return ""
    t = x.strip()
    return t if len(t) <= limit else t[: max(1, limit - 3)] + "..."

def _nullish(x) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)): return True
    s = str(x).strip().lower()
    return s in ("", "none", "null", "nan", "na", "n/a")

def _first_of_month(d: date) -> date:
    return date(d.year, d.month, 1)

def _prev_month_range(anchor: date):
    if anchor.month == 1:
        start = date(anchor.year - 1, 12, 1)
    else:
        start = date(anchor.year, anchor.month - 1, 1)
    end = _first_of_month(anchor) - timedelta(days=1)
    return start, end

# -------------------------- sales_file.csv → purchase -------------------------
def load_sales_mapping(file_path: Path) -> pd.DataFrame:
    """
    sales_file.csv columns (min): 'Order name', 'Product title'
    Return: ['journey_name','product_title'] (first non-blank per order)
    """
    if not file_path.exists():
        return pd.DataFrame(columns=["journey_name", "product_title"])
    df = pd.read_csv(file_path, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names):
        for nm in names:
            if nm.lower() in cols: return cols[nm.lower()]
        return None

    c_order = pick("order name", "order_name", "journey name", "journey_name")
    c_prod  = pick("product title", "product", "title")
    if c_order is None or c_prod is None:
        return pd.DataFrame(columns=["journey_name", "product_title"])

    df["journey_name"]  = df[c_order].astype(str).str.strip()
    df["product_title"] = df[c_prod].astype(str).str.strip()
    bad = df["product_title"].str.lower().isin(["", "nan", "none", "null", "na", "n/a"])
    df = df[~bad].copy()
    return (df.groupby("journey_name", as_index=False)
              .agg(product_title=("product_title", "first")))

# ---------------- product_title_mapped.xlsx → ad_name → mapped product --------
def load_ad_product_map(file_path: Path) -> pd.DataFrame:
    """
    product_title_mapped.xlsx columns: 'Ad name', 'Product title'
    Return: ['ad_name','ad_product'] cleaned.
    """
    if not file_path.exists():
        return pd.DataFrame(columns=["ad_name", "ad_product"])
    df = pd.read_excel(file_path, dtype=str)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "ad name" not in df.columns or "product title" not in df.columns:
        return pd.DataFrame(columns=["ad_name", "ad_product"])
    df = df.rename(columns={"ad name": "ad_name", "product title": "ad_product"})
    df["ad_name"] = df["ad_name"].astype(str).str.strip()
    df["ad_product"] = df["ad_product"].astype(str).str.strip()
    df = df[(df["ad_name"] != "") & (df["ad_product"] != "")]
    df = df.drop_duplicates(subset=["ad_name"], keep="first").reset_index(drop=True)
    return df[["ad_name", "ad_product"]]

def ad_to_product(ad_name: str, ad_map_df: pd.DataFrame) -> str:
    """Exact, then _V# strip, then CAT-Conversion fallback, then prefix match."""
    if not isinstance(ad_name, str) or not ad_name.strip(): return ""
    candidate = ad_name.strip()

    # 1) exact
    row = ad_map_df.loc[ad_map_df["ad_name"] == candidate]
    if not row.empty: return str(row.iloc[0]["ad_product"]).strip()

    # 2) strip version suffix _V1/_V10
    m = re.match(r"^(.*)_V\d{1,2}$", candidate)
    if m:
        base = m.group(1)
        row2 = ad_map_df.loc[ad_map_df["ad_name"] == base]
        if not row2.empty: return str(row2.iloc[0]["ad_product"]).strip()

    # 3) CAT conversion families → Category Ad
    low = candidate.lower()
    if ("cat" in low and "conversion" in low) or ("segmented_cat_conversion" in low):
        return "Category Ad"

    # 4) prefix fallback
    starts = ad_map_df.loc[ad_map_df["ad_name"].apply(lambda x: candidate.startswith(str(x)))]
    if not starts.empty: return str(starts.iloc[0]["ad_product"]).strip()

    return ""

# --------------------------- date & repeats UI --------------------------------
def _data_presets_ui(min_date: date, max_date: date):
    anchor = max_date
    this_month = _first_of_month(anchor)
    last_m_start, last_m_end = _prev_month_range(anchor)
    preset = st.selectbox(
        "Date range (by purchase date in data)",
        ["Last 7 days","Last 30 days","Today (data)","Yesterday (data)",
         "This month (data)","Last month (data)","Custom range","All data"],
        index=0, key="jr_date_preset",
    )
    if preset == "Today (data)":
        start_d, end_d = anchor, anchor
    elif preset == "Yesterday (data)":
        start_d = anchor - timedelta(days=1); end_d = start_d
    elif preset == "Last 7 days":
        start_d = anchor - timedelta(days=6); end_d = anchor
    elif preset == "Last 30 days":
        start_d = anchor - timedelta(days=29); end_d = anchor
    elif preset == "This month (data)":
        start_d, end_d = this_month, anchor
    elif preset == "Last month (data)":
        start_d, end_d = last_m_start, last_m_end
    elif preset == "Custom range":
        default_start = max(anchor - timedelta(days=6), min_date)
        default_end   = anchor
        dr = st.date_input("Pick a date range (from data)",
                           value=(default_start, default_end),
                           min_value=min_date, max_value=max_date,
                           key="jr_custom_range")
        if isinstance(dr, tuple) and len(dr) == 2:
            start_d, end_d = dr[0], dr[1]
        else:
            start_d, end_d = default_start, default_end
    else:
        return None, None, "All data"
    start_d = max(start_d, min_date); end_d = min(end_d, max_date)
    return start_d, end_d, preset

def _repeat_threshold_ui():
    hide_rr = st.checkbox("Hide rapid repeats", value=True, key="jr_hide_rr")
    choice = st.selectbox("Repeat threshold",
                          ["60 sec","30 sec","2 min","5 min","Custom (seconds)"],
                          index=0, key="jr_rr_choice")
    if choice == "30 sec": secs = 30
    elif choice == "60 sec": secs = 60
    elif choice == "2 min": secs = 120
    elif choice == "5 min": secs = 300
    else:
        secs = int(st.number_input("Custom seconds", min_value=1, value=60, step=1, key="jr_rr_custom"))
    return hide_rr, secs

# ---------------------------------- MAIN --------------------------------------
def show_journeys():
    st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
    st.markdown("<span class='qm-eyebrow'>Section 3</span>", unsafe_allow_html=True)
    st.markdown("<h2 class='title-h2'>Journeys &#183; TripleWhale Paths</h2>", unsafe_allow_html=True)

    # Load data
    data_path = JOURNEYS_FILE
    if not data_path.exists():
        st.error(f"Could not find '{data_path.name}'. Place the TripleWhale export alongside app.py.")
        st.markdown("</div>", unsafe_allow_html=True); return
    try:
        journeys_df = load_journey_data(str(data_path))
    except Exception as exc:
        st.error(f"Could not load journeys: {exc}")
        st.markdown("</div>", unsafe_allow_html=True); return
    if journeys_df.empty:
        st.info("No journeys remain after loading."); st.markdown("</div>", unsafe_allow_html=True); return

    # Types
    if not pd.api.types.is_datetime64_any_dtype(journeys_df["event_time"]):
        journeys_df["event_time"] = pd.to_datetime(journeys_df["event_time"], errors="coerce")
    et_col = "event_type_norm" if "event_type_norm" in journeys_df.columns else "event_type"
    journeys_df[et_col] = journeys_df[et_col].fillna("").astype(str)

    # Anchor by first purchase
    is_purchase_full = journeys_df[et_col].str.lower().eq("purchase")
    purchase_times_full = (journeys_df.loc[is_purchase_full]
                           .groupby("journey_name")["event_time"].min()
                           .rename("first_purchase_time"))
    if purchase_times_full.empty:
        st.info("No purchase-anchored journeys found."); st.markdown("</div>", unsafe_allow_html=True); return
    min_d, max_d = purchase_times_full.min().date(), purchase_times_full.max().date()

    # Layout
    left, right = st.columns([2.4, 1.0], gap="large")
    with right:
        st.markdown("""
        <style>
          .filter-card { padding:14px; border:1px solid #e8ecf2; border-radius:14px; background:#fbfcff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
          .tiny-note { color:#6b7b8f; font-size:.82rem; margin-top:4px; }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
        st.markdown("#### Filters", unsafe_allow_html=True)
        st.markdown("**Date range**")
        start_date, end_date, preset = _data_presets_ui(min_d, max_d)
        st.markdown("**Rapid repeats**")
        hide_rr, rr_seconds = _repeat_threshold_ui()
        st.markdown("<div class='tiny-note'>Removes steps closer than the threshold to the previous step (keeps earlier; Purchase always kept).</div>", unsafe_allow_html=True)

    # Apply date range (by first purchase)
    if start_date and end_date:
        keep = purchase_times_full[(purchase_times_full >= pd.Timestamp(start_date)) &
                                   (purchase_times_full <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))]
    else:
        keep = purchase_times_full
    if keep.empty:
        with left: st.info(f"No journeys with a purchase in the selected range ({preset}).")
        st.markdown("</div>", unsafe_allow_html=True); return

    # Trim to first purchase
    journeys_df = journeys_df.merge(keep.rename("first_purchase_time"), on="journey_name", how="inner")
    journeys_df = journeys_df[journeys_df["event_time"] <= journeys_df["first_purchase_time"]].copy()

    # Rule: non-purchase must have ad_name
    ad_ok = ~journeys_df["ad_name"].map(_nullish) if "ad_name" in journeys_df.columns else pd.Series(False, index=journeys_df.index)
    is_purchase_now = journeys_df[et_col].str.lower().eq("purchase")
    journeys_df = journeys_df[is_purchase_now | ad_ok].copy()

    # Ensure at least one non-purchase row per journey
    nonp_counts = (journeys_df.loc[~is_purchase_now].groupby("journey_name").size().rename("np_rows"))
    journeys_df = journeys_df.merge(nonp_counts, on="journey_name", how="left")
    journeys_df = journeys_df[journeys_df["np_rows"].fillna(0) > 0].copy()
    if journeys_df.empty:
        with left: st.info("No journeys remain after applying ad-name rule and date range.")
        st.markdown("</div>", unsafe_allow_html=True); return

    # Attach purchase product (sales_file.csv)
    sales_map = load_sales_mapping(SALES_FILE)
    if not sales_map.empty:
        journeys_df = journeys_df.merge(sales_map, on="journey_name", how="left")
    else:
        journeys_df["product_title"] = ""

    # Attach ad→product mapping for non-purchase steps
    ad_map_df = load_ad_product_map(AD_PRODUCT_MAP_FILE)
    if ad_map_df.empty:
        journeys_df["ad_product"] = ""
    else:
        journeys_df["ad_product"] = journeys_df["ad_name"].apply(lambda x: ad_to_product(x, ad_map_df))

    # Sort
    journeys_df = journeys_df.sort_values(["journey_name","event_time","event_sequence"]).reset_index(drop=True)

    # Summary
    summary_df = (journeys_df.groupby("journey_name")
                  .agg(steps=("event_sequence","count"),
                       first_event=("event_time","min"),
                       last_event=("event_time","max"),
                       purchase_time=("first_purchase_time","first"))
                  .reset_index())
    ad_np = (journeys_df.loc[journeys_df[et_col].str.lower().ne("purchase")]
             .groupby("journey_name")["ad_name"].agg(_first_valid_value).rename("ad_name_np"))
    summary_df = summary_df.merge(ad_np, on="journey_name", how="left")
    if "source" in journeys_df.columns:
        summary_df = summary_df.merge(
            journeys_df.groupby("journey_name")["source"].agg(_first_valid_value).rename("source"),
            on="journey_name", how="left",
        )
    summary_df["journey_span"] = summary_df["last_event"] - summary_df["first_event"]
    summary_df = summary_df.sort_values(["purchase_time","first_event"], ascending=[False,False]).reset_index(drop=True)

    st.caption(f"{len(summary_df)} journeys (purchase in {preset}). Non-purchase steps require Ad Name; Purchase may not. "
               "Purchase shows Item from sales_file.csv. Non-purchase shows Ad + mapped Product.")

    # Purchase product options (for filter)
    purchase_titles = (journeys_df.loc[journeys_df[et_col].str.lower().eq("purchase"), "product_title"]
                       .dropna().astype(str).str.strip())
    product_options = sorted({t for t in purchase_titles if t})

    # Right: refine
    with right:
        st.markdown("**Refine**")
        if product_options:
            product_filter = st.multiselect("Product title (Item)", product_options, key="jr_product_filter")
        else:
            product_filter = []
            st.caption("No product titles found in the selected date range.")
        source_filter = None
        if "source" in summary_df.columns:
            source_options = sorted({str(s).strip() for s in summary_df["source"].dropna() if str(s).strip()})
            source_filter = st.multiselect("Source", source_options, key="journeys_filter_source")

    # Apply list filters
    filtered_df = summary_df.copy()
    if product_filter:
        jp = (journeys_df.loc[journeys_df[et_col].str.lower().eq("purchase"), ["journey_name","product_title"]]
              .dropna())
        jp["product_title"] = jp["product_title"].astype(str).str.strip()
        allowed = set(jp[jp["product_title"].isin(product_filter)]["journey_name"].unique())
        filtered_df = filtered_df[filtered_df["journey_name"].isin(allowed)]
    if source_filter:
        filtered_df = filtered_df[filtered_df["source"].isin(source_filter)]
    if filtered_df.empty:
        with left:
            st.info("No journeys match the current filters.")
        st.markdown("</div>", unsafe_allow_html=True); return

    # Journey dropdown (post-filters)
    label_map = {}
    for row in filtered_df.itertuples():
        pieces = [row.journey_name]
        if "source" in filtered_df.columns and isinstance(row.source, str) and row.source.strip():
            pieces.append(_shorten(row.source, 36))
        ad_label = _shorten(row.ad_name_np, 48) if isinstance(row.ad_name_np, str) else ""
        if ad_label: pieces.append(ad_label)
        pieces.append(f"{int(row.steps)} steps")
        label_map[row.journey_name] = " | ".join([p for p in pieces if p])

    options = filtered_df["journey_name"].tolist()
    prev_choice = st.session_state.get("journeys_selected")
    default_index = options.index(prev_choice) if prev_choice in options else 0
    with right:
        selected_journey = st.selectbox("Journey", options, index=default_index,
                                        format_func=lambda j: label_map.get(j, j),
                                        key="journeys_selected")
        st.markdown("</div>", unsafe_allow_html=True)  # close filter-card

    # --------------------------- LEFT: render timeline -------------------------
    with left:
        st.markdown("""
        <style>
          .journey-hero { display:flex; align-items:center; gap:14px; margin:6px 0 10px 0; }
          .ad-hero { font-weight:800; font-size:1.06rem; padding:8px 12px; border-radius:16px;
                     border:1px solid rgba(0,0,0,0.06); background:linear-gradient(180deg,#fff,#fafafa);
                     box-shadow:0 1px 2px rgba(0,0,0,0.06); }
          .hero-pills { display:flex; gap:8px; flex-wrap:wrap; }
          .hero-pill { font-size:.86rem; padding:5px 12px; border-radius:999px; background:#f5f7fa; border:1px solid #edf0f3; }
          .journey-scroll { max-height:640px; overflow-y:auto; padding:10px 10px; background:#fff;
                            border:1px solid #e8ecf2; border-radius:14px; box-shadow:inset 0 1px 3px rgba(0,0,0,.05); }
          .journey-step { display:grid; grid-template-columns: 52px 1fr; gap:12px; padding:10px; margin:8px 0;
                          background:#ffffff; border:1px solid #eef1f6; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
          .journey-seq { display:flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:12px;
                         background:#f7f9fc; font-weight:700; font-size:.95rem; border:1px solid #eef1f6; }
          .journey-eventline { display:flex; align-items:center; gap:8px; }
          .journey-event { font-weight:800; }
          .ad-badge { margin-left:auto; font-weight:800; padding:4px 10px; border-radius:10px; background:#fff5e6; border:1px solid #ffe2b8; }
          .inline-chip { margin-left: 8px; }   /* NEW: place mapped product inline with ad badge */
          .journey-page { color:#52637a; font-size:.94rem; margin:2px 0 2px 26px; }
          .journey-meta { color:#66798f; font-size:.86rem; display:flex; gap:12px; margin-left:26px; }
          .journey-empty { opacity:.55; }
          .prod-box { margin:6px 0 0 26px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
          .prod-chip { font-size:.86rem; padding:4px 10px; border:1px solid #e7eaf0; background:#fbfcfe; border-radius:999px; }
          .prod-chip-ad { font-size:.82rem; padding:3px 8px; border:1px solid #eddff3; background:#faf5ff; border-radius:999px; }
          .ev-view .journey-seq { background:#f2f7ff; }
          .ev-click .journey-seq { background:#f6f3ff; }
          .ev-cart .journey-seq { background:#fff6f0; }
          .ev-checkout .journey-seq { background:#f1fff4; }
          .ev-purchase { border:1px solid #d2f5e5; background:#f7fffb; }
          .ev-purchase .journey-seq { background:#f0fff8; border:1px solid #d2f5e5; }
        </style>
        """, unsafe_allow_html=True)

        # Header chips
        sr = summary_df[summary_df["journey_name"] == selected_journey].iloc[0]
        ad_hero = html.escape(_shorten(sr.ad_name_np, 80) or "--")
        steps_label = f'{int(sr.steps)} steps'
        span_label = _format_timedelta(sr["last_event"] - sr["first_event"])
        st.markdown(
            f"<div class='journey-hero'>"
            f"<div class='ad-hero'>Ad • <span>{ad_hero}</span></div>"
            f"<div class='hero-pills'><div class='hero-pill'>Journey Span: {html.escape(span_label)}</div>"
            f"<div class='hero-pill'>{html.escape(steps_label)}</div></div></div>",
            unsafe_allow_html=True,
        )

        # Build timeline (chronological to purchase, then render purchase on top)
        base = journeys_df[journeys_df["journey_name"] == selected_journey].copy().reset_index(drop=True)
        p_mask = base[et_col].str.lower().eq("purchase"); fp_idx = base.index[p_mask][0]
        base = base.loc[:fp_idx].copy()

        if hide_rr and rr_seconds > 0:
            base["delta_prev_chrono"] = base["event_time"].diff()
            keep_bursts = ((base.index == 0) |
                           (base[et_col].str.lower().eq("purchase")) |
                           (base["delta_prev_chrono"].isna()) |
                           (base["delta_prev_chrono"] >= pd.Timedelta(seconds=int(rr_seconds))))
            base = base[keep_bursts].copy().reset_index(drop=True)

        p_mask2 = base[et_col].str.lower().eq("purchase"); fp_idx2 = base.index[p_mask2][0]
        prior = base.iloc[:fp_idx2] if fp_idx2 > 0 else base.iloc[0:0]
        journey_df = pd.concat([base.iloc[[fp_idx2]], prior.iloc[::-1]], ignore_index=True)

        # Display fields
        def _disp(s): return s.strip() if isinstance(s, str) and s.strip() else "--"
        journey_df["event_display"] = (journey_df["event_type"] if "event_type" in journey_df else journey_df[et_col]).apply(_disp)
        journey_df["page_display"]  = journey_df.get("page_location", pd.Series([""]*len(journey_df))).apply(_disp)
        if pd.api.types.is_datetime64_any_dtype(journey_df["event_time"]):
            journey_df["event_time_display"] = journey_df["event_time"].dt.strftime("%b %d, %Y - %I:%M:%S %p")
        else:
            journey_df["event_time_display"] = journey_df.get("event_timestamp", pd.Series(["--"]*len(journey_df))).apply(_disp)
        journey_df["delta_prev"] = journey_df["event_time"] - journey_df["event_time"].shift(1)
        journey_df.loc[0,"delta_prev"] = pd.Timedelta(0)
        journey_df["delta_prev_label"] = journey_df["delta_prev"].abs().apply(_format_timedelta)
        journey_df["source_display"] = journeys_df.get("source", pd.Series([""]*len(journey_df))).apply(lambda x: _shorten(x, 32))
        journey_df["ad_display"] = journey_df["ad_name"].apply(lambda x: _shorten(x, 48) if isinstance(x, str) and x.strip() else "")
        journey_df["ad_product_display"] = journey_df["ad_product"].apply(lambda x: _shorten(x, 60) if isinstance(x, str) and x.strip() else "")

        def _etype_class_and_icon(s: str):
            s = (s or "").lower()
            if "purchase" in s: return "ev-purchase","🛍️"
            if "add_to_cart" in s or "addtocart" in s or "cart" in s: return "ev-cart","🛒"
            if "checkout" in s or "begin_checkout" in s: return "ev-checkout","💳"
            if "click" in s or "ad_click" in s: return "ev-click","🖱️"
            if "search" in s: return "ev-search","🔎"
            if "view" in s or "page_view" in s or "view_content" in s or "view_item" in s: return "ev-view","👀"
            return "ev-other","•"
        journey_df["etype_class"], journey_df["etype_icon"] = zip(*journey_df[et_col].apply(_etype_class_and_icon))

        # Render
        tl = ["<div class='journey-scroll'>"]
        for row in journey_df.itertuples(index=False):
            seq_label  = html.escape(str(row.event_sequence)) if pd.notna(row.event_sequence) else "--"
            event_label = html.escape(row.event_display) if isinstance(row.event_display, str) else "--"
            page_label  = html.escape(row.page_display) if isinstance(row.page_display, str) else "--"
            meta_bits = []
            if row.event_time_display and row.event_time_display != "--":
                meta_bits.append(f"<span><strong>{html.escape(row.event_time_display)}</strong></span>")
            if row.delta_prev_label not in ("--","0s"):
                meta_bits.append(f"<span>+{html.escape(row.delta_prev_label)} since prior</span>")
            if isinstance(row.source_display, str) and row.source_display.strip():
                meta_bits.append(f"<span>{html.escape(row.source_display)}</span>")
            meta_html = "".join(meta_bits) if meta_bits else "<span class='journey-empty'>No metadata</span>"

            # --- INLINE ad badge + mapped product (non-purchase only) ---
            show_ad = ("purchase" not in (row.event_display or "").lower()) and isinstance(row.ad_display, str) and row.ad_display.strip()
            ad_html = f"<span class='ad-badge'><strong>{html.escape(row.ad_display)}</strong></span>" if show_ad else ""
            ad_prod_inline = ""
            if show_ad and isinstance(row.ad_product_display, str) and row.ad_product_display.strip():
                ad_prod_inline = f"<span class='prod-chip-ad inline-chip'>{html.escape(row.ad_product_display)}</span>"

            # Purchase item chip
            purchase_item_html = ""
            if "purchase" in (row.event_display or "").lower():
                title = getattr(row, "product_title", "")
                title = title.strip() if isinstance(title, str) else ""
                if title:
                    purchase_item_html = f"<div class='prod-box'>🧾 Item: <span class='prod-chip'>{html.escape(title)}</span></div>"

            tl.append(
                f"<div class='journey-step {row.etype_class}'>"
                f"  <div class='journey-seq'>{seq_label}</div>"
                f"  <div>"
                f"    <div class='journey-eventline'><span class='ev-icon'>{row.etype_icon}</span>"
                f"      <span class='journey-event'>{event_label}</span>{ad_html}{ad_prod_inline}"
                f"    </div>"
                f"    <div class='journey-page'>{page_label}</div>"
                f"    <div class='journey-meta'>{meta_html}</div>"
                f"    {purchase_item_html}"
                f"  </div>"
                f"</div>"
            )
        tl.append("</div>")
        st.markdown("".join(tl), unsafe_allow_html=True)

        # Export displayed journey
        export_cols = ["journey_name","event_sequence","event_timestamp","event_type",et_col,
                       "page_location","source","ad_name","ad_product","product_title"]
        export_cols = [c for c in export_cols if c in journey_df.columns]
        export_df = journey_df[export_cols].copy()
        st.download_button(
            "Download journey (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"journey_{selected_journey.replace('#','').replace(' ','_')}.csv",
            mime="text/csv",
            key=f"download_journey_{selected_journey}",
        )

    # Close panel
    st.markdown("</div>", unsafe_allow_html=True)



    colA, colB = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"):
            go("home")
    st.markdown("</div>", unsafe_allow_html=True)
    
    
# # ──────────────────────────────────────────────────────────────────────────────
# # Journeys · TripleWhale Paths
# # - Right-side compact filters (Product title, Date presets from data, Rapid repeats, Source, Journey)
# # - Left big scrollable timeline (Purchase on top, earlier steps below)
# # - Product title mapped from sales_file.csv (Order name == journey_name), shown ONLY on Purchase
# # - Non-purchase rows must have Ad Name (null Ad Name allowed only for Purchase)
# # ──────────────────────────────────────────────────────────────────────────────
# from pathlib import Path
# from datetime import date, timedelta
# import pandas as pd
# import html
# import streamlit as st

# # Change path if needed
# SALES_FILE = Path(__file__).with_name("sales_file.csv")


# # ------------------------------- helpers -------------------------------------
# def _first_valid_value(s: pd.Series):
#     for v in s:
#         if pd.notna(v) and str(v).strip():
#             return v
#     return pd.NA

# def _format_timedelta(td) -> str:
#     if pd.isna(td):
#         return "--"
#     secs = int(td.total_seconds())
#     if secs < 60:
#         return f"{secs}s"
#     mins, s = divmod(secs, 60)
#     if mins < 60:
#         return f"{mins}m {s}s"
#     hrs, m = divmod(mins, 60)
#     if hrs < 24:
#         return f"{hrs}h {m}m"
#     days, h = divmod(hrs, 24)
#     return f"{days}d {h}h"

# def _shorten(value: object, limit: int = 48) -> str:
#     if not isinstance(value, str):
#         return ""
#     t = value.strip()
#     return t if len(t) <= limit else t[: max(1, limit - 3)] + "..."

# def _nullish(x) -> bool:
#     if x is None or (isinstance(x, float) and pd.isna(x)):
#         return True
#     s = str(x).strip().lower()
#     return s in ("", "none", "null", "nan", "na", "n/a")

# def _first_of_month(d: date) -> date:
#     return date(d.year, d.month, 1)

# def _prev_month_range(anchor: date):
#     if anchor.month == 1:
#         start = date(anchor.year - 1, 12, 1)
#     else:
#         start = date(anchor.year, anchor.month - 1, 1)
#     end = _first_of_month(anchor) - timedelta(days=1)
#     return start, end


# # Map ONE clean product title per order (Order name == journey_name)
# def load_sales_mapping(file_path: Path) -> pd.DataFrame:
#     """
#     Reads sales_file.csv with:
#       - 'Order name' (journey name)
#       - 'Product title'
#     Returns: DataFrame ['journey_name', 'product_title'] (first non-blank title per order).
#     """
#     if not file_path.exists():
#         return pd.DataFrame(columns=["journey_name", "product_title"])

#     df = pd.read_csv(file_path, dtype=str)
#     cols = {c.strip().lower(): c for c in df.columns}

#     def pick(*names):
#         for nm in names:
#             key = nm.strip().lower()
#             if key in cols:
#                 return cols[key]
#         return None

#     c_order = pick("order name", "order_name", "journey name", "journey_name")
#     c_prod  = pick("product title", "product", "title")

#     if c_order is None or c_prod is None:
#         return pd.DataFrame(columns=["journey_name", "product_title"])

#     df["journey_name"]  = df[c_order].astype(str).str.strip()
#     df["product_title"] = df[c_prod].astype(str).str.strip()

#     # Drop blanks / null-like strings
#     bad = df["product_title"].str.lower().isin(["", "nan", "none", "null", "na", "n/a"])
#     df = df[~bad].copy()

#     # Take FIRST valid title per order
#     grouped = (
#         df.groupby("journey_name", as_index=False)
#           .agg(product_title=("product_title", "first"))
#     )
#     return grouped


# # UI pieces (data-anchored date presets; repeat threshold)
# def _data_presets_ui(min_date: date, max_date: date):
#     """Returns start_date, end_date, label (both dates or None for All data). Anchored to max_date in DATA."""
#     anchor = max_date
#     this_month_start = _first_of_month(anchor)
#     last_month_start, last_month_end = _prev_month_range(anchor)

#     preset = st.selectbox(
#         "Date range (by purchase date in data)",
#         ["Last 7 days", "Last 30 days", "Today (data)", "Yesterday (data)", "This month (data)", "Last month (data)", "Custom range", "All data"],
#         index=0,
#         key="jr_date_preset",
#     )
#     if preset == "Today (data)":
#         start_d, end_d = anchor, anchor
#     elif preset == "Yesterday (data)":
#         start_d = anchor - timedelta(days=1); end_d = start_d
#     elif preset == "Last 7 days":
#         start_d = anchor - timedelta(days=6); end_d = anchor
#     elif preset == "Last 30 days":
#         start_d = anchor - timedelta(days=29); end_d = anchor
#     elif preset == "This month (data)":
#         start_d = this_month_start; end_d = anchor
#     elif preset == "Last month (data)":
#         start_d, end_d = last_month_start, last_month_end
#     elif preset == "Custom range":
#         default_start = max(anchor - timedelta(days=6), min_date)
#         default_end   = anchor
#         dr = st.date_input(
#             "Pick a date range (from data)",
#             value=(default_start, default_end),
#             min_value=min_date,
#             max_value=max_date,
#             key="jr_custom_range",
#         )
#         if isinstance(dr, tuple) and len(dr) == 2:
#             start_d, end_d = dr[0], dr[1]
#         else:
#             start_d, end_d = default_start, default_end
#     else:
#         return None, None, "All data"

#     # clamp to dataset span
#     start_d = max(start_d, min_date)
#     end_d   = min(end_d, max_date)
#     return start_d, end_d, preset


# def _repeat_threshold_ui():
#     """Returns (hide_enabled: bool, threshold_seconds: int)"""
#     hide_rr = st.checkbox("Hide rapid repeats", value=True, key="jr_hide_rr")
#     choice = st.selectbox(
#         "Repeat threshold",
#         ["60 sec", "30 sec", "2 min", "5 min", "Custom (seconds)"],
#         index=0,
#         key="jr_rr_choice",
#     )
#     if choice == "30 sec":
#         seconds = 30
#     elif choice == "60 sec":
#         seconds = 60
#     elif choice == "2 min":
#         seconds = 120
#     elif choice == "5 min":
#         seconds = 300
#     else:
#         seconds = int(st.number_input("Custom seconds", min_value=1, value=60, step=1, key="jr_rr_custom"))
#     return hide_rr, seconds


# # ---------------------------------- MAIN -------------------------------------
# def show_journeys():
#     st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
#     st.markdown("<span class='qm-eyebrow'>Section 3</span>", unsafe_allow_html=True)
#     st.markdown("<h2 class='title-h2'>Journeys &#183; TripleWhale Paths</h2>", unsafe_allow_html=True)

#     data_path = JOURNEYS_FILE
#     if not data_path.exists():
#         st.error(f"Could not find '{data_path.name}'. Place the TripleWhale export alongside app.py.")
#         st.markdown("</div>", unsafe_allow_html=True); return

#     try:
#         journeys_df = load_journey_data(str(data_path))
#     except Exception as exc:
#         st.error(f"Could not load journeys: {exc}")
#         st.markdown("</div>", unsafe_allow_html=True); return

#     if journeys_df.empty:
#         st.info("No journeys remain after loading.")
#         st.markdown("</div>", unsafe_allow_html=True); return

#     # Ensure datetime
#     if not pd.api.types.is_datetime64_any_dtype(journeys_df["event_time"]):
#         journeys_df["event_time"] = pd.to_datetime(journeys_df["event_time"], errors="coerce")

#     # Canonical event type
#     et_col = "event_type_norm" if "event_type_norm" in journeys_df.columns else "event_type"
#     journeys_df[et_col] = journeys_df[et_col].fillna("").astype(str)

#     # Purchase times on full data (for anchors/presets)
#     is_purchase_full = journeys_df[et_col].str.lower().eq("purchase")
#     purchase_times_full = (
#         journeys_df.loc[is_purchase_full].groupby("journey_name")["event_time"].min().rename("first_purchase_time")
#     )
#     if purchase_times_full.empty:
#         st.info("No purchase-anchored journeys found.")
#         st.markdown("</div>", unsafe_allow_html=True); return

#     min_d = purchase_times_full.min().date()
#     max_d = purchase_times_full.max().date()

#     # Layout: left (timeline) | right (filter card)
#     left, right = st.columns([2.4, 1.0], gap="large")

#     # ---------- RIGHT: filter card (top area) ----------
#     with right:
#         st.markdown("""
#             <style>
#               .filter-card { padding:14px; border:1px solid #e8ecf2; border-radius:14px; background:#fbfcff;
#                              box-shadow:0 1px 2px rgba(0,0,0,.04); }
#               .filter-card h4 { margin:0 0 8px 0; font-size:1rem; }
#               .tiny-note { color:#6b7b8f; font-size:.82rem; margin-top:4px; }
#             </style>
#         """, unsafe_allow_html=True)

#         st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
#         st.markdown("#### Filters", unsafe_allow_html=True)

#         # Date presets (data-anchored)
#         st.markdown("**Date range**")
#         start_date, end_date, preset = _data_presets_ui(min_d, max_d)

#         # Rapid repeats
#         st.markdown("**Rapid repeats**")
#         hide_rr, rr_seconds = _repeat_threshold_ui()
#         st.markdown("<div class='tiny-note'>Removes steps closer than the threshold to the previous step (keeps earlier; Purchase always kept).</div>", unsafe_allow_html=True)

#     # Apply DATE FILTER to journeys by first purchase time
#     if start_date and end_date:
#         start_ts = pd.Timestamp(start_date)
#         end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
#         keep_journeys = purchase_times_full[(purchase_times_full >= start_ts) & (purchase_times_full <= end_ts)]
#     else:
#         keep_journeys = purchase_times_full

#     if keep_journeys.empty:
#         with left:
#             st.info(f"No journeys with a purchase in the selected range ({preset}).")
#         st.markdown("</div>", unsafe_allow_html=True); return

#     # Keep journeys in range & trim to first purchase
#     journeys_df = journeys_df.merge(keep_journeys.rename("first_purchase_time"), on="journey_name", how="inner")
#     journeys_df = journeys_df[journeys_df["event_time"] <= journeys_df["first_purchase_time"]].copy()

#     # Non-purchase rows must have Ad Name
#     ad_ok = ~journeys_df["ad_name"].map(_nullish) if "ad_name" in journeys_df.columns else pd.Series(False, index=journeys_df.index)
#     is_purchase_now = journeys_df[et_col].str.lower().eq("purchase")
#     journeys_df = journeys_df[is_purchase_now | ad_ok].copy()

#     # Ensure at least one non-purchase row per journey
#     non_purchase_counts = journeys_df.loc[~is_purchase_now].groupby("journey_name").size().rename("np_rows")
#     journeys_df = journeys_df.merge(non_purchase_counts, on="journey_name", how="left")
#     journeys_df = journeys_df[journeys_df["np_rows"].fillna(0) > 0].copy()
#     if journeys_df.empty:
#         with left:
#             st.info("No journeys remain after applying ad-name rule and date range.")
#         st.markdown("</div>", unsafe_allow_html=True); return

#     # Attach single product title (ONLY shows on purchase row)
#     sales_map = load_sales_mapping(SALES_FILE)
#     if not sales_map.empty:
#         journeys_df = journeys_df.merge(sales_map, on="journey_name", how="left")
#     else:
#         journeys_df["product_title"] = ""

#     journeys_df = journeys_df.sort_values(["journey_name", "event_time", "event_sequence"]).reset_index(drop=True)

#     # ---------- SUMMARY for right-hand picker & left header ----------
#     summary_df = (
#         journeys_df.groupby("journey_name")
#         .agg(
#             steps=("event_sequence", "count"),
#             first_event=("event_time", "min"),
#             last_event=("event_time", "max"),
#             purchase_time=("first_purchase_time", "first"),
#         )
#         .reset_index()
#     )
#     ad_np = (
#         journeys_df.loc[journeys_df[et_col].str.lower().ne("purchase")]
#         .groupby("journey_name")["ad_name"].agg(_first_valid_value).rename("ad_name_np")
#     )
#     summary_df = summary_df.merge(ad_np, on="journey_name", how="left")
#     if "source" in journeys_df.columns:
#         summary_df = summary_df.merge(
#             journeys_df.groupby("journey_name")["source"].agg(_first_valid_value).rename("source"),
#             on="journey_name", how="left",
#         )
#     if "page_location" in journeys_df.columns:
#         summary_df = summary_df.merge(
#             journeys_df.groupby("journey_name")["page_location"].agg(_first_valid_value).rename("first_page"),
#             on="journey_name", how="left",
#         )
#     summary_df["journey_span"] = summary_df["last_event"] - summary_df["first_event"]
#     summary_df = summary_df.sort_values(["purchase_time", "first_event"], ascending=[False, False]).reset_index(drop=True)

#     st.caption(
#         f"{len(summary_df)} journeys (purchase in {preset}). "
#         "Non-purchase steps require Ad Name; Purchase may not. Products show only on Purchase."
#     )

#     # Product options (from purchase rows after date/ad filters)
#     purchase_titles = (
#         journeys_df.loc[journeys_df[et_col].str.lower().eq("purchase"), "product_title"]
#         .dropna().astype(str).str.strip()
#     )
#     product_options = sorted({t for t in purchase_titles if t})

#     # ---------- RIGHT: refine (Product title at top, then Source, then Journey) ----------
#     with right:
#         st.markdown("**Refine**")

#         if product_options:
#             product_filter = st.multiselect("Product title (Item)", product_options, key="jr_product_filter")
#         else:
#             product_filter = []
#             st.caption("No product titles found in the selected date range.")

#         source_filter = None
#         if "source" in summary_df.columns:
#             source_options = sorted({str(s).strip() for s in summary_df["source"].dropna() if str(s).strip()})
#             source_filter = st.multiselect("Source", source_options, key="journeys_filter_source")

#     # Apply product filter + source filter to the journey list
#     filtered_df = summary_df.copy()

#     if product_filter:
#         # Journey -> product title (from purchase only)
#         journey_products = journeys_df.loc[
#             journeys_df[et_col].str.lower().eq("purchase"),
#             ["journey_name", "product_title"]
#         ].dropna()
#         journey_products["product_title"] = journey_products["product_title"].astype(str).str.strip()
#         allowed = set(journey_products[journey_products["product_title"].isin(product_filter)]["journey_name"].unique())
#         filtered_df = filtered_df[filtered_df["journey_name"].isin(allowed)]

#     if source_filter:
#         filtered_df = filtered_df[filtered_df["source"].isin(source_filter)]

#     if filtered_df.empty:
#         with left:
#             msg = "No journeys match the current filters."
#             if product_filter:
#                 msg += " Try clearing the product title filter."
#             st.info(msg)
#             with st.expander("All journeys (summary)"):
#                 _view = summary_df[["journey_name", "source", "ad_name_np", "steps", "journey_span"]].copy()
#                 _view["journey_span"] = _view["journey_span"].apply(_format_timedelta)
#                 _view = _view.rename(columns={
#                     "journey_name": "Journey", "source": "Source", "ad_name_np": "Ad Name",
#                     "steps": "Steps", "journey_span": "Journey Span",
#                 })
#                 st.dataframe(_view, use_container_width=True, hide_index=True)
#         st.markdown("</div>", unsafe_allow_html=True);  # close filter-card
#         st.markdown("</div>", unsafe_allow_html=True);  # close panel
#         return

#     # Journey picker (post-filters)
#     label_map: dict[str, str] = {}
#     for row in filtered_df.itertuples():
#         pieces = [row.journey_name]
#         if "source" in filtered_df.columns and isinstance(row.source, str) and row.source.strip():
#             pieces.append(_shorten(row.source, 36))
#         ad_label = _shorten(row.ad_name_np, 48) if isinstance(row.ad_name_np, str) else ""
#         if ad_label:
#             pieces.append(ad_label)
#         pieces.append(f'{int(row.steps)} steps')
#         label_map[row.journey_name] = " | ".join([p for p in pieces if p])

#     options = filtered_df["journey_name"].tolist()
#     previous_choice = st.session_state.get("journeys_selected")
#     default_index = options.index(previous_choice) if previous_choice in options else 0

#     with right:
#         selected_journey = st.selectbox(
#             "Journey",
#             options,
#             index=default_index,
#             format_func=lambda j: label_map.get(j, j),
#             key="journeys_selected",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)  # close filter-card

#     # ---------- LEFT: header + legend + timeline ----------
#     with left:
#         st.markdown("""
#             <style>
#             .journey-hero { display:flex; align-items:center; gap:14px; margin:6px 0 10px 0; }
#             .ad-hero { font-weight:800; font-size:1.06rem; padding:8px 12px; border-radius:16px;
#                        border:1px solid rgba(0,0,0,0.06); background:linear-gradient(180deg,#fff, #fafafa);
#                        box-shadow:0 1px 2px rgba(0,0,0,0.06); }
#             .hero-pills { display:flex; gap:8px; flex-wrap:wrap; }
#             .hero-pill { font-size:.86rem; padding:5px 12px; border-radius:999px; background:#f5f7fa; border:1px solid #edf0f3; }
#             .journey-shell { margin-top:6px; }
#             .journey-scroll { max-height:640px; overflow-y:auto; padding:10px 10px; background:#fff;
#                               border:1px solid #e8ecf2; border-radius:14px; box-shadow:inset 0 1px 3px rgba(0,0,0,.05); }
#             .journey-step { display:grid; grid-template-columns: 52px 1fr; gap:12px; padding:10px; margin:8px 0;
#                             background:#ffffff; border:1px solid #eef1f6; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
#             .journey-seq { display:flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:12px;
#                            background:#f7f9fc; font-weight:700; font-size:.95rem; border:1px solid #eef1f6; }
#             .ev-icon { margin-right:8px; }
#             .journey-eventline { display:flex; align-items:center; gap:8px; }
#             .journey-event { font-weight:800; }
#             .ad-badge { margin-left:auto; font-weight:800; padding:4px 10px; border-radius:10px; background:#fff5e6; border:1px solid #ffe2b8; }
#             .journey-page { color:#52637a; font-size:.94rem; margin:2px 0 2px 26px; }
#             .journey-meta { color:#66798f; font-size:.86rem; display:flex; gap:12px; margin-left:26px; }
#             .journey-empty { opacity:.55; }
#             .prod-box { margin:6px 0 0 26px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
#             .prod-chip { font-size:.86rem; padding:4px 10px; border:1px solid #e7eaf0; background:#fbfcfe; border-radius:999px; }
#             .ev-view .journey-seq { background:#f2f7ff; }
#             .ev-click .journey-seq { background:#f6f3ff; }
#             .ev-cart .journey-seq { background:#fff6f0; }
#             .ev-checkout .journey-seq { background:#f1fff4; }
#             .ev-purchase { border:1px solid #d2f5e5; background:#f7fffb; }
#             .ev-purchase .journey-seq { background:#f0fff8; border:1px solid #d2f5e5; }
#             .legend { display:flex; gap:12px; color:#567; font-size:.85rem; margin:6px 0 12px 0; }
#             .legend span { background:#f6f8fb; border:1px solid #eef1f6; padding:6px 10px; border-radius:999px; }
#             </style>
#         """, unsafe_allow_html=True)

#         # Header chips
#         summary_row = summary_df[summary_df["journey_name"] == selected_journey].iloc[0]
#         ad_hero = html.escape(_shorten(summary_row.ad_name_np, 80) or "--")
#         steps_label = f'{int(summary_row.steps)} steps'
#         span_label = _format_timedelta(summary_row["last_event"] - summary_row["first_event"])

#         st.markdown(
#             f"<div class='journey-hero'>"
#             f"<div class='ad-hero'>Ad • <span>{ad_hero}</span></div>"
#             f"<div class='hero-pills'>"
#             f"<div class='hero-pill'>Journey Span: {html.escape(span_label)}</div>"
#             f"<div class='hero-pill'>{html.escape(steps_label)}</div>"
#             f"</div>"
#             f"</div>",
#             unsafe_allow_html=True,
#         )
#         st.markdown("<div class='legend'><span>👀 View</span><span>🖱️ Click</span><span>🛒 Add to Cart</span><span>💳 Checkout</span><span>🛍️ Purchase</span></div>", unsafe_allow_html=True)

#         # Build timeline for selected journey (chronological → optional repeats → display with Purchase on top)
#         base_df = journeys_df[journeys_df["journey_name"] == selected_journey].copy().reset_index(drop=True)
#         jp_mask = base_df[et_col].str.lower().eq("purchase")
#         fp_idx = base_df.index[jp_mask][0]
#         base_df = base_df.loc[:fp_idx].copy()  # chronological up to first purchase

#         if hide_rr and rr_seconds > 0:
#             base_df["delta_prev_chrono"] = base_df["event_time"].diff()
#             keep_bursts = (
#                 (base_df.index == 0)
#                 | (base_df[et_col].str.lower().eq("purchase"))
#                 | (base_df["delta_prev_chrono"].isna())
#                 | (base_df["delta_prev_chrono"] >= pd.Timedelta(seconds=int(rr_seconds)))
#             )
#             base_df = base_df[keep_bursts].copy().reset_index(drop=True)

#         jp_mask2 = base_df[et_col].str.lower().eq("purchase")
#         fp_idx2 = base_df.index[jp_mask2][0]
#         prior = base_df.iloc[:fp_idx2] if fp_idx2 > 0 else base_df.iloc[0:0]
#         journey_df = pd.concat([base_df.iloc[[fp_idx2]], prior.iloc[::-1]], ignore_index=True)

#         # Display fields
#         def _disp(s):
#             return s.strip() if isinstance(s, str) and s.strip() else "--"

#         journey_df["event_display"] = journey_df["event_type"] if "event_type" in journey_df else journey_df[et_col]
#         journey_df["event_display"] = journey_df["event_display"].apply(_disp)
#         journey_df["page_display"] = journey_df.get("page_location", pd.Series([""] * len(journey_df))).apply(_disp)

#         if pd.api.types.is_datetime64_any_dtype(journey_df["event_time"]):
#             journey_df["event_time_display"] = journey_df["event_time"].dt.strftime("%b %d, %Y - %I:%M:%S %p")
#         else:
#             journey_df["event_time_display"] = journey_df.get("event_timestamp", pd.Series(["--"] * len(journey_df))).apply(_disp)

#         journey_df["delta_prev"] = journey_df["event_time"] - journey_df["event_time"].shift(1)
#         journey_df.loc[0, "delta_prev"] = pd.Timedelta(0)
#         journey_df["delta_prev_label"] = journey_df["delta_prev"].abs().apply(_format_timedelta)

#         journey_df["source_display"] = journey_df.get("source", pd.Series([""] * len(journey_df))).apply(lambda x: _shorten(x, 32))
#         journey_df["ad_display"] = journey_df["ad_name"].apply(lambda x: _shorten(x, 48) if isinstance(x, str) and x.strip() else "")

#         def _etype_class_and_icon(s: str):
#             s = (s or "").lower()
#             if "purchase" in s: return "ev-purchase", "🛍️"
#             if "add_to_cart" in s or "addtocart" in s or "cart" in s: return "ev-cart", "🛒"
#             if "checkout" in s or "begin_checkout" in s: return "ev-checkout", "💳"
#             if "click" in s or "ad_click" in s: return "ev-click", "🖱️"
#             if "search" in s: return "ev-search", "🔎"
#             if "view" in s or "page_view" in s or "view_content" in s or "view_item" in s: return "ev-view", "👀"
#             return "ev-other", "•"

#         journey_df["etype_class"], journey_df["etype_icon"] = zip(*journey_df[et_col].apply(_etype_class_and_icon))

#         # Render timeline
#         tl = ["<div class='journey-shell'><div class='journey-scroll'>"]
#         for row in journey_df.itertuples(index=False):
#             seq_label  = html.escape(str(row.event_sequence)) if pd.notna(row.event_sequence) else "--"
#             event_label = html.escape(row.event_display) if isinstance(row.event_display, str) else "--"
#             page_label = html.escape(row.page_display) if isinstance(row.page_display, str) else "--"

#             meta_bits = []
#             if row.event_time_display and row.event_time_display != "--":
#                 meta_bits.append(f"<span><strong>{html.escape(row.event_time_display)}</strong></span>")
#             if row.delta_prev_label not in ("--", "0s"):
#                 meta_bits.append(f"<span>+{html.escape(row.delta_prev_label)} since prior</span>")
#             if isinstance(row.source_display, str) and row.source_display.strip():
#                 meta_bits.append(f"<span>{html.escape(row.source_display)}</span>")
#             meta_html = "".join(meta_bits) if meta_bits else "<span class='journey-empty'>No metadata</span>"

#             show_ad_badge = (("purchase" not in (row.event_display or "").lower()) and isinstance(row.ad_display, str) and row.ad_display.strip())
#             ad_html = f"<span class='ad-badge'><strong>{html.escape(row.ad_display)}</strong></span>" if show_ad_badge else ""

#             # ONE product title ONLY on purchase step
#             prod_html = ""
#             if "purchase" in (row.event_display or "").lower():
#                 title = getattr(row, "product_title", "")
#                 title = title.strip() if isinstance(title, str) else ""
#                 if title:
#                     prod_html = f"<div class='prod-box'>🧾 Item: <span class='prod-chip'>{html.escape(title)}</span></div>"

#             tl.append(
#                 f"<div class='journey-step {row.etype_class}'>"
#                 f"  <div class='journey-seq'>{seq_label}</div>"
#                 f"  <div>"
#                 f"    <div class='journey-eventline'><span class='ev-icon'>{row.etype_icon}</span>"
#                 f"      <span class='journey-event'>{event_label}</span>{ad_html}"
#                 f"    </div>"
#                 f"    <div class='journey-page'>{page_label}</div>"
#                 f"    <div class='journey-meta'>{meta_html}</div>"
#                 f"    {prod_html}"
#                 f"  </div>"
#                 f"</div>"
#             )
#         tl.append("</div></div>")
#         st.markdown("".join(tl), unsafe_allow_html=True)

#         # Export displayed journey
#         export_cols = [
#             "journey_name", "event_sequence", "event_timestamp", "event_type", et_col,
#             "page_location", "source", "ad_name", "product_title"
#         ]
#         export_cols = [c for c in export_cols if c in journey_df.columns]
#         export_df = journey_df[export_cols].copy()
#         st.download_button(
#             "Download journey (CSV)",
#             data=export_df.to_csv(index=False).encode("utf-8"),
#             file_name=f"journey_{selected_journey.replace('#', '').replace(' ', '_')}.csv",
#             mime="text/csv",
#             key=f"download_journey_{selected_journey}",
#         )

#         # Summary of filtered journeys
#         matches_df = filtered_df[["journey_name", "ad_name_np", "steps", "journey_span", "source"]].copy()
#         matches_df["journey_span"] = matches_df["journey_span"].apply(_format_timedelta)
#         matches_df = matches_df.rename(columns={
#             "journey_name": "Journey", "ad_name_np": "Ad Name", "steps": "Steps",
#             "journey_span": "Journey Span", "source": "Source",
#         })
#         with st.expander("Filtered journeys (summary)"):
#             st.dataframe(matches_df, use_container_width=True, hide_index=True)

#     # close outer panel
#     st.markdown("</div>", unsafe_allow_html=True)


# # ──────────────────────────────────────────────────────────────────────────────
# # SECTION 3 — OPTIMIZER (placeholder UI shell)
# # ──────────────────────────────────────────────────────────────────────────────
# def show_optimizer():
#     """Section 3 — Marketing Inputs (budgeted).
#     Uses means from results; unit cost converts spend→units.
#     Contextual variables (no budget) come from Section 4.
#     """
#     # ---------- helpers ----------
#     def _beta_for(row: pd.Series, var: str) -> float:
#         for key in [f"Weighted_Beta_{var}", f"Selected_Beta_{var}", f"Beta_{var}"]:
#             if key in row and pd.notna(row[key]):
#                 try: return float(row[key])
#                 except Exception: pass
#         return 0.0

#     def _mean_for(row: pd.Series, var: str) -> float:
#         for key in [f"Mean_{var}", f"Selected_Mean_{var}"]:
#             if key in row and pd.notna(row[key]):
#                 try: return float(row[key])
#                 except Exception: pass
#         return 0.0

#     def _intercept_for(row: pd.Series) -> float:
#         for key in ["Weighted_B0", "Selected_B0 (Original)", "Selected_B0", "B0 (Original)"]:
#             if key in row and pd.notna(row[key]):
#                 try: return float(row[key])
#                 except Exception: pass
#         return 0.0

#     def _combo_label(row: pd.Series, grouping_keys: list[str]) -> str:
#         parts = [f"{k}:{row[k]}" for k in grouping_keys if k in row.index]
#         return " | ".join(parts) if parts else "(combo)"

#     def _build_grid(row: pd.Series, media_vars: list[str], unit_costs: dict, existing: pd.DataFrame | None):
#         # Columns: Variable, MEAN, CPU, SPEND (editable)
#         data = []
#         for v in media_vars:
#             mean_x = _mean_for(row, v)
#             cpu = float(unit_costs.get(v, 0.0))
#             spend_default = 0.0
#             if existing is not None and not existing.empty:
#                 try:
#                     spend_default = float(existing.loc[existing["Variable"] == v, "SPEND"].values[0])
#                 except Exception:
#                     pass
#             data.append({"Variable": v, "MEAN": mean_x, "CPU": cpu, "SPEND": spend_default})
#         return pd.DataFrame(data)

#     def _evaluate(row: pd.Series, grid_df: pd.DataFrame, contextual_levels: dict | None) -> dict:
#         # baseline from means across all features
#         b0 = _intercept_for(row)
#         baseline = b0
#         beta_feats = []
#         for c in row.index:
#             if isinstance(c, str) and (c.startswith("Weighted_Beta_") or c.startswith("Selected_Beta_") or c.startswith("Beta_")):
#                 beta_feats.append(c.split("Beta_")[-1])
#         for feat in sorted(set(beta_feats)):
#             baseline += _beta_for(row, feat) * _mean_for(row, feat)

#         # contextual deltas (no budget): x* - mean
#         ctx_contrib = 0.0
#         if contextual_levels:
#             for k, xstar in contextual_levels.items():
#                 if k not in list(grid_df["Variable"]):  # only non-media here
#                     beta = _beta_for(row, k)
#                     if beta != 0:
#                         dx = float(xstar) - _mean_for(row, k)
#                         ctx_contrib += beta * dx

#         # marketing deltas via spend→units
#         used_budget = float(grid_df["SPEND"].fillna(0).sum())
#         mkt_rows, mkt_contrib = [], 0.0
#         for _, r in grid_df.iterrows():
#             v = r["Variable"]; cpu = float(r["CPU"]) if pd.notna(r["CPU"]) else 0.0
#             spend = float(r["SPEND"]) if pd.notna(r["SPEND"]) else 0.0
#             mean_x = float(r["MEAN"]) if pd.notna(r["MEAN"]) else 0.0
#             units = 0.0 if cpu <= 0 else spend / cpu
#             dx = units - mean_x
#             beta = _beta_for(row, v)
#             contrib = beta * dx
#             mkt_contrib += contrib
#             mkt_rows.append({"Variable": v, "MEAN": mean_x, "CPU": cpu, "SPEND": spend,
#                              "Units": units, "Delta_X": dx, "Beta": beta, "Contribution": contrib})

#         predicted = baseline + ctx_contrib + mkt_contrib
#         return {
#             "baseline": baseline,
#             "ctx_contrib": ctx_contrib,
#             "mkt_contrib": mkt_contrib,
#             "predicted": predicted,
#             "used_budget": used_budget,
#             "per_var": pd.DataFrame(mkt_rows),
#         }

#     def _baseline_spend(grid_df: pd.DataFrame) -> pd.DataFrame:
#         g = grid_df.copy()
#         g["SPEND"] = g.apply(lambda r: float(r["MEAN"]) * float(r["CPU"]) if (pd.notna(r["MEAN"]) and pd.notna(r["CPU"])) else 0.0, axis=1)
#         return g

#     def _zero_spend(grid_df: pd.DataFrame) -> pd.DataFrame:
#         g = grid_df.copy(); g["SPEND"] = 0.0; return g

#     def _greedy_max(row: pd.Series, grid_df: pd.DataFrame, budget: float) -> pd.DataFrame:
#         # allocate all budget to best beta-per-dollar (simple stub)
#         g = _zero_spend(grid_df)
#         eff = []
#         for _, r in g.iterrows():
#             v = r["Variable"]; cpu = float(r["CPU"]) if pd.notna(r["CPU"]) else 0.0
#             b = _beta_for(row, v)
#             eff.append((v, (b / cpu) if cpu > 0 else -np.inf))
#         eff.sort(key=lambda x: x[1], reverse=True)
#         if eff and np.isfinite(eff[0][1]) and eff[0][1] > 0:
#             vbest = eff[0][0]
#             idx = g.index[g["Variable"] == vbest][0]
#             g.at[idx, "SPEND"] = float(budget)
#         return g

#     # ---------- UI ----------
#     st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
#     st.markdown("<span class='qm-eyebrow'>Section 3</span>", unsafe_allow_html=True)
#     st.markdown("<h2>Marketing Inputs</h2>", unsafe_allow_html=True)
#     st.caption("Budget applies to marketing variables only. Contextual changes come from the Contextual Inputs page.")

#     saved_models = st.session_state.get("saved_models")
#     grouping_keys = st.session_state.get("grouping_keys", [])
#     media_vars = st.session_state.get("optimizer_media", [])
#     unit_costs = st.session_state.get("optimizer_unit_costs", {})
#     contextual_levels = st.session_state.get("contextual_levels", {})

#     if saved_models is None or saved_models.empty:
#         st.warning("No consolidated models in session. Run Section 1 first.")
#         colA, _ = st.columns([1,4])
#         with colA:
#             if st.button("← Back to Home"): go("home")
#         st.markdown("</div>", unsafe_allow_html=True)
#         return

#     if not grouping_keys:
#         blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
#         grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
#     labels = [" | ".join([f"{k}:{saved_models.iloc[i][k]}" for k in grouping_keys]) for i in range(len(saved_models))]

#     st.markdown("### Choose Combination")
#     combo_idx = st.selectbox("Combination", options=list(range(len(saved_models))), format_func=lambda i: labels[i], key="combo_select_a")
#     row = saved_models.iloc[combo_idx]

#     grid_key = f"mkt_grid_{combo_idx}"
#     existing = st.session_state.get(grid_key)
#     planner_grid = _build_grid(row, media_vars, unit_costs, existing)

#     left, right = st.columns([3,2])
#     with left:
#         st.markdown("### Marketing Inputs Table")
#         colcfg = {
#             "Variable": st.column_config.TextColumn(disabled=True),
#             "MEAN":     st.column_config.NumberColumn(format="%.3f", disabled=True),
#             "CPU":      st.column_config.NumberColumn(help="Currency per 1 unit", format="%.6f", disabled=True),
#             "SPEND":    st.column_config.NumberColumn(format="%.2f"),
#         }
#         planner_grid = st.data_editor(planner_grid, use_container_width=True, hide_index=True,
#                                       column_config=colcfg, key=f"mkt_grid_edit_{combo_idx}")

#     with right:
#         st.markdown("### Budget & Actions")
#         total_budget = st.number_input("Total budget", value=10000.0, step=100.0, key=f"mkt_budget_{combo_idx}")
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             if st.button("Baseline Spend", key=f"btn_base_{combo_idx}"):
#                 planner_grid = _baseline_spend(planner_grid)
#         with c2:
#             if st.button("Zero Spend", key=f"btn_zero_{combo_idx}"):
#                 planner_grid = _zero_spend(planner_grid)
#         with c3:
#             if st.button("Max within Budget", key=f"btn_max_{combo_idx}"):
#                 planner_grid = _greedy_max(row, planner_grid, total_budget)
#         run = st.button("Evaluate Scenario", type="primary", key=f"btn_eval_mkt_{combo_idx}")
#         if st.button("Go to Contextual Inputs →", key=f"btn_go_ctx_{combo_idx}"):
#             go("contextual")

#     st.session_state[grid_key] = planner_grid

#     if run:
#         res = _evaluate(row, planner_grid, contextual_levels)
#         m1, m2, m3 = st.columns(3)
#         with m1: st.metric("Projected Volume (Pred)", f"{res['predicted']:,.2f}",
#                            delta=f"+{(res['predicted']-res['baseline']):,.2f}")
#         with m2: st.metric("Baseline Volume (Mean)", f"{res['baseline']:,.2f}")
#         with m3: st.metric("Budget Used", f"{res['used_budget']:,.2f}")
#         tab1, tab2 = st.tabs(["Per-variable", "Contribs"])
#         with tab1: st.dataframe(res["per_var"], use_container_width=True, hide_index=True)
#         with tab2: st.write({"Contextual": res["ctx_contrib"], "Marketing": res["mkt_contrib"]})

#     colA, _ = st.columns([1,4])
#     with colA:
#         if st.button("← Back to Home"): go("home")
#     st.markdown("</div>", unsafe_allow_html=True)





#     # Build combo labels
#     if not grouping_keys:
#         blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
#         grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
#     combo_labels = []
#     for _, row in saved_models.iterrows():
#         parts = [f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns]
#         combo_labels.append(" | ".join(parts) if parts else f"row_{_}")

#     # —— Layout: left inputs / right budget-bounds — better space mgmt ——
#     left, right = st.columns([3, 2])

#     with left:
#         st.markdown("### Combinations")
#         sel_indices = st.multiselect(
#             "Select combinations",
#             options=list(range(len(saved_models))),
#             default=list(range(len(saved_models)))[: min(5, len(saved_models))],
#             format_func=lambda i: combo_labels[i]
#         )
#         if media_vars:
#             st.markdown("### Media variables")
#             st.caption(", ".join(media_vars))
#         else:
#             st.info("No media variables configured. Go to Section 2 to set them.")

#     with right:
#         st.markdown("### Budget & Bounds")
#         total_budget = st.number_input("Total budget", value=10000.0, step=100.0)
#         lb = st.number_input("Lower bound per media var", value=0.0, step=10.0)
#         ub = st.number_input("Upper bound per media var", value=1e9, step=1000.0)
#         run = st.button("▶️ Run IPOPT (skeleton)", type="primary")

#     if run:
#         unit_costs = st.session_state.get("optimizer_unit_costs", {})
#         if not sel_indices:
#             st.error("Pick at least one combination.")
#         elif not media_vars:
#             st.error("Define media variables in Section 2.")
#         elif not unit_costs or any(v not in unit_costs or unit_costs[v] <= 0 for v in media_vars):
#             missing = [v for v in media_vars if v not in unit_costs or unit_costs[v] <= 0]
#             st.error("Set positive unit costs for: " + ", ".join(missing))
#         else:
#             n_vars = len(media_vars)
#             per_var_spend = float(np.clip(total_budget / max(n_vars, 1), lb, ub))
#             allocation_rows = []
#             summary_rows = []

#             for i in sel_indices:
#                 row = saved_models.iloc[i]
#                 combo_id = " | ".join([f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns])
#                 alloc_map = {v: per_var_spend for v in media_vars}
#                 res = predict_from_allocation(row, media_vars, alloc_map, unit_costs)

#                 for v in media_vars:
#                     allocation_rows.append({
#                         "Combination": combo_id,
#                         "Variable": v,
#                         "Spend": alloc_map[v],
#                         "UnitCost": unit_costs.get(v, np.nan),
#                         "Delta_X": res["per_var_dx"][v],
#                         "Beta": _beta_for(row, v),
#                         "Contribution": _beta_for(row, v) * res["per_var_dx"][v],
#                     })

#                 summary_rows.append({
#                     "Combination": combo_id,
#                     "Budget": total_budget,
#                     "Baseline_Y": res["baseline_y"],
#                     "Predicted_Y": res["predicted_y"],
#                     "Lift_Y": res["lift_y"],
#                     "Lift_per_$": (res["lift_y"] / total_budget) if total_budget else np.nan,
#                 })

#             alloc_df = pd.DataFrame(allocation_rows)
#             summ_df = pd.DataFrame(summary_rows)
#             st.session_state["optimizer_allocation"] = alloc_df
#             st.session_state["optimizer_summary"] = summ_df

#             st.success("Allocation evaluated using unit costs and model betas (linear response). IPOPT hook next.")
#             tab1, tab2 = st.tabs(["Allocation breakdown", "Summary"])
#             with tab1:
#                 st.dataframe(alloc_df, use_container_width=True, hide_index=True)
#                 st.download_button("📥 Download Allocation Breakdown", data=alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation_breakdown.csv", mime="text/csv")
#             with tab2:
#                 st.dataframe(summ_df, use_container_width=True, hide_index=True)
#                 st.download_button("📥 Download Summary", data=summ_df.to_csv(index=False).encode("utf-8"), file_name="allocation_summary.csv", mime="text/csv")

#     colA, colB = st.columns([1,4])
#     with colA:
#         if st.button("← Back to Home"):
#             go("home")
#     st.markdown("</div>", unsafe_allow_html=True)


#     # Build combo labels
#     if not grouping_keys:
#         # fallback: any non-weighted/non-selected metric columns at start
#         blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
#         grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
#     combo_labels = []
#     for _, row in saved_models.iterrows():
#         parts = [f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns]
#         combo_labels.append(" | ".join(parts) if parts else f"row_{_}")

#     # Selection UI
#     st.markdown("**Select combinations to optimize**")
#     sel_indices = st.multiselect(
#         "Combinations",
#         options=list(range(len(saved_models))),
#         default=list(range(len(saved_models)))[: min(5, len(saved_models))],
#         format_func=lambda i: combo_labels[i]
#     )

#     st.markdown("**Budget**")
#     total_budget = st.number_input("Total budget (currency units)", value=10000.0, step=100.0)

#     if not media_vars:
#         st.info("No media variables configured. Go to Section 2 to set media/control variables.")
#     else:
#         st.markdown("**Media variables (decision investments)**")
#         st.caption(", ".join(media_vars))

#     # Placeholder bounds grid (future: per combo & per var bounds)
#     st.markdown("**Bounds (skeleton)**")
#     lb = st.number_input("Lower bound per media var", value=0.0, step=10.0)
#     ub = st.number_input("Upper bound per media var", value=1e9, step=1000.0)

#     run = st.button("▶️ Run IPOPT (skeleton)", type="primary")

#     if run:
#         unit_costs = st.session_state.get("optimizer_unit_costs", {})
#         if not sel_indices:
#             st.error("Pick at least one combination.")
#         elif not media_vars:
#             st.error("Define media variables in Section 2.")
#         elif not unit_costs or any(v not in unit_costs or unit_costs[v] <= 0 for v in media_vars):
#             missing = [v for v in media_vars if v not in unit_costs or unit_costs[v] <= 0]
#             st.error("Set positive unit costs for: " + ", ".join(missing))
#         else:
#             n_vars = len(media_vars)
#             per_var_spend = float(np.clip(total_budget / max(n_vars, 1), lb, ub))
#             allocation_rows = []
#             summary_rows = []

#             for i in sel_indices:
#                 row = saved_models.iloc[i]
#                 combo_id = " | ".join([f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns])
#                 alloc_map = {v: per_var_spend for v in media_vars}
#                 res = predict_from_allocation(row, media_vars, alloc_map, unit_costs)

#                 for v in media_vars:
#                     allocation_rows.append({
#                         "Combination": combo_id,
#                         "Variable": v,
#                         "Spend": alloc_map[v],
#                         "UnitCost": unit_costs.get(v, np.nan),
#                         "Delta_X": res["per_var_dx"][v],
#                         "Beta": _beta_for(row, v),
#                         "Contribution": _beta_for(row, v) * res["per_var_dx"][v],
#                     })

#                 summary_rows.append({
#                     "Combination": combo_id,
#                     "Budget": total_budget,
#                     "Baseline_Y": res["baseline_y"],
#                     "Predicted_Y": res["predicted_y"],
#                     "Lift_Y": res["lift_y"],
#                     "Lift_per_$": (res["lift_y"] / total_budget) if total_budget else np.nan,
#                 })

#             alloc_df = pd.DataFrame(allocation_rows)
#             summ_df = pd.DataFrame(summary_rows)
#             st.session_state["optimizer_allocation"] = alloc_df
#             st.session_state["optimizer_summary"] = summ_df

#             st.success("Allocation evaluated using unit costs and model betas (linear response). IPOPT hook next.")
#             st.subheader("Per-variable allocation breakdown")
#             st.dataframe(alloc_df, use_container_width=True, hide_index=True)
#             st.download_button("📥 Download Allocation Breakdown", data=alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation_breakdown.csv", mime="text/csv")

#             st.subheader("Combination summary")
#             st.dataframe(summ_df, use_container_width=True, hide_index=True)
#             st.download_button("📥 Download Summary", data=summ_df.to_csv(index=False).encode("utf-8"), file_name="allocation_summary.csv", mime="text/csv")

#     colA, colB = st.columns([1,4])
#     with colA:
#         if st.button("← Back to Home"):
#             go("home")
#     st.markdown("</div>", unsafe_allow_html=True)


# # ──────────────────────────────────────────────────────────────────────────────
# # HOME — Three Cards (Consolidator, Optimizer Configurer, Optimizer)
# # ──────────────────────────────────────────────────────────────────────────────

def home_cards():
    st.markdown("<div class='qm-hero'>QuantMatrix - Model & Optimizer Suite</div>", unsafe_allow_html=True)
    st.markdown("<div class='qm-sub'>Section 1: Model Consolidator &bull; Section 2: Insights &bull; Section 3: Journeys &bull; Section 4: Optimizer Configurer &bull; Section 5: Marketing Inputs</div>", unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class='qm-card qm-accent-yellow'>
              <div class='qm-eyebrow'>Section 1</div>
              <div class='qm-title'>Model Consolidator</div>
              <div class='qm-copy'>Upload your model results and collapse to a single model per combination via a winner model or a MAPE-weighted ensemble.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Open Consolidator", key="btn_cons", on_click=lambda: go("consolidator"))

    with col2:
        st.markdown(
            """
            <div class='qm-card qm-accent-green'>
              <div class='qm-eyebrow'>Section 2</div>
              <div class='qm-title'>Insights</div>
              <div class='qm-copy'>View contribution shares, impressions, and portfolio analytics across combinations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Open Insights", key="btn_insights", on_click=lambda: go("insights"))

    with col3:
        st.markdown(
            """
            <div class='qm-card qm-accent-blue'>
              <div class='qm-eyebrow'>Section 3</div>
              <div class='qm-title'>Journeys</div>
              <div class='qm-copy'>Explore TripleWhale journeys, filter by source, and inspect event sequences up to purchase.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Open Journeys", key="btn_journeys", on_click=lambda: go("journeys"))

    col4, col5 = st.columns(2)

    with col4:
        st.markdown(
            """
            <div class='qm-card qm-accent-green'>
              <div class='qm-eyebrow'>Section 4</div>
              <div class='qm-title'>Optimizer Configurer</div>
              <div class='qm-copy'>Set objective, constraints, guardrails, and scenario metadata before running the optimization.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Configure Optimizer", key="btn_cfg", on_click=lambda: go("optimizer_config"))

    with col5:
        st.markdown(
            """
            <div class='qm-card qm-accent-blue'>
              <div class='qm-eyebrow'>Section 5</div>
              <div class='qm-title'>Marketing Inputs</div>
              <div class='qm-copy'>Run the solver with your chosen decision variables and iterations. Hook this to your backend.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Open Marketing Inputs", key="btn_opt", on_click=lambda: go("optimizer"))

# # ──────────────────────────────────────────────────────────────────────────────
# # SECTION 4 — CONTEXTUAL INPUTS

# def show_contextual():
#     """Section 4 — Contextual Inputs (no budget). Set scenario values for non-media vars.
#     These levels contribute via beta * (x_scenario - mean). Defaults to mean.
#     """
#     def _beta_for(row: pd.Series, var: str) -> float:
#         for key in [f"Weighted_Beta_{var}", f"Selected_Beta_{var}", f"Beta_{var}"]:
#             if key in row and pd.notna(row[key]):
#                 try:
#                     return float(row[key])
#                 except Exception:
#                     pass
#         return 0.0

#     def _mean_for(row: pd.Series, var: str) -> float:
#         for key in [f"Mean_{var}", f"Selected_Mean_{var}"]:
#             if key in row and pd.notna(row[key]):
#                 try:
#                     return float(row[key])
#                 except Exception:
#                     pass
#         return 0.0

#     def _intercept_for(row: pd.Series) -> float:
#         for key in ["Weighted_B0", "Selected_B0 (Original)", "Selected_B0", "B0 (Original)"]:
#             if key in row and pd.notna(row[key]):
#                 try:
#                     return float(row[key])
#                 except Exception:
#                     pass
#         return 0.0

#     st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
#     st.markdown("<span class='qm-eyebrow'>Section 4</span>", unsafe_allow_html=True)
#     st.markdown("<h2>Contextual Inputs</h2>", unsafe_allow_html=True)
#     st.caption("No budget here. Set levels for non-media variables. Saved levels feed into Marketing Inputs evaluation.")

#     saved_models = st.session_state.get("saved_models")
#     grouping_keys = st.session_state.get("grouping_keys", [])
#     other_vars = st.session_state.get("optimizer_other", [])

#     if saved_models is None or saved_models.empty:
#         st.warning("No consolidated models in session. Run Section 1 first.")
#         colA, _ = st.columns([1,4])
#         with colA:
#             if st.button("← Back to Home"):
#                 go("home")
#         st.markdown("</div>", unsafe_allow_html=True)
#         return

#     if not grouping_keys:
#         blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
#         grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
#     labels = [" | ".join([f"{k}:{saved_models.iloc[i][k]}" for k in grouping_keys]) for i in range(len(saved_models))]

#     st.markdown("### Choose Combination (for means)")
#     combo_idx = st.selectbox("Combination", options=list(range(len(saved_models))), format_func=lambda i: labels[i], key="combo_select_c")
#     row = saved_models.iloc[combo_idx]

#     # Build editable grid for contextual levels
#     stored = st.session_state.get("contextual_levels", {})
#     data = []
#     for v in other_vars:
#         mean_x = _mean_for(row, v)
#         scenario = stored.get(v, mean_x)
#         data.append({"Variable": v, "Mean": mean_x, "Scenario": scenario, "Beta": _beta_for(row, v)})
#     grid = pd.DataFrame(data)

#     colcfg = {
#         "Variable": st.column_config.TextColumn(disabled=True),
#         "Mean": st.column_config.NumberColumn(format="%.4f", disabled=True),
#         "Scenario": st.column_config.NumberColumn(format="%.4f"),
#         "Beta": st.column_config.NumberColumn(format="%.6f", disabled=True),
#     }
#     grid = st.data_editor(grid, use_container_width=True, hide_index=True, column_config=colcfg, key=f"ctx_grid_{combo_idx}")

#     # Actions
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         if st.button("Reset to Mean"):
#             for i in range(len(grid)):
#                 grid.at[i, "Scenario"] = grid.at[i, "Mean"]
#     with c2:
#         if st.button("Save Levels"):
#             st.session_state["contextual_levels"] = {str(r["Variable"]): float(r["Scenario"]) for _, r in grid.iterrows()}
#             st.success("Saved contextual levels.")
#     with c3:
#         if st.button("Go to Marketing Inputs →"):
#             go("optimizer")

#     # Quick evaluation (contextual only)
#     b0 = _intercept_for(row)
#     baseline = b0 + sum(_beta_for(row, r["Variable"]) * r["Mean"] for _, r in grid.iterrows())
#     predicted_ctx = b0 + sum(_beta_for(row, r["Variable"]) * r["Scenario"] for _, r in grid.iterrows())
#     st.metric("Contextual-only Δ", f"{(predicted_ctx - baseline):,.2f}")

#     # Nav
#     colA, _ = st.columns([1,4])
#     with colA:
#         if st.button("← Back to Home"):
#             go("home")
#     st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
if st.session_state.section == "home":
    home_cards()
elif st.session_state.section == "consolidator":
    show_consolidator()
elif st.session_state.section == "insights":
    show_insights()
elif st.session_state.section == "journeys":
    show_journeys()
# elif st.session_state.section == "optimizer_config":
#     show_optimizer_config()
# elif st.session_state.section == "optimizer":
#     show_optimizer()
# elif st.session_state.section == "contextual":
#     show_contextual()
else:
    home_cards()
