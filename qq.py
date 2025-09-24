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
    "optimizer_config": "Optimizer Configurer",
    "optimizer": "Marketing Inputs",
    "contextual": "Contextual Inputs",
}

if "section" not in st.session_state:
    st.session_state.section = "home"


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

    # =================== BUSINESS-CLEAN: TABLE + SIGNED DIVERGING BAR ===================
    if not display.empty:
        df_view = display.copy()
        df_view["Pretty"] = df_view["Variable"].map(_pretty)
        df_view["Direction"] = np.where(df_view["SignedShare"] >= 0, "Positive", "Negative")
        df_view["ShareLabel"] = df_view["SignedShare"].map(lambda x: f"<b>{abs(x):.1%}</b>")
        df_view = df_view.sort_values("AbsShare", ascending=True)

        # ------- layout: table left, bar right -------
        col_table, col_bar = st.columns([1.1, 1.4])

        # TABLE (left)
        with col_table:
            tbl = df_view.copy().sort_values("AbsShare", ascending=False)
            tbl["Share (%)"] = (tbl["AbsShare"] * 100).round(1)
            tbl = tbl[["Pretty", "Beta", "Mean", "Effect", "Direction", "Share (%)"]].rename(columns={"Pretty": "Variable"})
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

        # BAR (right) — signed, negative axis
        with col_bar:
            fig = px.bar(
                df_view,
                y="Pretty", x="SignedShare",
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
                categoryarray=df_view["Pretty"].tolist()
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
            combo_labels = saved_models["Brand"].astype(str).fillna("—")
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




        # Nav
        colA, colB, colC = st.columns([1,1,2])
        with colA:
            if st.button("← Back to Home"):
                go("home")
        with colB:
            if st.button("Go to Optimizer Configurer →", type="secondary"):
                go("optimizer_config")
        with colC:
            if st.button("Go to Marketing Inputs →", type="secondary"):
                go("optimizer")

        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — OPTIMIZER CONFIGURER (placeholder UI shell)
# ──────────────────────────────────────────────────────────────────────────────

def show_optimizer_config():
    st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
    st.markdown("<span class='qm-eyebrow'>Section 2</span>", unsafe_allow_html=True)
    st.markdown("<h2>Optimizer Configurer</h2>", unsafe_allow_html=True)
    st.caption("Classify variables and set unit costs. You can also upload a unit‑cost file. Your choices are saved for Section 3.")

    saved_models = st.session_state.get("saved_models")
    raw_results  = st.session_state.get("raw_model_results")

    with st.expander("Optional: Upload a results file to derive variables (if you skipped Section 1)"):
        up_alt = st.file_uploader("Upload MODEL_RESULTS or Consolidated CSV/Parquet", type=["csv","parquet"], key="cfg_alt_upload")
        if up_alt is not None:
            df_alt = pd.read_csv(up_alt) if up_alt.name.lower().endswith(".csv") else pd.read_parquet(up_alt)
            if saved_models is None:
                if any(c.startswith("Weighted_Beta_") for c in df_alt.columns):
                    saved_models = df_alt
                else:
                    raw_results = df_alt

    candidates = extract_candidate_variables(raw_results, saved_models)
    all_feats = candidates.get("all_features", [])

    if not all_feats:
        st.warning("No variables detected yet. Please run Section 1 (Consolidator) or upload a results file above.")
        colA, _ = st.columns([1,4])
        with colA:
            if st.button("← Back to Home"):
                go("home")
        st.markdown("</div>", unsafe_allow_html=True)
        return


    # —— 2-column layout: left = classification, right = unit costs ——
    left, right = st.columns([3, 4])

    with left:
        st.markdown("### Variables")
        c_info1, c_info2, c_info3 = st.columns(3)
        c_info1.metric("Total", len(all_feats))
        c_info2.metric("Have Mean_*", len(candidates.get("mean_vars", [])))
        c_info3.metric("From Betas", len(set(candidates.get("beta_vars", [])) | set(candidates.get("weighted_beta_vars", []))))

        # Selections with suggestions
        suggested_media = suggest_media_variables(all_feats)
        default_media = st.session_state.get("optimizer_media", suggested_media)
        sel_media = st.multiselect("Media variables (spend allocation)", options=all_feats, default=default_media, key="sel_media")

        remaining = [v for v in all_feats if v not in sel_media]
        default_other = st.session_state.get("optimizer_other", remaining)
        sel_other = st.multiselect("Other variables (controls / non-spend)", options=all_feats, default=default_other, key="sel_other")

        overlap = set(sel_media) & set(sel_other)
        if overlap:
            st.warning("Removed overlaps from 'Other variables': " + ", ".join(sorted(overlap)))
            sel_other = [v for v in sel_other if v not in overlap]

        if st.button("💾 Save Variable Classes", type="primary", key="btn_save_classes"):
            st.session_state["optimizer_media"] = sel_media
            st.session_state["optimizer_other"] = sel_other
            st.session_state["optimizer_features_all"] = all_feats
            st.success("Saved variable classes.")

    with right:
        st.markdown("### Unit costs (CPC/CPx)")
        st.caption("Currency per 1 unit for each media variable (₹ per click/impression/GRP …). If the var is already spend, use 1.0.")

        # Optional upload of unit-cost file
        uc_up = st.file_uploader("Optional: Upload unit cost file (CSV/Parquet)", type=["csv","parquet"], key="uc_upload")
        uploaded_uc_map = {}
        if uc_up is not None:
            uc_df = pd.read_csv(uc_up) if uc_up.name.lower().endswith(".csv") else pd.read_parquet(uc_up)
            uploaded_uc_map = parse_unit_cost_file(uc_df)
            if uploaded_uc_map:
                st.success(f"Loaded {len(uploaded_uc_map)} unit costs from file.")
                st.dataframe(pd.DataFrame({"Variable": list(uploaded_uc_map.keys()), "UnitCost": list(uploaded_uc_map.values())}), use_container_width=True, hide_index=True)
            else:
                st.warning("Could not detect the right columns. Expect ['variable','unit_cost'] or similar.")

        # Merge defaults → uploaded → session values
        session_uc = st.session_state.get("optimizer_unit_costs", {})
        base_uc = {v: (1.0 if "spend" in v.lower() else 0.0) for v in st.session_state.get("optimizer_media", sel_media)}
        # priority: session > uploaded > base
        merged = {**base_uc, **uploaded_uc_map, **session_uc}
        # Build editable grid for current media
        media_now = st.session_state.get("optimizer_media", sel_media)
        grid_df = pd.DataFrame({
            "Variable": media_now,
            "UnitCost": [float(merged.get(v, 0.0)) for v in media_now]
        })
        edited = st.data_editor(grid_df, num_rows="dynamic", use_container_width=True, hide_index=True)

        if st.button("💾 Save Unit Costs", key="btn_save_costs"):
            try:
                uc_map = {str(r["Variable"]): float(r["UnitCost"]) for _, r in edited.dropna().iterrows()}
            except Exception:
                uc_map = {}
            st.session_state["optimizer_unit_costs"] = uc_map
            st.success("Unit costs saved.")

        st.info("Tip: Adjust classes on the left first; the grid follows your Media selection.")

    # Bottom nav
    bottom_l, bottom_r = st.columns([1,4])
    with bottom_l:
        if st.button("← Back to Home"):
            go("home")
    with bottom_r:
        cgo1, cgo2 = st.columns(2)
        with cgo1:
            if st.button("Go to Marketing Inputs →", key="btn_go_marketing", type="secondary"):
                go("optimizer")
        with cgo2:
            if st.button("Go to Contextual Inputs →", key="btn_go_context", type="secondary"):
                go("contextual")

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — OPTIMIZER (placeholder UI shell)
# ──────────────────────────────────────────────────────────────────────────────
def show_optimizer():
    """Section 3 — Marketing Inputs (budgeted).
    Uses means from results; unit cost converts spend→units.
    Contextual variables (no budget) come from Section 4.
    """
    # ---------- helpers ----------
    def _beta_for(row: pd.Series, var: str) -> float:
        for key in [f"Weighted_Beta_{var}", f"Selected_Beta_{var}", f"Beta_{var}"]:
            if key in row and pd.notna(row[key]):
                try: return float(row[key])
                except Exception: pass
        return 0.0

    def _mean_for(row: pd.Series, var: str) -> float:
        for key in [f"Mean_{var}", f"Selected_Mean_{var}"]:
            if key in row and pd.notna(row[key]):
                try: return float(row[key])
                except Exception: pass
        return 0.0

    def _intercept_for(row: pd.Series) -> float:
        for key in ["Weighted_B0", "Selected_B0 (Original)", "Selected_B0", "B0 (Original)"]:
            if key in row and pd.notna(row[key]):
                try: return float(row[key])
                except Exception: pass
        return 0.0

    def _combo_label(row: pd.Series, grouping_keys: list[str]) -> str:
        parts = [f"{k}:{row[k]}" for k in grouping_keys if k in row.index]
        return " | ".join(parts) if parts else "(combo)"

    def _build_grid(row: pd.Series, media_vars: list[str], unit_costs: dict, existing: pd.DataFrame | None):
        # Columns: Variable, MEAN, CPU, SPEND (editable)
        data = []
        for v in media_vars:
            mean_x = _mean_for(row, v)
            cpu = float(unit_costs.get(v, 0.0))
            spend_default = 0.0
            if existing is not None and not existing.empty:
                try:
                    spend_default = float(existing.loc[existing["Variable"] == v, "SPEND"].values[0])
                except Exception:
                    pass
            data.append({"Variable": v, "MEAN": mean_x, "CPU": cpu, "SPEND": spend_default})
        return pd.DataFrame(data)

    def _evaluate(row: pd.Series, grid_df: pd.DataFrame, contextual_levels: dict | None) -> dict:
        # baseline from means across all features
        b0 = _intercept_for(row)
        baseline = b0
        beta_feats = []
        for c in row.index:
            if isinstance(c, str) and (c.startswith("Weighted_Beta_") or c.startswith("Selected_Beta_") or c.startswith("Beta_")):
                beta_feats.append(c.split("Beta_")[-1])
        for feat in sorted(set(beta_feats)):
            baseline += _beta_for(row, feat) * _mean_for(row, feat)

        # contextual deltas (no budget): x* - mean
        ctx_contrib = 0.0
        if contextual_levels:
            for k, xstar in contextual_levels.items():
                if k not in list(grid_df["Variable"]):  # only non-media here
                    beta = _beta_for(row, k)
                    if beta != 0:
                        dx = float(xstar) - _mean_for(row, k)
                        ctx_contrib += beta * dx

        # marketing deltas via spend→units
        used_budget = float(grid_df["SPEND"].fillna(0).sum())
        mkt_rows, mkt_contrib = [], 0.0
        for _, r in grid_df.iterrows():
            v = r["Variable"]; cpu = float(r["CPU"]) if pd.notna(r["CPU"]) else 0.0
            spend = float(r["SPEND"]) if pd.notna(r["SPEND"]) else 0.0
            mean_x = float(r["MEAN"]) if pd.notna(r["MEAN"]) else 0.0
            units = 0.0 if cpu <= 0 else spend / cpu
            dx = units - mean_x
            beta = _beta_for(row, v)
            contrib = beta * dx
            mkt_contrib += contrib
            mkt_rows.append({"Variable": v, "MEAN": mean_x, "CPU": cpu, "SPEND": spend,
                             "Units": units, "Delta_X": dx, "Beta": beta, "Contribution": contrib})

        predicted = baseline + ctx_contrib + mkt_contrib
        return {
            "baseline": baseline,
            "ctx_contrib": ctx_contrib,
            "mkt_contrib": mkt_contrib,
            "predicted": predicted,
            "used_budget": used_budget,
            "per_var": pd.DataFrame(mkt_rows),
        }

    def _baseline_spend(grid_df: pd.DataFrame) -> pd.DataFrame:
        g = grid_df.copy()
        g["SPEND"] = g.apply(lambda r: float(r["MEAN"]) * float(r["CPU"]) if (pd.notna(r["MEAN"]) and pd.notna(r["CPU"])) else 0.0, axis=1)
        return g

    def _zero_spend(grid_df: pd.DataFrame) -> pd.DataFrame:
        g = grid_df.copy(); g["SPEND"] = 0.0; return g

    def _greedy_max(row: pd.Series, grid_df: pd.DataFrame, budget: float) -> pd.DataFrame:
        # allocate all budget to best beta-per-dollar (simple stub)
        g = _zero_spend(grid_df)
        eff = []
        for _, r in g.iterrows():
            v = r["Variable"]; cpu = float(r["CPU"]) if pd.notna(r["CPU"]) else 0.0
            b = _beta_for(row, v)
            eff.append((v, (b / cpu) if cpu > 0 else -np.inf))
        eff.sort(key=lambda x: x[1], reverse=True)
        if eff and np.isfinite(eff[0][1]) and eff[0][1] > 0:
            vbest = eff[0][0]
            idx = g.index[g["Variable"] == vbest][0]
            g.at[idx, "SPEND"] = float(budget)
        return g

    # ---------- UI ----------
    st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
    st.markdown("<span class='qm-eyebrow'>Section 3</span>", unsafe_allow_html=True)
    st.markdown("<h2>Marketing Inputs</h2>", unsafe_allow_html=True)
    st.caption("Budget applies to marketing variables only. Contextual changes come from the Contextual Inputs page.")

    saved_models = st.session_state.get("saved_models")
    grouping_keys = st.session_state.get("grouping_keys", [])
    media_vars = st.session_state.get("optimizer_media", [])
    unit_costs = st.session_state.get("optimizer_unit_costs", {})
    contextual_levels = st.session_state.get("contextual_levels", {})

    if saved_models is None or saved_models.empty:
        st.warning("No consolidated models in session. Run Section 1 first.")
        colA, _ = st.columns([1,4])
        with colA:
            if st.button("← Back to Home"): go("home")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not grouping_keys:
        blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
        grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
    labels = [" | ".join([f"{k}:{saved_models.iloc[i][k]}" for k in grouping_keys]) for i in range(len(saved_models))]

    st.markdown("### Choose Combination")
    combo_idx = st.selectbox("Combination", options=list(range(len(saved_models))), format_func=lambda i: labels[i], key="combo_select_a")
    row = saved_models.iloc[combo_idx]

    grid_key = f"mkt_grid_{combo_idx}"
    existing = st.session_state.get(grid_key)
    planner_grid = _build_grid(row, media_vars, unit_costs, existing)

    left, right = st.columns([3,2])
    with left:
        st.markdown("### Marketing Inputs Table")
        colcfg = {
            "Variable": st.column_config.TextColumn(disabled=True),
            "MEAN":     st.column_config.NumberColumn(format="%.3f", disabled=True),
            "CPU":      st.column_config.NumberColumn(help="Currency per 1 unit", format="%.6f", disabled=True),
            "SPEND":    st.column_config.NumberColumn(format="%.2f"),
        }
        planner_grid = st.data_editor(planner_grid, use_container_width=True, hide_index=True,
                                      column_config=colcfg, key=f"mkt_grid_edit_{combo_idx}")

    with right:
        st.markdown("### Budget & Actions")
        total_budget = st.number_input("Total budget", value=10000.0, step=100.0, key=f"mkt_budget_{combo_idx}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Baseline Spend", key=f"btn_base_{combo_idx}"):
                planner_grid = _baseline_spend(planner_grid)
        with c2:
            if st.button("Zero Spend", key=f"btn_zero_{combo_idx}"):
                planner_grid = _zero_spend(planner_grid)
        with c3:
            if st.button("Max within Budget", key=f"btn_max_{combo_idx}"):
                planner_grid = _greedy_max(row, planner_grid, total_budget)
        run = st.button("Evaluate Scenario", type="primary", key=f"btn_eval_mkt_{combo_idx}")
        if st.button("Go to Contextual Inputs →", key=f"btn_go_ctx_{combo_idx}"):
            go("contextual")

    st.session_state[grid_key] = planner_grid

    if run:
        res = _evaluate(row, planner_grid, contextual_levels)
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Projected Volume (Pred)", f"{res['predicted']:,.2f}",
                           delta=f"+{(res['predicted']-res['baseline']):,.2f}")
        with m2: st.metric("Baseline Volume (Mean)", f"{res['baseline']:,.2f}")
        with m3: st.metric("Budget Used", f"{res['used_budget']:,.2f}")
        tab1, tab2 = st.tabs(["Per-variable", "Contribs"])
        with tab1: st.dataframe(res["per_var"], use_container_width=True, hide_index=True)
        with tab2: st.write({"Contextual": res["ctx_contrib"], "Marketing": res["mkt_contrib"]})

    colA, _ = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"): go("home")
    st.markdown("</div>", unsafe_allow_html=True)





    # Build combo labels
    if not grouping_keys:
        blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
        grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
    combo_labels = []
    for _, row in saved_models.iterrows():
        parts = [f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns]
        combo_labels.append(" | ".join(parts) if parts else f"row_{_}")

    # —— Layout: left inputs / right budget-bounds — better space mgmt ——
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### Combinations")
        sel_indices = st.multiselect(
            "Select combinations",
            options=list(range(len(saved_models))),
            default=list(range(len(saved_models)))[: min(5, len(saved_models))],
            format_func=lambda i: combo_labels[i]
        )
        if media_vars:
            st.markdown("### Media variables")
            st.caption(", ".join(media_vars))
        else:
            st.info("No media variables configured. Go to Section 2 to set them.")

    with right:
        st.markdown("### Budget & Bounds")
        total_budget = st.number_input("Total budget", value=10000.0, step=100.0)
        lb = st.number_input("Lower bound per media var", value=0.0, step=10.0)
        ub = st.number_input("Upper bound per media var", value=1e9, step=1000.0)
        run = st.button("▶️ Run IPOPT (skeleton)", type="primary")

    if run:
        unit_costs = st.session_state.get("optimizer_unit_costs", {})
        if not sel_indices:
            st.error("Pick at least one combination.")
        elif not media_vars:
            st.error("Define media variables in Section 2.")
        elif not unit_costs or any(v not in unit_costs or unit_costs[v] <= 0 for v in media_vars):
            missing = [v for v in media_vars if v not in unit_costs or unit_costs[v] <= 0]
            st.error("Set positive unit costs for: " + ", ".join(missing))
        else:
            n_vars = len(media_vars)
            per_var_spend = float(np.clip(total_budget / max(n_vars, 1), lb, ub))
            allocation_rows = []
            summary_rows = []

            for i in sel_indices:
                row = saved_models.iloc[i]
                combo_id = " | ".join([f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns])
                alloc_map = {v: per_var_spend for v in media_vars}
                res = predict_from_allocation(row, media_vars, alloc_map, unit_costs)

                for v in media_vars:
                    allocation_rows.append({
                        "Combination": combo_id,
                        "Variable": v,
                        "Spend": alloc_map[v],
                        "UnitCost": unit_costs.get(v, np.nan),
                        "Delta_X": res["per_var_dx"][v],
                        "Beta": _beta_for(row, v),
                        "Contribution": _beta_for(row, v) * res["per_var_dx"][v],
                    })

                summary_rows.append({
                    "Combination": combo_id,
                    "Budget": total_budget,
                    "Baseline_Y": res["baseline_y"],
                    "Predicted_Y": res["predicted_y"],
                    "Lift_Y": res["lift_y"],
                    "Lift_per_$": (res["lift_y"] / total_budget) if total_budget else np.nan,
                })

            alloc_df = pd.DataFrame(allocation_rows)
            summ_df = pd.DataFrame(summary_rows)
            st.session_state["optimizer_allocation"] = alloc_df
            st.session_state["optimizer_summary"] = summ_df

            st.success("Allocation evaluated using unit costs and model betas (linear response). IPOPT hook next.")
            tab1, tab2 = st.tabs(["Allocation breakdown", "Summary"])
            with tab1:
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Allocation Breakdown", data=alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation_breakdown.csv", mime="text/csv")
            with tab2:
                st.dataframe(summ_df, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Summary", data=summ_df.to_csv(index=False).encode("utf-8"), file_name="allocation_summary.csv", mime="text/csv")

    colA, colB = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"):
            go("home")
    st.markdown("</div>", unsafe_allow_html=True)


    # Build combo labels
    if not grouping_keys:
        # fallback: any non-weighted/non-selected metric columns at start
        blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
        grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
    combo_labels = []
    for _, row in saved_models.iterrows():
        parts = [f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns]
        combo_labels.append(" | ".join(parts) if parts else f"row_{_}")

    # Selection UI
    st.markdown("**Select combinations to optimize**")
    sel_indices = st.multiselect(
        "Combinations",
        options=list(range(len(saved_models))),
        default=list(range(len(saved_models)))[: min(5, len(saved_models))],
        format_func=lambda i: combo_labels[i]
    )

    st.markdown("**Budget**")
    total_budget = st.number_input("Total budget (currency units)", value=10000.0, step=100.0)

    if not media_vars:
        st.info("No media variables configured. Go to Section 2 to set media/control variables.")
    else:
        st.markdown("**Media variables (decision investments)**")
        st.caption(", ".join(media_vars))

    # Placeholder bounds grid (future: per combo & per var bounds)
    st.markdown("**Bounds (skeleton)**")
    lb = st.number_input("Lower bound per media var", value=0.0, step=10.0)
    ub = st.number_input("Upper bound per media var", value=1e9, step=1000.0)

    run = st.button("▶️ Run IPOPT (skeleton)", type="primary")

    if run:
        unit_costs = st.session_state.get("optimizer_unit_costs", {})
        if not sel_indices:
            st.error("Pick at least one combination.")
        elif not media_vars:
            st.error("Define media variables in Section 2.")
        elif not unit_costs or any(v not in unit_costs or unit_costs[v] <= 0 for v in media_vars):
            missing = [v for v in media_vars if v not in unit_costs or unit_costs[v] <= 0]
            st.error("Set positive unit costs for: " + ", ".join(missing))
        else:
            n_vars = len(media_vars)
            per_var_spend = float(np.clip(total_budget / max(n_vars, 1), lb, ub))
            allocation_rows = []
            summary_rows = []

            for i in sel_indices:
                row = saved_models.iloc[i]
                combo_id = " | ".join([f"{k}:{row[k]}" for k in grouping_keys if k in saved_models.columns])
                alloc_map = {v: per_var_spend for v in media_vars}
                res = predict_from_allocation(row, media_vars, alloc_map, unit_costs)

                for v in media_vars:
                    allocation_rows.append({
                        "Combination": combo_id,
                        "Variable": v,
                        "Spend": alloc_map[v],
                        "UnitCost": unit_costs.get(v, np.nan),
                        "Delta_X": res["per_var_dx"][v],
                        "Beta": _beta_for(row, v),
                        "Contribution": _beta_for(row, v) * res["per_var_dx"][v],
                    })

                summary_rows.append({
                    "Combination": combo_id,
                    "Budget": total_budget,
                    "Baseline_Y": res["baseline_y"],
                    "Predicted_Y": res["predicted_y"],
                    "Lift_Y": res["lift_y"],
                    "Lift_per_$": (res["lift_y"] / total_budget) if total_budget else np.nan,
                })

            alloc_df = pd.DataFrame(allocation_rows)
            summ_df = pd.DataFrame(summary_rows)
            st.session_state["optimizer_allocation"] = alloc_df
            st.session_state["optimizer_summary"] = summ_df

            st.success("Allocation evaluated using unit costs and model betas (linear response). IPOPT hook next.")
            st.subheader("Per-variable allocation breakdown")
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Allocation Breakdown", data=alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation_breakdown.csv", mime="text/csv")

            st.subheader("Combination summary")
            st.dataframe(summ_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Summary", data=summ_df.to_csv(index=False).encode("utf-8"), file_name="allocation_summary.csv", mime="text/csv")

    colA, colB = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"):
            go("home")
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HOME — Three Cards (Consolidator, Optimizer Configurer, Optimizer)
# ──────────────────────────────────────────────────────────────────────────────

def home_cards():
    st.markdown("<div class='qm-hero'>QuantMatrix — Model & Optimizer Suite</div>", unsafe_allow_html=True)
    st.markdown("<div class='qm-sub'>Section 1: Model Consolidator • Section 2: Insights • Section 3: Optimizer Configurer • Section 4: Marketing Inputs</div>", unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        # Card content (text INSIDE the card)
        st.markdown(
            """
            <div class='qm-card qm-accent-yellow'>
              <div class='qm-eyebrow'>Section 1</div>
              <div class='qm-title'>Model Consolidator</div>
              <div class='qm-copy'>Upload your model results and collapse to a single model per combination via a winner model or a MAPE‑weighted ensemble.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Button BELOW the card
        st.button("Open Consolidator", key="btn_cons", on_click=lambda: go("consolidator"))

    with c2:
        st.markdown(
            """
            <div class='qm-card qm-accent-green'>
              <div class='qm-eyebrow'>Section 2</div>
              <div class='qm-title'>Insights</div>
              <div class='qm-copy'>View contribution shares and charts across variables per combination.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Open Insights", key="btn_insights", on_click=lambda: go("insights"))

    with c3:
        st.markdown(
            """
            <div class='qm-card qm-accent-green'>
              <div class='qm-eyebrow'>Section 3</div>
              <div class='qm-title'>Optimizer Configurer</div>
              <div class='qm-copy'>Set objective, constraints, guardrails, and scenario metadata before running the optimization.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Configure Optimizer", key="btn_cfg", on_click=lambda: go("optimizer_config"))

    with c4:
        st.markdown(
            """
            <div class='qm-card qm-accent-blue'>
              <div class='qm-eyebrow'>Section 4</div>
              <div class='qm-title'>Marketing Inputs</div>
              <div class='qm-copy'>Run the solver with your chosen decision variables and iterations. Hook this to your backend.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Open Marketing Inputs", key="btn_opt", on_click=lambda: go("optimizer"))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CONTEXTUAL INPUTS

def show_contextual():
    """Section 4 — Contextual Inputs (no budget). Set scenario values for non-media vars.
    These levels contribute via beta * (x_scenario - mean). Defaults to mean.
    """
    def _beta_for(row: pd.Series, var: str) -> float:
        for key in [f"Weighted_Beta_{var}", f"Selected_Beta_{var}", f"Beta_{var}"]:
            if key in row and pd.notna(row[key]):
                try:
                    return float(row[key])
                except Exception:
                    pass
        return 0.0

    def _mean_for(row: pd.Series, var: str) -> float:
        for key in [f"Mean_{var}", f"Selected_Mean_{var}"]:
            if key in row and pd.notna(row[key]):
                try:
                    return float(row[key])
                except Exception:
                    pass
        return 0.0

    def _intercept_for(row: pd.Series) -> float:
        for key in ["Weighted_B0", "Selected_B0 (Original)", "Selected_B0", "B0 (Original)"]:
            if key in row and pd.notna(row[key]):
                try:
                    return float(row[key])
                except Exception:
                    pass
        return 0.0

    st.markdown("<div class='qm-panel'>", unsafe_allow_html=True)
    st.markdown("<span class='qm-eyebrow'>Section 4</span>", unsafe_allow_html=True)
    st.markdown("<h2>Contextual Inputs</h2>", unsafe_allow_html=True)
    st.caption("No budget here. Set levels for non-media variables. Saved levels feed into Marketing Inputs evaluation.")

    saved_models = st.session_state.get("saved_models")
    grouping_keys = st.session_state.get("grouping_keys", [])
    other_vars = st.session_state.get("optimizer_other", [])

    if saved_models is None or saved_models.empty:
        st.warning("No consolidated models in session. Run Section 1 first.")
        colA, _ = st.columns([1,4])
        with colA:
            if st.button("← Back to Home"):
                go("home")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not grouping_keys:
        blacklist_prefix = ("Weighted_", "Selected_", "Mean_", "Models_Used", "Best_", "Avg_", "Equation_Complete", "Y_Pred_")
        grouping_keys = [c for c in saved_models.columns if not str(c).startswith(blacklist_prefix)][:3]
    labels = [" | ".join([f"{k}:{saved_models.iloc[i][k]}" for k in grouping_keys]) for i in range(len(saved_models))]

    st.markdown("### Choose Combination (for means)")
    combo_idx = st.selectbox("Combination", options=list(range(len(saved_models))), format_func=lambda i: labels[i], key="combo_select_c")
    row = saved_models.iloc[combo_idx]

    # Build editable grid for contextual levels
    stored = st.session_state.get("contextual_levels", {})
    data = []
    for v in other_vars:
        mean_x = _mean_for(row, v)
        scenario = stored.get(v, mean_x)
        data.append({"Variable": v, "Mean": mean_x, "Scenario": scenario, "Beta": _beta_for(row, v)})
    grid = pd.DataFrame(data)

    colcfg = {
        "Variable": st.column_config.TextColumn(disabled=True),
        "Mean": st.column_config.NumberColumn(format="%.4f", disabled=True),
        "Scenario": st.column_config.NumberColumn(format="%.4f"),
        "Beta": st.column_config.NumberColumn(format="%.6f", disabled=True),
    }
    grid = st.data_editor(grid, use_container_width=True, hide_index=True, column_config=colcfg, key=f"ctx_grid_{combo_idx}")

    # Actions
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset to Mean"):
            for i in range(len(grid)):
                grid.at[i, "Scenario"] = grid.at[i, "Mean"]
    with c2:
        if st.button("Save Levels"):
            st.session_state["contextual_levels"] = {str(r["Variable"]): float(r["Scenario"]) for _, r in grid.iterrows()}
            st.success("Saved contextual levels.")
    with c3:
        if st.button("Go to Marketing Inputs →"):
            go("optimizer")

    # Quick evaluation (contextual only)
    b0 = _intercept_for(row)
    baseline = b0 + sum(_beta_for(row, r["Variable"]) * r["Mean"] for _, r in grid.iterrows())
    predicted_ctx = b0 + sum(_beta_for(row, r["Variable"]) * r["Scenario"] for _, r in grid.iterrows())
    st.metric("Contextual-only Δ", f"{(predicted_ctx - baseline):,.2f}")

    # Nav
    colA, _ = st.columns([1,4])
    with colA:
        if st.button("← Back to Home"):
            go("home")
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
if st.session_state.section == "home":
    home_cards()
elif st.session_state.section == "consolidator":
    show_consolidator()
elif st.session_state.section == "insights":
    show_insights()
elif st.session_state.section == "optimizer_config":
    show_optimizer_config()
elif st.session_state.section == "optimizer":
    show_optimizer()
elif st.session_state.section == "contextual":
    show_contextual()
else:
    home_cards()
