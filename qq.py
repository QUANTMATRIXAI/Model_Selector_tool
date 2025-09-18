import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional

st.set_page_config(page_title="Elasticity & Coefficient Explorer", layout="wide")


def detect_dimension_columns(columns: List[str]) -> List[str]:
    dimension_cols: List[str] = []
    for col in columns:
        if col.startswith("Weighted_Elasticity") or col.startswith("Weighted_Beta_"):
            break
        dimension_cols.append(col)
    return dimension_cols


def reshape_metric(df: pd.DataFrame, id_cols: List[str], metric_cols: List[str], prefix: str, value_name: str) -> pd.DataFrame:
    if not metric_cols:
        return pd.DataFrame(columns=id_cols + ["Variable", value_name])
    long_df = df[id_cols + metric_cols].melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="Variable",
        value_name=value_name,
    )
    long_df["Variable"] = long_df["Variable"].str.replace(prefix, "", n=1)
    return long_df


def compute_extremes(long_df: pd.DataFrame, dimension_cols: List[str], value_col: str, metric_label: str) -> pd.DataFrame:
    result_columns = dimension_cols + [
        f"{metric_label}_top_positive_variable",
        f"{metric_label}_top_positive_value",
        f"{metric_label}_top_negative_variable",
        f"{metric_label}_top_negative_value",
    ]
    if long_df.empty:
        return pd.DataFrame(columns=result_columns)

    records = []
    if dimension_cols:
        grouped = long_df.groupby(dimension_cols, dropna=False)
    else:
        grouped = [((), long_df)]

    for keys, group in grouped:
        row = {}
        if dimension_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row.update(zip(dimension_cols, keys))

        positives = group[group[value_col] > 0]
        if not positives.empty:
            best_pos = positives.loc[positives[value_col].idxmax()]
            row[f"{metric_label}_top_positive_variable"] = best_pos["Variable"]
            row[f"{metric_label}_top_positive_value"] = best_pos[value_col]
        else:
            row[f"{metric_label}_top_positive_variable"] = pd.NA
            row[f"{metric_label}_top_positive_value"] = pd.NA

        negatives = group[group[value_col] < 0]
        if not negatives.empty:
            best_neg = negatives.loc[negatives[value_col].idxmin()]
            row[f"{metric_label}_top_negative_variable"] = best_neg["Variable"]
            row[f"{metric_label}_top_negative_value"] = best_neg[value_col]
        else:
            row[f"{metric_label}_top_negative_variable"] = pd.NA
            row[f"{metric_label}_top_negative_value"] = pd.NA

        records.append(row)

    return pd.DataFrame(records)


def build_rankings(long_df: pd.DataFrame, dimension_cols: List[str], value_col: str) -> pd.DataFrame:
    rankings = []
    for sign, mask, ascending in (
        ("Positive", long_df[value_col] > 0, False),
        ("Negative", long_df[value_col] < 0, True),
    ):
        subset = long_df[mask].copy()
        if subset.empty:
            continue
        subset["Sign"] = sign
        if dimension_cols:
            subset["Rank"] = (
                subset.groupby(dimension_cols)[value_col]
                .rank(method="dense", ascending=ascending)
                .astype(int)
            )
            subset = subset.sort_values(dimension_cols + ["Sign", "Rank"])  # type: ignore[arg-type]
        else:
            subset["Rank"] = subset[value_col].rank(method="dense", ascending=ascending).astype(int)
            subset = subset.sort_values(["Sign", "Rank"])
        rankings.append(subset.reset_index(drop=True))

    if not rankings:
        base_cols = dimension_cols + ["Variable", value_col, "Sign", "Rank"]
        return pd.DataFrame(columns=base_cols)

    return pd.concat(rankings, ignore_index=True)


def build_contribution_table(
    row: pd.Series,
    beta_cols: List[str],
    elasticity_cols: List[str],
    mean_lookup: Dict[str, str],
    mean_target: Optional[str],
) -> pd.DataFrame:
    mean_y = row.get(mean_target) if mean_target else 1
    contributions = []

    elasticity_set = set(elasticity_cols)

    for beta_col in beta_cols:
        variable = beta_col.replace("Weighted_Beta_", "", 1)
        beta_value = row.get(beta_col)
        mean_col = mean_lookup.get(variable)
        mean_value = row.get(mean_col) if mean_col else pd.NA
        matching_elasticity = f"Weighted_Elasticity_{variable}"
        elasticity_value = row.get(matching_elasticity) if matching_elasticity in elasticity_set else pd.NA

        if pd.notna(beta_value) and pd.notna(mean_value) and pd.notna(mean_y) and mean_y != 0:
            contribution = beta_value * mean_value / mean_y
        else:
            contribution = pd.NA

        contributions.append(
            {
                "Variable": variable,
                "Coefficient": beta_value,
                "Mean Value": mean_value,
                "Elasticity": elasticity_value,
                "Contribution Share": contribution,
            }
        )

    contrib_df = pd.DataFrame(contributions)
    if not contrib_df.empty:
        contrib_df = contrib_df.sort_values("Contribution Share", ascending=False, na_position="last")
    return contrib_df


def format_combination_label(row: pd.Series, dimension_cols: List[str]) -> str:
    if not dimension_cols:
        return "All Data"
    parts = [f"{col}: {row[col]}" for col in dimension_cols]
    return " | ".join(parts)


def main() -> None:
    st.title("Weighted Elasticity & Coefficient Dashboard")
    st.caption("Upload a results file to explore brand/channel drivers and their contributions.")

    uploaded_file = st.file_uploader("Upload CSV export", type=["csv"])
    sample_path = Path("Results_mape_below_30.csv")

    df: pd.DataFrame
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    elif sample_path.exists():
        st.info("No file uploaded. Using bundled sample `Results_mape_below_30.csv`.")
        df = pd.read_csv(sample_path)
    else:
        st.warning("Upload a CSV file to get started.")
        st.stop()

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        st.stop()

    dimension_cols = detect_dimension_columns(df.columns.tolist())
    beta_cols = [c for c in df.columns if c.startswith("Weighted_Beta_")]
    elasticity_cols = [c for c in df.columns if c.startswith("Weighted_Elasticity_")]
    mean_cols = [c for c in df.columns if c.startswith("Mean_")]
    mean_lookup = {c.replace("Mean_", "", 1): c for c in mean_cols}

    coef_long = reshape_metric(df, dimension_cols, beta_cols, "Weighted_Beta_", "Coefficient")
    elasticity_long = reshape_metric(df, dimension_cols, elasticity_cols, "Weighted_Elasticity_", "Elasticity")

    coef_extremes = compute_extremes(coef_long, dimension_cols, "Coefficient", "Coefficient")
    elasticity_extremes = compute_extremes(elasticity_long, dimension_cols, "Elasticity", "Elasticity")

    if not coef_extremes.empty and not elasticity_extremes.empty:
        extreme_summary = coef_extremes.merge(elasticity_extremes, on=dimension_cols, how="outer")
    elif not coef_extremes.empty:
        extreme_summary = coef_extremes.copy()
    else:
        extreme_summary = elasticity_extremes.copy()

    if dimension_cols and not extreme_summary.empty:
        extreme_summary = extreme_summary.sort_values(dimension_cols).reset_index(drop=True)

    coef_rankings = build_rankings(coef_long, dimension_cols, "Coefficient")
    elasticity_rankings = build_rankings(elasticity_long, dimension_cols, "Elasticity")

    mean_target_candidates = [c for c in df.columns if "mean" in c.lower()]
    default_target = "Y_Pred_at_Mean" if "Y_Pred_at_Mean" in df.columns else None
    if default_target is None and mean_target_candidates:
        default_target = mean_target_candidates[0]

    target_col: Optional[str] = None
    if mean_target_candidates:
        target_col = st.selectbox(
            "Select the mean response column (used for contribution shares)",
            options=mean_target_candidates,
            index=mean_target_candidates.index(default_target) if default_target in mean_target_candidates else 0,
        )
    else:
        st.warning("No columns containing 'mean' found; contributions will assume a denominator of 1.")

    tabs = st.tabs(["Overview", "Rankings", "Contribution Analysis"])

    with tabs[0]:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100))
        st.subheader("Top Drivers by Combination")
        if extreme_summary.empty:
            st.info("No coefficient or elasticity columns detected after melting.")
        else:
            st.dataframe(extreme_summary)

    with tabs[1]:
        st.subheader("Coefficient Rankings")
        if coef_rankings.empty:
            st.info("No positive or negative coefficients found.")
        else:
            st.dataframe(coef_rankings)

        st.subheader("Elasticity Rankings")
        if elasticity_rankings.empty:
            st.info("No positive or negative elasticities found.")
        else:
            st.dataframe(elasticity_rankings)

    with tabs[2]:
        st.subheader("Contribution Explorer")
        row: Optional[pd.Series] = None
        if dimension_cols:
            unique_combos = df[dimension_cols].drop_duplicates().reset_index(drop=True)
            combo_labels = unique_combos.apply(lambda r: format_combination_label(r, dimension_cols), axis=1)
            combo_options = combo_labels.tolist()
            selected_label = st.selectbox("Select combination", options=combo_options)
            selected_idx = combo_options.index(selected_label)
            selected_keys = unique_combos.iloc[selected_idx]

            row_mask = (df[dimension_cols] == selected_keys.values).all(axis=1)
            selection_df = df[row_mask]
            if selection_df.empty:
                st.warning("No rows match the selected combination.")
            else:
                row = selection_df.iloc[0]
                st.write("Selected combination:")
                st.write(row[dimension_cols])
        else:
            st.info("No grouping columns detected. Using the first row for contribution analysis.")
            row = df.iloc[0]

        if row is not None:
            contribution_table = build_contribution_table(row, beta_cols, elasticity_cols, mean_lookup, target_col)
            if contribution_table.empty:
                st.warning("Unable to compute contributions. Check that beta/mean columns are present and non-empty.")
            else:
                st.dataframe(contribution_table)

                chart_data = contribution_table.dropna(subset=["Contribution Share"]).head(25)
                if not chart_data.empty:
                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("Contribution Share", title="Contribution Share"),
                            y=alt.Y("Variable", sort="-x"),
                            color=alt.Color("Contribution Share", scale=alt.Scale(scheme="redblue")),
                            tooltip=["Variable", "Coefficient", "Mean Value", "Elasticity", "Contribution Share"],
                        )
                        .properties(height=500)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No contribution values available to chart.")


if __name__ == "__main__":
    main()
