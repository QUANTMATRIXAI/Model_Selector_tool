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


import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data

def load_data(file, apply_method_filter=True, apply_dist_elas_filter=True, apply_cat_elas_filter=True, apply_pval_filter=True):
    df = pd.read_excel(file, sheet_name='FinalM0')
 
    df['selectedmodels'] = 'no'
 
    # Conditionally remove 'Lasso' and 'Ridge' methods
    if apply_method_filter:
        df = df[~df['method'].isin(['Lasso', 'Ridge'])]
 
    # Conditionally apply elasticity filters separately
    if apply_dist_elas_filter:
        df = df[(df['Distribution_elas'].ge(0) | df['Distribution_elas'].isna())]
 
    if apply_cat_elas_filter:
        df = df[(df['Category_elas'].ge(0) | df['Category_elas'].isna())]
 
    # Conditionally apply p-value filter
    if apply_pval_filter:
        df = df[df['Price_pval'] == 'Yes']
 
    # Conditionally fill NaN values with column mean
   
    df[['Adj.Rsq', 'AIC', 'CSF.CSF']] = df[['Adj.Rsq', 'AIC', 'CSF.CSF']].apply(lambda x: x.fillna(x.mean()))
 
    # Select the range of columns from 'Price_elas' to 'beta0'
    if 'Price_elas' in df.columns and 'beta0' in df.columns:
        elas_to_beta_columns = df.loc[:, 'Price_elas':'beta0'].columns.tolist()
    else:
        elas_to_beta_columns = []
 
    selected_columns = [
        'method', 'Channel', 'Market', 'Brand', 'Variant', 'PackType', 'PPG', 'Region', 'Category',
        'SubCategory', 'PackSize', 'selectedmodels', 'Adj.Rsq', 'AIC', 'RPIto', 'MCV.MCV', 'CSF.CSF', 'actualdistvar',
        'Price_beta', 'Distribution_beta', 'Price_pval', 'Rsq', 'Vol_Var'
    ] + elas_to_beta_columns
 
    df = df[[col for col in selected_columns if col in df.columns]]
 
    required_columns = ['Market', 'Channel', 'Region', 'Category',
                        'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
 
    columns_with_multiple_unique_values = [
        col for col in required_columns if df[col].nunique() > 1
    ]
 
    for col in ['Market', 'Category', 'Channel']:
        if col not in columns_with_multiple_unique_values:
            columns_with_multiple_unique_values.append(col)
 
    median_csf = df.groupby(columns_with_multiple_unique_values)['CSF.CSF'].median().reset_index()
    median_csf.rename(columns={'CSF.CSF': 'Median_CSF'}, inplace=True)
 
    df = df.merge(median_csf, on=columns_with_multiple_unique_values, how='left')
    df['CSF_Diff'] = abs(df['CSF.CSF'] - df['Median_CSF'])
 
    min_diff_idx = df.groupby(columns_with_multiple_unique_values)['CSF_Diff'].idxmin()
    df.loc[min_diff_idx, 'selectedmodels'] = 'Yes'
 
    df.drop(columns=['Median_CSF', 'CSF_Diff'], inplace=True)
 
    return df












# Custom CSS for styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color:rgb(253, 241, 171); /* Ochre */
        color: black;
    }
    .stButton>button {
        background-color: #F0E68C; /* Ochre */
        color: black;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FFD700; /* Darker ochre */
        color: white;
    }
    .stSlider>div, .stSelectbox, .stMultiselect {
        color: black;
        border-color: black;
    }
    .stRadio div[role="radiogroup"] > label > div:first-child {
        background-color: #F0E68C; /* Ochre */
        color: black;
    }
    .stCheckbox label {
        color: black;
    }
    hr.thick {
        border: 2px solid black;
    }
    hr.thin {
        border: 1px solid gray;
    }
    </style>
""", unsafe_allow_html=True)









if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}


# Define the required columns
required_columns = [
    'Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand', 
    'PPG', 'Variant', 'PackType', 'PackSize', 'Year', 'Month', 
    'Week', 'date', 'BrCatId', 'SalesValue', 'Volume', 'VolumeUnits'
]

# Sidebar for file upload
st.sidebar.title("📂 Upload Section")
st.sidebar.markdown('<hr class="thick">', unsafe_allow_html=True) 

uploaded_file = st.sidebar.file_uploader("📄 Upload your D0 file (CSV format)", type=['csv'])

if uploaded_file:
    try:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Handle both "Date" and "date" columns
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'date' in df.columns:
            df.rename(columns={'date': 'date'}, inplace=True)
        
        # Filter the required columns
        filtered_columns = [col for col in required_columns if col in df.columns]
        D0 = df[filtered_columns]

    

        st.session_state["uploaded_files"]["D0"] = D0

        unique_market_values = D0['Market'].unique() if 'Market' in D0.columns else []
        unique_category_values = D0['Category'].unique() if 'Category' in D0.columns else []

        
    except Exception as e:
        # Catch and display any errors that occur
        st.sidebar.error(f"An error occurred while processing the file: {e}")
else:
    st.sidebar.info("Please upload a CSV file to proceed.")


# st.sidebar.markdown('<hr class="thin">', unsafe_allow_html=True)
# st.sidebar.markdown('<hr class="thin">', unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <style>
        .double-line {
            border: none;
            height: 1px;
            background-color: black;
            box-shadow: 0px 2px 2px black; /* Second line below */
            margin: 5px 0;
        }
    </style>
    <hr class="double-line">
    """,
    unsafe_allow_html=True
)
if uploaded_file:
    for market in unique_market_values:
        for category in unique_category_values:
            st.header(f"{str(market).upper()} - {str(category).upper()}")
            st.markdown('<hr class="thin">', unsafe_allow_html=True)





options = ["CONFIGURATION"]
option = st.pills(f"CREATE CONFIGURATION FILE! ", options,selection_mode="single",default="CONFIGURATION",)

if option == "CONFIGURATION":
    # Define available options
    modeling_type_options = [1, 2]
    msp_options = ["Yes", "No"]
    variable_options = ['NA', 'Market', 'Channel', 'Region', 'Category',
                        'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
    periodicity_options = ["Daily", "Weekly", "Monthly"]

    # Function to get a valid index
    def get_valid_index(option_list, stored_value, default_value):
        return option_list.index(stored_value) if stored_value in option_list else option_list.index(default_value)

    # Initialize session state if not already set
    if "config_data" not in st.session_state:
        st.session_state.config_data = {
            "Modeling Type": 2,
            "MSP at L0 (Yes/No)": "Yes",
            "Variable to use as L0": "Brand",
            "MSP at L2 (Yes/No)": "No",
            "Variable to use as L2": "NA",
            "MSP at L3 (Yes/No)": "No",
            "Variable to use as L3": "NA",
            "Periodicity of Data (Daily/Weekly/Monthly)": "Weekly",
            "Reference Period": 1
        }

    # Get valid indices for selectbox
    modeling_type_index = get_valid_index(modeling_type_options, st.session_state.config_data["Modeling Type"], 2)
    msp_l0_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L0 (Yes/No)"], "Yes")
    msp_l2_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L2 (Yes/No)"], "No")
    msp_l3_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L3 (Yes/No)"], "No")
    variable_l0_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L0"], "Brand")
    variable_l2_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L2"], "NA")
    variable_l3_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L3"], "NA")
    periodicity_index = get_valid_index(periodicity_options, st.session_state.config_data["Periodicity of Data (Daily/Weekly/Monthly)"], "Daily")

    # Create a form for user input
    with st.form("config_form"):
        modeling_type = st.selectbox("Modeling Type", options=modeling_type_options, index=modeling_type_index)

        col10, col11 = st.columns(2)
        with col10:
            msp_l0 = st.selectbox("MSP at L0 (Yes/No)", options=msp_options, index=msp_l0_index)
            msp_l2 = st.selectbox("MSP at L2 (Yes/No)", options=msp_options, index=msp_l2_index)
            msp_l3 = st.selectbox("MSP at L3 (Yes/No)", options=msp_options, index=msp_l3_index)
            periodicity = st.selectbox("Periodicity of Data (Daily/Weekly/Monthly)", options=periodicity_options, index=periodicity_index)
        
        with col11:
            variable_l0 = st.selectbox("Variable to use as L0", options=variable_options, index=variable_l0_index)
            variable_l2 = st.selectbox("Variable to use as L2", options=variable_options, index=variable_l2_index)
            variable_l3 = st.selectbox("Variable to use as L3", options=variable_options, index=variable_l3_index)
            reference_period = st.number_input("Reference Period", min_value=1, value=st.session_state.config_data["Reference Period"])

        # Submit button
        submit_button = st.form_submit_button("Save Configuration")

        # If submitted, update session state
        if submit_button:
            st.session_state.config_data = {
                "Modeling Type": modeling_type,
                "MSP at L0 (Yes/No)": msp_l0,
                "Variable to use as L0": variable_l0,
                "MSP at L2 (Yes/No)": msp_l2,
                "Variable to use as L2": variable_l2,
                "MSP at L3 (Yes/No)": msp_l3,
                "Variable to use as L3": variable_l3,
                "Periodicity of Data (Daily/Weekly/Monthly)": periodicity,
                "Reference Period": reference_period
            }
            st.write("Configuration saved successfully!")

    # Display current configuration
    config_df = pd.DataFrame([st.session_state.config_data])
    st.write("Current Configuration:")
    st.dataframe(config_df)
st.markdown('<hr class="thin">', unsafe_allow_html=True)








tab1, tab2, tab3 = st.tabs(["MSP L0", "MSP L0L2", "MSP L0L2L3"])


uploaded_file0= st.sidebar.file_uploader("📄 Upload L0 file: ", type="xlsx")
with st.sidebar.expander("🔧 Advanced Settings for L0", expanded=False):
    apply_method_filter0 = st.checkbox("Remove 'Lasso' and 'Ridge' methods", value=True, key="l0_method")
    apply_dist_elas_filter0 = st.checkbox("Apply Distribution_elas Filter (>= 0 or NaN)", value=True, key="l0_dist_elas")
    apply_cat_elas_filter0 = st.checkbox("Apply Category_elas Filter (>= 0 or NaN)", value=True, key="l0_cat_elas")
    apply_pval_filter0 = st.checkbox("Filter Price_pval == 'Yes'", value=True, key="l0_pval")




st.sidebar.markdown('<hr class="thin">', unsafe_allow_html=True)



if uploaded_file0:
    
    # tab1, =st.tabs(["MSP L0"])
    with tab1:
        # uploaded_file0= st.sidebar.file_uploader("Upload L0 file: ", type="xlsx")

        
        if uploaded_file0:
            try:
                # if "uploaded_file0" not in st.session_state or uploaded_file0 != st.session_state["uploaded_file0"]:

                if ("uploaded_file0" not in st.session_state or uploaded_file0 != st.session_state["uploaded_file0"]) or \
                    (apply_method_filter0 != st.session_state.get("apply_method_filter0", True) or
                    apply_dist_elas_filter0 != st.session_state.get("apply_dist_elas_filter0", True) or
                    apply_cat_elas_filter0 != st.session_state.get("apply_cat_elas_filter0", True) or
                    apply_pval_filter0 != st.session_state.get("apply_pval_filter0", True)):
                    
                    # Clear session state data when a new file is uploaded
                    if "L0" in st.session_state:
                        del st.session_state["L0"]

                   
                    
                    # Load new data
                    L0 = load_data(uploaded_file0,apply_method_filter0, apply_dist_elas_filter0, apply_cat_elas_filter0, apply_pval_filter0)

                    
                    # Save to session state
                    st.session_state["uploaded_file0"] = uploaded_file0
                    st.session_state["L0"] = L0
                    st.session_state["apply_method_filter0"] = apply_method_filter0
                    st.session_state["apply_dist_elas_filter0"] = apply_dist_elas_filter0
                    st.session_state["apply_cat_elas_filter0"] = apply_cat_elas_filter0
                    st.session_state["apply_pval_filter0"] = apply_pval_filter0


                    # Store the file name without the extension
                    file_name = uploaded_file0.name.split('.')[0]
                    
                    # Save the uploaded file reference and data to session state
                    st.session_state["uploaded_file0"] = uploaded_file0
                    st.session_state["L0"] = L0


                    st.session_state["file_name"] = file_name

                else:
                    # If the file hasn't changed, load the data from session state
                    L0 = st.session_state.get("L0")

            except Exception as e:
                # Catch and display any errors that occur
                st.sidebar.error(f"An error occurred while processing the file: {e}")
        else:
            st.sidebar.info("Please upload an Excel file to proceed.")


        



        col1, col2 = st.columns(2)

        with col1:
            
            # Streamlit app

            if "L0" not in st.session_state:
                st.session_state["L0"] = None

            if "file_name" not in st.session_state:
                st.session_state["file_name"] = None

            if "filtered_data" not in st.session_state:
                st.session_state["filtered_data"] = None



            file_name = st.session_state["file_name"]
            st.session_state[file_name] = st.session_state["L0"].copy()
            
            # Ensure `st.session_state["data"]` is not None before further processing
            if st.session_state["L0"] is not None:
                global_filtered_data = st.session_state["L0"].copy()

              
                selected_channels = st.session_state.get("selected_channels", [])

                # Filter data dynamically based on global filter selections
                filtered_global_data = L0.copy()

                if selected_channels:
                    filtered_global_data = filtered_global_data[filtered_global_data['Channel'].isin(selected_channels)]

                
                updated_channels = filtered_global_data['Channel'].unique()
                
           
                selected_channels = st.multiselect(
                    "Select Channel",
                    options=updated_channels,
                    default=st.session_state.get("selected_channels", []),
                    key="selected_channels",
        
                )

                st.markdown('<hr class="thick">', unsafe_allow_html=True)


                # Display each brand's data and filters
                unique_brands = filtered_global_data['Brand'].unique()

                for brand in unique_brands:
                    st.header(f"BRAND: {brand}")
                    brand_data = filtered_global_data[filtered_global_data['Brand'] == brand]

                    # Brand-specific filters beside graphs
                    col6, col7 = st.columns([3, 1])  # Make graphs column wider

                    with col7:
                        # Initialize filters dynamically
                        methods = st.multiselect(
                            f"Select Method for {brand}",
                            options=brand_data['method'].unique(),
                            key=f"method_{brand}"
                        )

                        # Filter brand_data dynamically based on the selected method
                        filtered_data_by_method = brand_data.copy()
                        if methods:
                            filtered_data_by_method = filtered_data_by_method[filtered_data_by_method['method'].isin(methods)]



                        
                        rpito_options = filtered_data_by_method['RPIto'].unique()
                        rpito_filter = st.multiselect(
                            f"RPIto for {brand}",
                            options=rpito_options,
                            key=f"rpito_{brand}"
                        )

                        # Further filter based on selected RPIto
                        filtered_data_by_rpito = filtered_data_by_method.copy()
                        if rpito_filter:
                            filtered_data_by_rpito = filtered_data_by_rpito[filtered_data_by_rpito['RPIto'].isin(rpito_filter)]


                        
                        actualdistvar_options = filtered_data_by_rpito['actualdistvar'].unique()
                        actualdistvar_filter = st.multiselect(
                            f"actualdistvar for {brand}",
                            options=actualdistvar_options,
                            key=f"actualdistvar_{brand}"
                        )

                        # Further filter based on selected RPIto
                        filtered_data_by_actualdistvar = filtered_data_by_rpito.copy()
                        if actualdistvar_filter:
                            filtered_data_by_actualdistvar = filtered_data_by_actualdistvar[filtered_data_by_actualdistvar['actualdistvar'].isin(actualdistvar_filter)]
    
                        

                        # Handle sliders with identical min and max
                        adjrsq_min = float(filtered_data_by_actualdistvar['Adj.Rsq'].min())
                        adjrsq_max = float(filtered_data_by_actualdistvar['Adj.Rsq'].max())
                        if adjrsq_min == adjrsq_max:
                            adjrsq_min -= 0.01
                            adjrsq_max += 0.01
                        adjrsquare_filter = st.slider(
                            f"Adj Rsquare for {brand}",
                            min_value=adjrsq_min,
                            max_value=adjrsq_max,
                            value=(adjrsq_min, adjrsq_max),
                            key=f"adjrsq_{brand}"
                        )

                        # Further filter based on Adj.Rsq
                        filtered_data_by_adjrsq = filtered_data_by_actualdistvar.copy()
                        filtered_data_by_adjrsq = filtered_data_by_adjrsq[
                            (filtered_data_by_adjrsq['Adj.Rsq'] >= adjrsquare_filter[0]) & 
                            (filtered_data_by_adjrsq['Adj.Rsq'] <= adjrsquare_filter[1])
                        ]

                        aic_min = float(filtered_data_by_adjrsq['AIC'].min())
                        aic_max = float(filtered_data_by_adjrsq['AIC'].max())
                        if aic_min == aic_max:
                            aic_min -= 0.01
                            aic_max += 0.01
                        aic_filter = st.slider(
                            f"AIC for {brand}",
                            min_value=aic_min,
                            max_value=aic_max,
                            value=(aic_min, aic_max),
                            key=f"aic_{brand}"
                        )

                        # Further filter based on AIC
                        filtered_data_by_aic = filtered_data_by_adjrsq.copy()
                        filtered_data_by_aic = filtered_data_by_aic[
                            (filtered_data_by_aic['AIC'] >= aic_filter[0]) & 
                            (filtered_data_by_aic['AIC'] <= aic_filter[1])
                        ]

                        csf_min = float(filtered_data_by_aic['CSF.CSF'].min())
                        csf_max = float(filtered_data_by_aic['CSF.CSF'].max())
                        if csf_min == csf_max:
                            csf_min -= 0.01
                            csf_max += 0.01
                        csf_filter = st.slider(
                            f"CSF for {brand}",
                            min_value=csf_min,
                            max_value=csf_max,
                            value=(csf_min, csf_max),
                            key=f"csf_{brand}"
                        )

                        # Final filtered data based on CSF
                        filtered_data = filtered_data_by_aic.copy()
                        filtered_data = filtered_data[
                            (filtered_data['CSF.CSF'] >= csf_filter[0]) & 
                            (filtered_data['CSF.CSF'] <= csf_filter[1])
                        ]

                        # st.session_state["filtered_data"]=filtered_data


                       


                    # Ensure 'selectedmodels' column exists
                    filtered_data["selectedmodels"] = "No"

                    # Initialize session state if not present
                    if "saved_selections" not in st.session_state:
                        st.session_state["saved_selections"] = {}

                    if "saved_buttons" not in st.session_state:
                        st.session_state["saved_buttons"] = {}

                    if "reset_buttons" not in st.session_state:
                        st.session_state["reset_buttons"] = {}

                    # Function to update 'selectedmodels' column ensuring one "Yes" per (Channel, Brand)
                    def update_selectedmodels(group, brand):
                        channel = group["Channel"].iloc[0]  # Get the Channel for this group

                        # Check if selection was saved for this (Channel, Brand)
                        key = (channel, brand)
                        if key in st.session_state["saved_selections"]:
                            saved_index = st.session_state["saved_selections"][key]
                            if saved_index in group.index:
                                group["selectedmodels"] = "No"
                                group.loc[saved_index, "selectedmodels"] = "Yes"
                                return group  # Return saved selection

                        # Otherwise, select the row closest to the median CSF.CSF
                        if not group.empty:
                            median_csf = group["CSF.CSF"].median()
                            closest_index = (group["CSF.CSF"] - median_csf).abs().idxmin()
                            group["selectedmodels"] = "No"
                            group.loc[closest_index, "selectedmodels"] = "Yes"

                        return group

                    # Apply selection logic per (Channel, Brand)
                    filtered_data = filtered_data.groupby(["Channel", "Brand"], group_keys=False).apply(update_selectedmodels, brand=brand)

                    # UI for selecting models
                    for (channel, brand), group in filtered_data.groupby(["Channel", "Brand"]):
                        available_indices = group.index.tolist()

                        default_index = (
                            group[group["selectedmodels"] == "Yes"].index[0]
                            if "Yes" in group["selectedmodels"].values
                            else available_indices[0]
                        )

                        selected_index = st.selectbox(
                            f"Select Model Index for {channel} - {brand}",
                            options=available_indices,
                            index=available_indices.index(default_index),
                            key=f"selectbox_{channel}_{brand}"
                        )

                        # Ensure selectedmodels is locked once saved
                        if selected_index in group.index:
                            key = (channel, brand)
                            if key in st.session_state["saved_selections"]:
                                saved_index = st.session_state["saved_selections"][key]
                                group["selectedmodels"] = "No"
                                group.loc[saved_index, "selectedmodels"] = "Yes"
                            else:
                                group["selectedmodels"] = "No"
                                group.loc[selected_index, "selectedmodels"] = "Yes"

                        col3, col4 = st.columns([3, 2])

                        # Dynamically set button text based on session state
                        save_button_text = st.session_state["saved_buttons"].get((channel, brand), f"SAVE Selection {channel} - {brand}")
                        reset_button_text = st.session_state["reset_buttons"].get((channel, brand), f"RESET Selection {channel} - {brand}")

                        with col3:
                            # Button to save selection
                            if st.button(save_button_text, key=f"save_{channel}_{brand}"):
                                if (channel, brand) in st.session_state["saved_selections"]:
                                    saved_index = st.session_state["saved_selections"][(channel, brand)]
                                    if saved_index != selected_index:
                                        st.error(f"Selection already saved for {channel} - {brand} at index {saved_index}.\n\n Please 'RESET' first before changing.")
                                    else:
                                        st.success(f"Selection already saved for {channel} - {brand} at index {saved_index}.")
                                else:
                                    st.session_state["saved_selections"][(channel, brand)] = selected_index
                                    st.session_state["saved_buttons"][(channel, brand)] = f"SAVED ✅ ({channel} - {brand})"
                                    st.session_state["reset_buttons"][(channel, brand)] = f"RESET Selection {channel} - {brand}"
                                    st.success(f"Selection saved for {channel} - {brand} at index {selected_index}.")

                        with col4:
                            # Button to reset selection
                            if st.button(reset_button_text, key=f"reset_{channel}_{brand}"):
                                if (channel, brand) in st.session_state["saved_selections"]:
                                    del st.session_state["saved_selections"][(channel, brand)]
                                st.session_state["reset_buttons"][(channel, brand)] = f"RESET 🔄 ({channel} - {brand})"
                                st.session_state["saved_buttons"][(channel, brand)] = f"SAVE Selection {channel} - {brand}"
                                st.success(f"Selection reset for {channel} - {brand}.\n\n Now updates dynamically.")

                        # Store updates in session state
                        st.session_state["filtered_data"] = filtered_data

                        # Ensure L0 is updated properly with "Yes" for each (Channel, Brand)
                        if "L0" in st.session_state:
                            L0 = st.session_state["L0"]

                            # Reset all selectedmodels to "No" for this Channel-Brand pair in L0
                            L0.loc[(L0["Channel"] == channel) & (L0["Brand"] == brand), "selectedmodels"] = "No"

                            # Apply the saved selection or median-based selection
                            selected_row = group[group["selectedmodels"] == "Yes"]
                            if not selected_row.empty:
                                selected_idx = selected_row.index[0]
                                L0.loc[selected_idx, "selectedmodels"] = "Yes"

                            st.session_state["L0"] = L0.copy()

                            # Save updates to the session state file
                            file_name = st.session_state.get("file_name", "L0")
                            st.session_state[file_name] = L0.copy()






                    #     # Store the updated filtered_data in session state
                    # st.session_state["filtered_data"] = filtered_data

                    # # Update only the 'selectedmodels' column in L0 for rows marked as 'Yes' in filtered_data
                    # if "L0" in st.session_state:
                    #     L0 = st.session_state["L0"]

                    #     for (channel, brand), group in filtered_data.groupby(['Channel', 'Brand']):
                    #         # Reset all 'selectedmodels' to 'No' for this (Channel, Brand) in L0
                    #         L0.loc[(L0['Channel'] == channel) & (L0['Brand'] == brand), 'selectedmodels'] = 'No'

                    #         # Find the row in filtered_data where 'selectedmodels' is 'Yes'
                    #         selected_row = group[group['selectedmodels'] == 'Yes']

                    #         if not selected_row.empty:
                    #             closest_index = selected_row.index[0]  # Get the first matching index

                    #             # Update only the 'selectedmodels' column in L0
                    #             L0.loc[closest_index, 'selectedmodels'] = 'Yes'




                        # st.session_state["L0"] = L0.copy()

                        # # Store updated data in session state with file name
                        # file_name = st.session_state["file_name"]
                        # st.session_state[file_name] = L0.copy()

                                    




                    with col6:
                        # graph_option = st.radio(f"Choose graph type for {brand}", options=["MCV", "CSF"], key=f"graph_option_{brand}")
                        # options=["CSF","MCV"]
                        # graph_option = st.pills(f"Choose graph type for {brand}",options,selection_mode="single",default="CSF",key=f"graph_option_{brand}")

                        # if graph_option == "MCV":
                        #     # Sort data by 'MCV.MCV' in ascending order
                        #     sorted_data = filtered_data.sort_values(by='MCV.MCV', ascending=True)
                        #     sorted_data['Index'] = sorted_data.index.astype(str)
                                                
                        #     # Plotly chart for MCV with a better color palette
                        #     fig_mcv = px.bar(
                        #         sorted_data,
                        #         x='Index',  # Use the 'Index' column for x-axis
                        #         y='MCV.MCV',
                        #         template="plotly_white",
                        #         color='RPIto',  # Color bars based on the 'RPIto' column
                        #         text_auto=True,  # Display y-axis values on top of bars
                        #         category_orders={"Index": sorted_data['Index'].tolist()},  # Ensure bars follow the sorted order
                        #         color_discrete_sequence=px.colors.qualitative.Set3,  # Use a better color palette
                        #         hover_data={
                        #             "RPIto":False,
                        #             "Index": True, 
                        #             "Adj.Rsq": ':.2f', 
                        #             "AIC": ':.2f'
                        #         } 
                        #     )

                        #     # Update layout to remove title and position legend at top-left
                        #     fig_mcv.update_layout(
                        #         title="",  # Set title to empty string to avoid 'undefined'
                        #         xaxis=dict(title="Index", color='black', showgrid=False, showticklabels=False),
                        #         yaxis=dict(title="MCV", color='black', showgrid=False, tickformat=".1f"),
                        #         plot_bgcolor='white',
                        #         paper_bgcolor='white',
                        #         font_color='black',
                        #         legend=dict(
                        #             orientation="h",  # Horizontal orientation
                        #             xanchor="left",  # Anchor to the left
                        #             x=0,  # Position at the left edge
                        #             yanchor="bottom",  # Anchor to the bottom of the legend
                        #             y=1.02  # Position above the chart
                        #         )
                        #     )

                        #     # Display the chart
                        #     st.plotly_chart(fig_mcv, use_container_width=True)


                        # elif graph_option == "CSF":
                        #     # Sort data by 'CSF.CSF' in ascending order
                        #     sorted_data = filtered_data.sort_values(by='CSF.CSF', ascending=True)
                        #     sorted_data['Index'] = sorted_data.index.astype(str)
                            
                        #     # Plotly chart for CSF
                        #     fig_csf = px.bar(
                        #         sorted_data,
                        #         x='Index',  # Ensure index is treated as string for display
                        #         y='CSF.CSF',
                        #         template="plotly_white",
                        #         # color_discrete_sequence=["#FFD700"],
                        #         color='RPIto',
                        #         text_auto=True,  # Display y-axis values on top of bars
                        #         category_orders={"Index": sorted_data['Index'].tolist()},
                        #         color_discrete_sequence=px.colors.qualitative.Set3,  # Use a better color palette
                        #         hover_data={
                        #             "RPIto":False,
                        #             "Index": True, 
                        #             "Adj.Rsq": ':.2f', 
                        #             "AIC": ':.2f'
                        #         }
                        #     )
                        #     fig_csf.update_layout(
                        #         title="",
                        #         xaxis=dict(title="Index", color='black', showgrid=False,showticklabels=False),
                        #         yaxis=dict(title="CSF", color='black', showgrid=False,tickformat=".2f"),
                        #         plot_bgcolor='white',
                        #         paper_bgcolor='white',
                        #         font_color='black',
                        #         legend=dict(
                        #             orientation="h",  # Horizontal orientation
                        #             xanchor="left",  # Anchor to the left
                        #             x=0,  # Position at the left edge
                        #             yanchor="bottom",  # Anchor to the bottom of the legend
                        #             y=1.02  # Position above the chart
                        #         )
                        #     )
                        #     st.plotly_chart(fig_csf, use_container_width=True)

                        options = ["CSF", "MCV"]
                        graph_option = st.pills(f"Choose graph type for {brand}", options, selection_mode="single", default="CSF", key=f"graph_option_{brand}")

                        # Define sorting options
                        # sort_options_mcv = ["MCV.MCV", "Adj.Rsq", "AIC"]
                        # sort_options_csf = ["CSF.CSF", "Adj.Rsq", "AIC"]

                        if graph_option == "MCV":
                            sort_by = st.radio(
                                "SORT BY:", ["MCV.MCV", "Adj.Rsq", "AIC"], index=0, horizontal=True, key=f"sort_option_mcv_{brand}"
                            )

                            
                            
                            # Sort data based on selected column
                            sorted_data = filtered_data.sort_values(by=sort_by, ascending=True)
                            sorted_data['Index'] = sorted_data.index.astype(str)

                            fig_mcv = px.bar(
                                sorted_data,
                                x='Index',
                                y='MCV.MCV',
                                template="plotly_white",
                                color='RPIto',
                                text_auto=True,
                                category_orders={"Index": sorted_data['Index'].tolist()},
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                hover_data={
                                    "RPIto": False,
                                    "Index": True,
                                    "Adj.Rsq": ':.2f',
                                    "AIC": ':.2f'
                                }
                            )

                            fig_mcv.update_layout(
                                title="",
                                xaxis=dict(title="Index", color='black', showgrid=False, showticklabels=False),
                                yaxis=dict(title="MCV", color='black', showgrid=False, tickformat=".1f"),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font_color='black',
                                legend=dict(
                                    orientation="h",
                                    xanchor="left",
                                    x=0,
                                    yanchor="bottom",
                                    y=1.02
                                )
                            )

                            st.plotly_chart(fig_mcv, use_container_width=True)

                        elif graph_option == "CSF":
                            sort_by = st.radio(
                                    "SORT BY:", ["CSF.CSF", "Adj.Rsq", "AIC"], index=0, horizontal=True, key=f"sort_option_csf_{brand}"
                                )
                                                        
                            # Sort data based on selected column
                            sorted_data = filtered_data.sort_values(by=sort_by, ascending=True)
                            sorted_data['Index'] = sorted_data.index.astype(str)

                            fig_csf = px.bar(
                                sorted_data,
                                x='Index',
                                y='CSF.CSF',
                                template="plotly_white",
                                color='RPIto',
                                text_auto=True,
                                category_orders={"Index": sorted_data['Index'].tolist()},
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                hover_data={
                                    "RPIto": False,
                                    "Index": True,
                                    "Adj.Rsq": ':.2f',
                                    "AIC": ':.2f'
                                }
                            )

                            fig_csf.update_layout(
                                title="",
                                xaxis=dict(title="Index", color='black', showgrid=False, showticklabels=False),
                                yaxis=dict(title="CSF", color='black', showgrid=False, tickformat=".2f"),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font_color='black',
                                legend=dict(
                                    orientation="h",
                                    xanchor="left",
                                    x=0,
                                    yanchor="bottom",
                                    y=1.02
                                )
                            )

                            st.plotly_chart(fig_csf, use_container_width=True)



                    

                        

                    # options=["MAKE SELECTIONS","SUBMIT SELECTIONS"]
                    # submit_option = st.pills(f"",options,selection_mode="single",default="SUBMIT SELECTIONS",key=f"show_data_{brand}")
                    # # # Checkbox for brand-specific data
                    # # # if st.checkbox(f"Show Data for {brand}", key=f"show_data_{brand}",value=True):
                    # if submit_option=="MAKE SELECTIONS":

                    with st.expander("SHOW DATA"):

                        # st.error("CLICK ON 'SUBMIT SELECTIONS' TO PERMANENTLY SAVE THE CHANGES.")
                        # Apply global filters to the full dataset
                        global_filtered_data = st.session_state["L0"].copy()
                        # if selected_categories:
                        #     global_filtered_data = global_filtered_data[global_filtered_data['Category'].isin(selected_categories)]
                        # if selected_markets:
                        #     global_filtered_data = global_filtered_data[global_filtered_data['Market'].isin(selected_markets)]
                        if selected_channels:
                            global_filtered_data = global_filtered_data[global_filtered_data['Channel'].isin(selected_channels)]

                        # Extract the brand-specific data from the globally filtered data
                        brand_data = global_filtered_data.loc[global_filtered_data['Brand'] == brand].copy()

                        # Apply brand-specific filters
                        filtered_data = brand_data.copy()
                        if methods:
                            filtered_data = filtered_data[filtered_data['method'].isin(methods)]

                        
 
                        if rpito_filter:
                            filtered_data = filtered_data[filtered_data['RPIto'].isin(rpito_filter)]


                        if actualdistvar_filter:
                            filtered_data = filtered_data[filtered_data['actualdistvar'].isin(actualdistvar_filter)]

                        filtered_data = filtered_data[
                            (filtered_data['Adj.Rsq'] >= adjrsquare_filter[0]) & 
                            (filtered_data['Adj.Rsq'] <= adjrsquare_filter[1])
                        ]
                        filtered_data = filtered_data[
                            (filtered_data['CSF.CSF'] >= csf_filter[0]) & 
                            (filtered_data['CSF.CSF'] <= csf_filter[1])
                        ]
                        filtered_data = filtered_data[
                            (filtered_data['AIC'] >= aic_filter[0]) & 
                            (filtered_data['AIC'] <= aic_filter[1])
                        ]

                        # st.session_state["filtered_data"]=filtered_data



                        #     # Ensure the 'selectedmodels' column exists
                        # filtered_data['selectedmodels'] = 'No'

                        # # Function to update selectedmodels for each (Channel, Brand) group
                        # def update_selectedmodels(group):
                        #     if not group.empty:
                        #         median_csf = group['CSF.CSF'].median()  # Find the median
                        #         closest_index = (group['CSF.CSF'] - median_csf).abs().idxmin()  # Get index closest to median
                        #         group.loc[:, 'selectedmodels'] = 'No'  # Reset all to 'No'
                        #         group.loc[closest_index, 'selectedmodels'] = 'Yes'  # Set the closest row as 'Yes'
                        #     return group

                        # # Apply function to ensure only one 'Yes' per (Channel, Brand) combination
                        # filtered_data = filtered_data.groupby(['Channel', 'Brand'], group_keys=False).apply(update_selectedmodels)

                        # # Store the updated filtered_data in session state
                        # st.session_state["filtered_data"] = filtered_data

                        # # Update only the 'selectedmodels' column in L0 for rows marked as 'Yes' in filtered_data
                        # if "L0" in st.session_state:
                        #     L0 = st.session_state["L0"]

                        #     for (channel, brand), group in filtered_data.groupby(['Channel', 'Brand']):
                        #         # Reset all 'selectedmodels' to 'No' for this (Channel, Brand) in L0
                        #         L0.loc[(L0['Channel'] == channel) & (L0['Brand'] == brand), 'selectedmodels'] = 'No'

                        #         # Find the row in filtered_data where 'selectedmodels' is 'Yes'
                        #         selected_row = group[group['selectedmodels'] == 'Yes']

                        #         if not selected_row.empty:
                        #             closest_index = selected_row.index[0]  # Get the first matching index

                        #             # Update only the 'selectedmodels' column in L0
                        #             L0.loc[closest_index, 'selectedmodels'] = 'Yes'

                        # # Store the updated L0 in session state **before saving under file_name**
                        #     st.session_state["L0"] = L0.copy()

                        #     # Store updated data in session state with file name
                        #     file_name = st.session_state["file_name"]
                        #     st.session_state[file_name] = L0.copy()




                        # # Incorporate previously saved changes, if any
                        # if f"filtered_data_{brand}" in st.session_state:
                        #     # Update the filtered data with previously saved changes
                        #     modified_rows = st.session_state[f"filtered_data_{brand}"]
                        #     filtered_data.update(modified_rows)

                        # # Reset 'selectedmodels' column if not present
                        # if "selectedmodels" not in filtered_data.columns:
                        #     filtered_data['selectedmodels'] = 'No'

                        # # Calculate the default selection based on the median of CSF.CSF
                        # if not filtered_data.empty:
                        #     median_csf_index = (
                        #         (filtered_data['CSF.CSF'] - filtered_data['CSF.CSF'].median())
                        #         .abs()
                        #         .idxmin()
                        #     )
                        # else:
                        #     median_csf_index = None

                        # # Display the table with a selectbox to choose a row
                        # # st.write("Select a row to choose as the selected method:")
                        # selected_index = st.selectbox(
                        #     f"Select index for {brand}",
                        #     options=filtered_data.index,
                        #     index=filtered_data.index.get_loc(median_csf_index) if median_csf_index in filtered_data.index else 0,
                        #     format_func=lambda x: f"{x}"  # Optional: Add more clarity to selection
                        # )

                        # # Track the state for each brand selection
                        # if brand not in st.session_state:
                        #     st.session_state[brand] = {'previous_states': [], 'last_selected_index': None}  # Initialize stack and last index

                        # # If a new 'Yes' is selected, save the current state before the change
                        # if selected_index != st.session_state[brand].get('last_selected_index', None):
                        #     # Save the current data before applying the new 'Yes'
                        #     st.session_state[brand]['previous_states'].append(st.session_state["L0"].copy())
                        #     st.session_state[brand]['last_selected_index'] = selected_index  # Update the last selected index

                        # # Now handle the "Revert" button
                        # # Create two columns to place the buttons side by side
                        # col8, col9 = st.columns([5, 1])  # Adjust the column widths as needed

                        # # First button: Save Selected Method
                        # with col8:
                        #     if st.button(f"Save Selected Method for {brand}", key=f"save_selected_method_{brand}"):
                        #         # Check if there are multiple 'Yes' for the current brand under the global filters
                        #         global_brand_filter = (
                        #             (st.session_state["L0"]['Brand'] == brand) &
                        #             # (st.session_state["L0"]['Market'].isin(selected_markets)) &
                        #             # (st.session_state["L0"]['Category'].isin(selected_categories)) &
                        #             (st.session_state["L0"]['Channel'].isin(selected_channels))
                        #         )

                        #         # Reset if there is already a 'Yes' for the brand under the filters
                        #         if (st.session_state["L0"].loc[global_brand_filter, 'selectedmodels'] == 'Yes').sum() > 0:
                        #             # Reset all rows for the current brand and filters to 'No'
                        #             st.session_state["L0"].loc[global_brand_filter, 'selectedmodels'] = 'No'

                        #         # Set the selected row's 'selectedmodels' column to 'Yes'
                        #         st.session_state["L0"].loc[selected_index, 'selectedmodels'] = 'Yes'

                        #         # Update the filtered data for the current brand
                        #         filtered_data['selectedmodels'] = st.session_state["L0"].loc[
                        #             filtered_data.index, 'selectedmodels'
                        #         ]

                        #         st.session_state["L0"].loc[filtered_data.index, 'selectedmodels'] = filtered_data['selectedmodels']
                        #         file_name = st.session_state["file_name"]
                        #         st.session_state[file_name] = st.session_state["L0"].copy() 

                        #         st.success(f"Selected method for {brand} saved successfully!")

                        # # Second button: Revert Selection
                        # with col9:
                        #     if st.button(f"REVERT", key=f"revert_selected_method_{brand}"):
                        #         if st.session_state[brand]['previous_states']:
                        #             # Pop the most recent state to revert to the previous selection
                        #             last_state = st.session_state[brand]['previous_states'].pop()
                        #             st.session_state["L0"] = last_state.copy()  # Revert to the last state

                        #             # Update the filtered data for the current brand with reverted selections
                        #             filtered_data['selectedmodels'] = st.session_state["L0"].loc[
                        #                 filtered_data.index, 'selectedmodels'
                        #             ]

                        #             st.session_state["L0"].loc[filtered_data.index, 'selectedmodels'] = filtered_data['selectedmodels']
                        #             file_name = st.session_state["file_name"]
                        #             st.session_state[file_name] = st.session_state["L0"].copy()

                        #             st.success(f"Reverted!")
                        #         else:
                        #             st.warning("No previous state found to revert to!")
                        
                        # # filtered_data_reset = filtered_data.reset_index()

                        # Display the filtered data for this brand, with the updated 'selectedmodels' after save/revert
                        st.dataframe(filtered_data[['Market', 'Category', 'Channel', 'MCV.MCV', 'CSF.CSF', 'method', 'selectedmodels', 'RPIto', 'Adj.Rsq', 'AIC','actualdistvar']])


                    # elif submit_option=="SUBMIT SELECTIONS":
                    #     st.success("TO SELECT MODELS, PLEASE CLICK ON 'MAKE SELECTIONS' BUTTON.")

                        

                    st.markdown('<hr class="thin">', unsafe_allow_html=True)  # Add thin line after each brand

                # Show the final modified data after changes
                if st.button("Show Final Modified Data"):
                    # # Combine the original data with any updated rows
                    # final_modified_data = st.session_state["data"].copy()
                    
                    file_name = st.session_state["file_name"]
                    original_columns = st.session_state["L0"].columns.tolist()
                    st.session_state[file_name].columns = original_columns

                    # st.session_state[file_name] = final_modified_data

                    download_file_name = f"{file_name}.csv" if not file_name.endswith(".csv") else file_name

                    # st.write(f"Final Modified Data (File Name: {file_name}):")
                    st.dataframe(st.session_state[file_name])

                    st.download_button("Download the final modified data", 
                            data=st.session_state[file_name].to_csv(index=False), 
                            file_name=download_file_name, 
                            mime="text/csv")

                st.markdown('<hr class="thin">', unsafe_allow_html=True)






    with col2:


        if uploaded_file0:

        # st.markdown(
        #     """
        #     <style>
        #         /* Ensure AgGrid stays fixed in the viewport */
        #         .streamlit-expanderHeader {
        #             position: sticky;
        #             top: 330px;
        #             z-index: 1000;
        #         }
        #         /* More specific targeting for AgGrid to stay fixed */
        #         div[data-testid="stAgGrid"] {
        #             position: fixed;
        #             top: 330px;
        #             right: 1%;
        #             width: 40%;
        #             z-index: 1000;
        #         }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )


        # Custom CSS to make only the specific DataFrame sticky
        # Apply CSS to make ONLY this specific DataFrame sticky
        # Apply CSS to make ONLY this specific DataFrame sticky
        # Apply CSS to make ONLY this specific DataFrame sticky
        # st.markdown(
        #     """
        #     <style>
        #         /* Target only the DataFrame inside the sticky-container */
        #         [data-testid="stVerticalBlock"] > div[data-testid="column"] > div#sticky-container {
        #             position: fixed;
        #             top: 100px;  /* Adjust based on navbar */
        #             right: 20px;
        #             width: 30%;
        #             z-index: 1000;
        #             background: white;
        #             box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        #         }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )

            if "MSP_L0" not in st.session_state:
                st.session_state["MSP_L0"] = None


        
            # st.markdown('<hr class="thick">', unsafe_allow_html=True)

            # options = ["CONFIGURATION"]
            # option = st.pills(f"", options,selection_mode="single",default="CONFIGURATION",)

            # if option == "CONFIGURATION":
            #     # Define available options
            #     modeling_type_options = [1, 2]
            #     msp_options = ["Yes", "No"]
            #     variable_options = ['NA', 'Market', 'Channel', 'Region', 'Category',
            #                         'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
            #     periodicity_options = ["Daily", "Weekly", "Monthly"]

            #     # Function to get a valid index
            #     def get_valid_index(option_list, stored_value, default_value):
            #         return option_list.index(stored_value) if stored_value in option_list else option_list.index(default_value)

            #     # Initialize session state if not already set
            #     if "config_data" not in st.session_state:
            #         st.session_state.config_data = {
            #             "Modeling Type": 2,
            #             "MSP at L0 (Yes/No)": "Yes",
            #             "Variable to use as L0": "Brand",
            #             "MSP at L2 (Yes/No)": "No",
            #             "Variable to use as L2": "NA",
            #             "MSP at L3 (Yes/No)": "No",
            #             "Variable to use as L3": "NA",
            #             "Periodicity of Data (Daily/Weekly/Monthly)": "Weekly",
            #             "Reference Period": 1
            #         }

            #     # Get valid indices for selectbox
            #     modeling_type_index = get_valid_index(modeling_type_options, st.session_state.config_data["Modeling Type"], 2)
            #     msp_l0_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L0 (Yes/No)"], "Yes")
            #     msp_l2_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L2 (Yes/No)"], "No")
            #     msp_l3_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L3 (Yes/No)"], "No")
            #     variable_l0_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L0"], "Brand")
            #     variable_l2_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L2"], "NA")
            #     variable_l3_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L3"], "NA")
            #     periodicity_index = get_valid_index(periodicity_options, st.session_state.config_data["Periodicity of Data (Daily/Weekly/Monthly)"], "Daily")

            #     # Create a form for user input
            #     with st.form("config_form"):
            #         modeling_type = st.selectbox("Modeling Type", options=modeling_type_options, index=modeling_type_index)

            #         col10, col11 = st.columns(2)
            #         with col10:
            #             msp_l0 = st.selectbox("MSP at L0 (Yes/No)", options=msp_options, index=msp_l0_index)
            #             msp_l2 = st.selectbox("MSP at L2 (Yes/No)", options=msp_options, index=msp_l2_index)
            #             msp_l3 = st.selectbox("MSP at L3 (Yes/No)", options=msp_options, index=msp_l3_index)
            #             periodicity = st.selectbox("Periodicity of Data (Daily/Weekly/Monthly)", options=periodicity_options, index=periodicity_index)
                    
            #         with col11:
            #             variable_l0 = st.selectbox("Variable to use as L0", options=variable_options, index=variable_l0_index)
            #             variable_l2 = st.selectbox("Variable to use as L2", options=variable_options, index=variable_l2_index)
            #             variable_l3 = st.selectbox("Variable to use as L3", options=variable_options, index=variable_l3_index)
            #             reference_period = st.number_input("Reference Period", min_value=1, value=st.session_state.config_data["Reference Period"])

            #         # Submit button
            #         submit_button = st.form_submit_button("Save Configuration")

            #         # If submitted, update session state
            #         if submit_button:
            #             st.session_state.config_data = {
            #                 "Modeling Type": modeling_type,
            #                 "MSP at L0 (Yes/No)": msp_l0,
            #                 "Variable to use as L0": variable_l0,
            #                 "MSP at L2 (Yes/No)": msp_l2,
            #                 "Variable to use as L2": variable_l2,
            #                 "MSP at L3 (Yes/No)": msp_l3,
            #                 "Variable to use as L3": variable_l3,
            #                 "Periodicity of Data (Daily/Weekly/Monthly)": periodicity,
            #                 "Reference Period": reference_period
            #             }
            #             st.write("Configuration saved successfully!")

            #     # Display current configuration
            #     config_df = pd.DataFrame([st.session_state.config_data])
            #     st.write("Current Configuration:")
            #     st.dataframe(config_df)
                    
            # st.markdown('<hr class="thick">', unsafe_allow_html=True)

            

            # Fetch D0 file from session state
            if "D0" in st.session_state["uploaded_files"]:
                D0 = st.session_state["uploaded_files"]["D0"]

                # # Rename date column if needed
                # if 'date' in D0.columns:
                #     D0.rename(columns={'date': 'Date'}, inplace=True)

                # Filter out unwanted brands
                if 'Brand' in D0.columns:
                    D0 = D0[~D0['Brand'].str.lower().isin(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])]

            else:
                st.sidebar.warning("D0 file not found in session state.")

            # Fetch Weighted Average Files from session state
            weighted_files = [
                "Wtd_avg_MCV_Brand", "Wtd_avg_MCV_PackType", "Wtd_avg_MCV_PackSize",
                "Wtd_avg_MCV_PPG", "Wtd_avg_MCV_Variant", "Wtd_avg_MCV_Category",
                "Wtd_avg_MCV_SubCategory"
            ]

            df_dict = {
                "Brand": pd.DataFrame(), "PackType": pd.DataFrame(), "PackSize": pd.DataFrame(),
                "PPG": pd.DataFrame(), "Variant": pd.DataFrame(), "Category": pd.DataFrame(),
                "SubCategory": pd.DataFrame()
            }

            for file_name in weighted_files:
                if file_name in st.session_state:
                    try:
                        df = st.session_state[file_name]  # ✅ Directly use stored DataFrame
                        
                        # # Print columns for debugging
                        # st.write(f"Columns in {file_name}:", df.columns.tolist())


                        # Check if 'selectedmodels' exists (lowercase)
                        if 'selectedmodels' in df.columns:
                            df = df[df['selectedmodels'].str.lower() == 'yes']
                            key = file_name.replace("Wtd_avg_MCV_", "").replace(".xlsx", "")
                            df_dict[key] = df  # ✅ Store filtered DataFrame
                            # if df_L0 is not None:
                            #     st.session_state.df_L0 = df_L0


                        else:
                            st.sidebar.warning(f"'{file_name}' does not contain 'selectedmodels' column after processing.")

                    except Exception as e:
                        st.sidebar.error(f"Error processing {file_name}: {e}")
                # else:
                #     st.sidebar.warning(f"{file_name} not found in session state.")

            # st.markdown('<hr class="thin">', unsafe_allow_html=True)


        #-----------------------------------------------------------------------------------------------------------------------------------------
            # # Display processed datasets
            # st.subheader("Processed D0 Data")
            # if "D0" in st.session_state["uploaded_files"]:
            #     st.dataframe(D0)
            # else:
            #     st.write("No D0 data available.")

            # st.subheader("Processed Weighted Average Files")
            # for key, df in df_dict.items():
            #     st.write(f"### {key}")
            #     if not df.empty:
            #         st.dataframe(df)
            #     else:
            #         st.write("No data available.")


        # User Input for Time Period (Weekly data)

            if "config_data" in st.session_state:
                config_df=pd.DataFrame([st.session_state.config_data])

                # Check the conditions
                if (config_df['Periodicity of Data (Daily/Weekly/Monthly)']
                    .astype(str)  # Ensure it's a string Series
                    .str.strip()  # Remove extra spaces
                    .str.lower()  # Convert to lowercase
                    .eq('weekly') # Check for 'weekly'
                    .any()):

                    from datetime import timedelta
                    import pandas as pd

                    def defined_period_data(dataframe, weeks):
                        """
                        Filters the DataFrame to include rows from the most recent weeks based on the Date column.
                        Converts all dates to short format (YYYY-MM-DD) for consistency.
                        Assumes the data is strictly weekly.

                        Parameters:
                        - dataframe (pd.DataFrame): The input DataFrame with a date column.
                        - weeks (int): Number of weeks of data to retain.

                        Returns:
                        - pd.DataFrame: Filtered DataFrame with data from the specified period.
                        """
                        # Detect the date column (case-insensitive search for "date")
                        date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
                        if not date_column:
                            raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                        
                        # Function to handle multiple date formats and remove time if present
                        def parse_date(date):
                            for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
                                        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):  # Supported formats
                                try:
                                    # Convert to datetime
                                    parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
                                    if pd.notnull(parsed_date):
                                        return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
                                except ValueError:
                                    continue
                            raise ValueError(f"Date '{date}' does not match any supported formats.")
                        
                        # Apply parsing and conversion to short format to the date column
                        dataframe[date_column] = dataframe[date_column].apply(parse_date)
                        
                        # Convert the date column back to datetime for further processing
                        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                        
                        # Sort the DataFrame by date
                        dataframe = dataframe.sort_values(by=date_column)
                        
                        # Calculate cutoff date based on weeks
                        recent_date = dataframe[date_column].max()
                        cutoff_date = recent_date - timedelta(weeks=int(weeks))
                        
                        # Filter the DataFrame
                        filtered_df = dataframe[dataframe[date_column] > cutoff_date]
                        
                        return filtered_df


                    # Example Usage:
                    Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

                else:
                    print("Conditions not met. Skipping execution.")




        #--------------------------------------------------------

        # User Input for Time Period (Monthly data)

            if "config_data" in st.session_state:
                config_df=pd.DataFrame([st.session_state.config_data])
                # Check the conditions
                if (config_df['Periodicity of Data (Daily/Weekly/Monthly)'].str.strip().str.lower().eq('monthly').any()):

                    from datetime import timedelta
                    import pandas as pd

                    def defined_period_data(dataframe, months):
                        """
                        Filters the DataFrame to include rows from the most recent months based on the Date column.
                        Converts all dates to short format (YYYY-MM-DD) for consistency.
                        Assumes the data is strictly monthly.

                        Parameters:
                        - dataframe (pd.DataFrame): The input DataFrame with a date column.
                        - months (int): Number of months of data to retain.

                        Returns:
                        - pd.DataFrame: Filtered DataFrame with data from the specified period.
                        """
                        # Detect the date column (case-insensitive search for "date")
                        date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
                        if not date_column:
                            raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                        
                        # Function to handle multiple date formats and remove time if present
                        def parse_date(date):
                            for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
                                        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):  # Supported formats
                                try:
                                    # Convert to datetime
                                    parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
                                    if pd.notnull(parsed_date):
                                        return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
                                except ValueError:
                                    continue
                            raise ValueError(f"Date '{date}' does not match any supported formats.")
                        
                        # Apply parsing and conversion to short format to the date column
                        dataframe[date_column] = dataframe[date_column].apply(parse_date)
                        
                        # Convert the date column back to datetime for further processing
                        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                        
                        # Sort the DataFrame by date
                        dataframe = dataframe.sort_values(by=date_column)
                        
                        # Convert the date column to periods (monthly)
                        dataframe['Month_Sorting'] = dataframe[date_column].dt.to_period('M')
                        
                        # Calculate the cutoff month
                        recent_month = dataframe['Month_Sorting'].max()
                        cutoff_month = recent_month - months
                        
                        # Filter the DataFrame based on the cutoff month
                        filtered_df = dataframe[dataframe['Month_Sorting'] > cutoff_month]
                        
                        # Drop the temporary 'Month' column
                        filtered_df = filtered_df.drop(columns=['Month_Sorting'], errors='ignore')
                        
                        return filtered_df


                    # Example Usage:
                    Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

                else:
                    print("Conditions not met. Skipping execution.")

        #---------------------------------------------------------------------------

        # User Input for Time Period (Daily data)

            if "config_data" in st.session_state:
                config_df=pd.DataFrame([st.session_state.config_data])

                # Check the conditions
                if (config_df['Periodicity of Data (Daily/Weekly/Monthly)'].str.strip().str.lower().eq('daily').any()):

                    from datetime import timedelta
                    import pandas as pd

                    def defined_period_data(dataframe, days):
                        """
                        Filters the DataFrame to include rows from the most recent days based on the Date column.
                        Converts all dates to short format (YYYY-MM-DD) for consistency.
                        Assumes the data is strictly daily.

                        Parameters:
                        - dataframe (pd.DataFrame): The input DataFrame with a date column.
                        - days (int): Number of days of data to retain.

                        Returns:
                        - pd.DataFrame: Filtered DataFrame with data from the specified period.
                        """
                        # Detect the date column (case-insensitive search for "date")
                        date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
                        if not date_column:
                            raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                        
                        # Function to handle multiple date formats and remove time if present
                        def parse_date(date):
                            for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
                                        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):   # Supported formats
                                try:
                                    # Convert to datetime
                                    parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
                                    if pd.notnull(parsed_date):
                                        return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
                                except ValueError:
                                    continue
                            raise ValueError(f"Date '{date}' does not match any supported formats.")
                        
                        # Apply parsing and conversion to short format to the date column
                        dataframe[date_column] = dataframe[date_column].apply(parse_date)
                        
                        # Convert the date column back to datetime for further processing
                        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                        
                        # Sort the DataFrame by date
                        dataframe = dataframe.sort_values(by=date_column)
                        
                        # Calculate the cutoff date based on days
                        recent_date = dataframe[date_column].max()
                        cutoff_date = recent_date - timedelta(days=days)
                        
                        # Filter the DataFrame based on the cutoff date
                        filtered_df = dataframe[dataframe[date_column] > cutoff_date]
                        
                        return filtered_df 


                    # Example Usage:
                    Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

                else:
                    print("Conditions not met. Skipping execution.")


        #---------------------------------------------------------------------------

        # # Taking User Input for Brand Names

        #     if "config_data" in st.session_state:
        #         config_df=pd.DataFrame([st.session_state.config_data])
        #         # Check the conditions
        #         if (config_df['Modeling Type'].eq(1).any() and 
        #             config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
        #             config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):
                    
        #             # Take user input as a single string
        #             user_input = input("Enter Elements of Variable which is used as L0, separated by commas: ")

        #             # Split the input string into a list and convert each item to lowercase
        #             brand_names = [name.strip().lower() for name in user_input.split(',')]

        #             print(brand_names)
        #         else:
        #             print("Conditions not met. Skipping execution.")



        #--------------------------------------------------------------

        # User Input for L0 Variable

            def select_defined_dataframe(df_dict, selected_key):
                """
                Selects a single DataFrame from a dictionary based on user input,
                ignoring those that are not defined.

                Args:
                    df_dict (dict): A dictionary where keys are DataFrame names (strings)
                                    and values are the DataFrame objects.
                    selected_key (str): The name of the DataFrame to select.


                Returns:
                    pd.DataFrame: The selected DataFrame, or None if the selection is invalid.
                """
                # Filter the dictionary to include only defined DataFrames
                defined_dfs = {key: df for key, df in df_dict.items() if isinstance(df, pd.DataFrame)}

                if not defined_dfs:
                    print("No DataFrames are defined!")
                    return None

                print("Available DataFrames:")
                for key in defined_dfs.keys():
                    print(f"- {key}")

                # Validate the input key
                if selected_key in defined_dfs:
                    print(f"\nSelected DataFrame: {selected_key}")
                    return defined_dfs[selected_key]
                else:
                    print("Invalid selection! Please try again.")
                    return None
                





            if "df_L0" not in st.session_state:
                st.session_state.df_L0 = None
                st.session_state.df_L0_modified = False  # Track if it was modified
                st.session_state.df_L0_original = None  # Store original values

            if "config_data" in st.session_state:
                config_df = pd.DataFrame([st.session_state.config_data])

                if config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any():
                    # Get user input outside the function
                    selected_key_L0 = config_df.loc[0, 'Variable to use as L0']
                    st.session_state.selected_key_L0 = selected_key_L0

                    # Load df_L0 only if it was never modified
                    if not st.session_state.df_L0_modified:
                        df_L0 = select_defined_dataframe(df_dict, selected_key_L0)
                        if df_L0 is not None:
                            st.session_state.df_L0 = df_L0.copy()
                            st.session_state.df_L0_original = df_L0.copy()  # Store original values
                        else:
                            st.error("df_L0 not found in session state!")

            # from streamlit_modal import Modal

            # # Define the columns to display
            # columns_to_show = ['Channel', 'Brand', 'MCV.MCV', 'CSF.CSF', 'method', 
            #                 'selectedmodels', 'RPIto', 'Adj.Rsq', 'AIC', 'actualdistvar']

            # with st.expander("Edit DataFrame"):
            #     st.write("### Edit df_L0")

            #     # Show only selected columns
            #     filtered_df = st.session_state.df_L0[columns_to_show].copy()

            #     # Set column config to make only 'MCV.MCV' editable
            #     column_config = {
            #         "MCV.MCV": st.column_config.NumberColumn("MCV.MCV", help="Edit this value"),
            #     }

            #     # Make a copy with 'MCV.MCV' editable, others read-only
            #     edited_df = st.data_editor(
            #         filtered_df, 
            #         column_config=column_config, 
            #         disabled=[col for col in columns_to_show if col != "MCV.MCV"]
            #     )

            #     col1, col2 = st.columns(2)

            #     with col1:
            #         if st.button("SAVE CHANGES"):
            #             if not edited_df.equals(filtered_df):  # Check if edits were made
            #                 st.session_state.df_L0["MCV.MCV"] = edited_df["MCV.MCV"].copy()  # Save only MCV.MCV column
            #                 st.session_state.df_L0_modified = True  # Mark as modified
            #                 st.success("Changes saved successfully!")
            #             else:
            #                 st.info("No changes detected. Using the original DataFrame.")

            #     with col2:
            #         if st.button("RESET"):
            #             if st.session_state.df_L0_original is not None:
            #                 st.session_state.df_L0["MCV.MCV"] = st.session_state.df_L0_original["MCV.MCV"].copy()
            #                 st.session_state.df_L0_modified = False  # Reset modification flag
            #                 st.success("DataFrame reset to original values!")



        #---------------------------------------------------------------

        # User Input for L2 Variable

            # if "config_data" in st.session_state:
            #     config_df=pd.DataFrame([st.session_state.config_data])
            #     if (config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and 
            #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

            #         # Get user input outside the function
            #         selected_key_L2 = config_df.loc[0, 'Variable to use as L2']

            #         # Call the function with the user input
            #         df_L2 = select_defined_dataframe(df_dict, selected_key_L2)

            #         if df_L2 is not None:
            #             st.session_state.df_L2 = df_L2
            #             print("\nHere is the selected DataFrame:")
            #             print(df_L2)
                        
            #     else:
            #         print("The value in 'MSP at L2 (Yes/No)' is not 'Yes'. Skipping execution.")



        #------------------------------------------------------

            # if "config_data" in st.session_state:
            #     config_df=pd.DataFrame([st.session_state.config_data])
            #     if (config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and 
            #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
            #         config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

            #         # Get user input outside the function
            #         selected_key_L3 = config_df.loc[0, 'Variable to use as L3']

            #         # Call the function with the user input
            #         df_L3 = select_defined_dataframe(df_dict, selected_key_L3)

            #         if df_L3 is not None:
            #             st.session_state.df_L3 = df_L3
            #             print("\nHere is the selected DataFrame:")
            #             print(df_L3)

            #     else:
            #         print("The value in 'MSP at L3 (Yes/No)' is not 'Yes'. Skipping execution.")



        #---------------------------------------------------------------

        # Selecting desired Volume Units Column from the DF

            def adjust_volume_based_on_other_df(original_df, other_df):
                """
                Adjusts the 'Volume' or 'VolumeUnits' columns in the original DataFrame based on the 'Vol_Var' column in another DataFrame.

                Parameters:
                    original_df (pd.DataFrame): The original DataFrame containing 'Volume' and 'VolumeUnits' columns.
                    other_df (pd.DataFrame): The other DataFrame containing the 'Vol_Var' column.

                Returns:
                    pd.DataFrame: The adjusted original DataFrame.

                Raises:
                    ValueError: If 'Vol_Var' is missing in other_df or contains invalid/multiple unique values.
                """
                # Ensure 'Vol_Var' column exists in the other DataFrame
                if 'Vol_Var' not in other_df.columns:
                    raise ValueError("The other DataFrame must contain a 'Vol_Var' column.")
                
                # Get unique values in 'Vol_Var'
                vol_var_values = other_df['Vol_Var'].unique()
                
                # Ensure 'Vol_Var' contains only one unique value
                if len(vol_var_values) != 1:
                    raise ValueError("The 'Vol_Var' column must contain a single unique value across the DataFrame.")
                
                vol_var_value = vol_var_values[0]
                
                # Adjust original_df based on 'Vol_Var' value
                if vol_var_value == 'Volume':
                    # Drop 'VolumeUnits' and rename 'Volume' in the original DataFrame
                    if 'VolumeUnits' in original_df.columns:
                        original_df = original_df.drop(columns=['VolumeUnits'])
                    original_df = original_df.rename(columns={'Volume': 'Volume'})
                
                elif vol_var_value == 'VolumeUnits':
                    # Drop 'Volume' and rename 'VolumeUnits' in the original DataFrame
                    if 'Volume' in original_df.columns:
                        original_df = original_df.drop(columns=['Volume'])
                    original_df = original_df.rename(columns={'VolumeUnits': 'Volume'})
                
                else:
                    raise ValueError("Invalid value in 'Vol_Var' column. Expected 'Volume' or 'VolumeUnits'.")

                return original_df




            def add_vol_var_column(target_df, reference_df, column_name):
                """
                Adds a new column to the target DataFrame based on the unique value in the reference DataFrame's Vol_Var column.
                
                Parameters:
                - target_df (pd.DataFrame): The target DataFrame where the column will be added.
                - reference_df (pd.DataFrame): The reference DataFrame containing the Vol_Var column.
                - column_name (str): The name of the column to be added to the target DataFrame.
                
                Returns:
                - pd.DataFrame: The updated target DataFrame with the new column added.
                """
                # Ensure the Vol_Var column exists in the reference DataFrame
                if "Vol_Var" not in reference_df.columns:
                    raise ValueError("The reference DataFrame must contain the 'Vol_Var' column.")
                
                # Check if all values in the Vol_Var column are the same
                unique_values = reference_df["Vol_Var"].unique()
                if len(unique_values) != 1:
                    raise ValueError("The 'Vol_Var' column in the reference DataFrame must have a single unique value.")
                
                # Get the unique value
                vol_var_value = unique_values[0]
                
                # Add the new column to the target DataFrame with the same value
                target_df[column_name] = vol_var_value
                
                return target_df









            D0['Channel'] = D0['Channel'].str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.replace("'", '', regex=False).str.lower()
            D0['Brand'] = D0['Brand'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['Market'] = D0['Market'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['PPG'] = D0['PPG'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['Variant'] = D0['Variant'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['Category'] = D0['Category'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['PackType'] = D0['PackType'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['PackSize'] = D0['PackSize'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
            D0['SubCategory'] = D0['SubCategory'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()




            if 'df_L0' in st.session_state:
                df_L0 = st.session_state.df_L0
                D0 = adjust_volume_based_on_other_df(D0, df_L0)
        
            else:
                st.error("df_L0 not found in session state!")

            D0_L0 = D0.copy()
            if 'selected_key_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0
                D0_L0[selected_key_L0] = D0_L0[selected_key_L0].astype('str')





            # Drop columns dynamically
            D0_L0 = D0_L0.drop(columns=[
                    'Year', 'Month', 'Week', 'BrCatId'
                ])
            


            if 'selected_key_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0
            # Group the DataFrame
                D0_L0 = (
                    D0_L0.groupby(['Market', 'Channel','Category',selected_key_L0, 'date'], as_index=False)
                    .agg({
                        'SalesValue': 'sum',  # Sum of SalesValue
                        'Volume': 'sum',      # Sum of Volume 
                    })
                )

        #--------------------------------------------------------------



            if "df_L0_L0" not in st.session_state:
                st.session_state.df_L0_L0 = None
                st.session_state.df_L0_L0_modified = False  # Track if it was modified
                st.session_state.df_L0_L0_original = None  # Store original values      


            if "df_L0" in st.session_state and "selected_key_L0" in st.session_state:
                df_L0 = st.session_state.df_L0
                selected_key_L0 = st.session_state.selected_key_L0

                columns_to_keep = ["Market", "Channel", selected_key_L0, "MCV.MCV"]

                # Load df_L0_L0 from session state if modified, else generate from df_L0
                if st.session_state.df_L0_L0_modified:
                    df_L0_L0 = st.session_state.df_L0_L0  # Keep the saved edited version
                else:
                    df_L0_L0 = df_L0[columns_to_keep].copy()  # Generate fresh copy from df_L0
                    df_L0_L0[selected_key_L0] = df_L0_L0[selected_key_L0].astype(str)
                    df_L0_L0["original_MCV"] = df_L0_L0["MCV.MCV"]

                    # Save the original version for reset purposes
                    st.session_state.df_L0_L0_original = df_L0_L0.copy()

                st.session_state.df_L0_L0 = df_L0_L0  # Ensure df_L0_L0 is always stored

                with st.expander("✏️EDIT MCV"):
                    st.write("")

                    # Define column configuration (only 'MCV.MCV' is editable)
                    column_config = {
                        "MCV.MCV": st.column_config.NumberColumn("MCV.MCV", help="Edit this value"),
                    }

                    # Editable DataFrame
                    edited_df_L0_L0 = st.data_editor(
                        df_L0_L0,
                        column_config=column_config,
                        disabled=[col for col in columns_to_keep if col != "MCV.MCV"] + ["original_MCV"],  # Disable all except 'MCV.MCV'
                        use_container_width=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("SAVE CHANGES"):
                            if not edited_df_L0_L0.equals(df_L0_L0):  # Check if edits were made
                                st.session_state.df_L0_L0 = edited_df_L0_L0.copy()  # Save changes persistently
                                st.session_state.df_L0_L0_modified = True  # Mark as modified
                                st.success("Changes saved! These changes will persist until reset.")

                            else:
                                st.info("No changes detected. Using the original DataFrame.")

                    with col2:
                        if st.button("RESET"):
                            if "df_L0_L0_original" in st.session_state:
                                st.session_state.df_L0_L0 = st.session_state.df_L0_L0_original.copy()  # Reset to original
                                st.session_state.df_L0_L0_modified = False  # Reset modification flag
                                st.success("DataFrame reset to original values!")


                    st.info("The above edits will be considered as final and will not change even if the models are reselected.\n\nIf you want to change selected models, please click on 'RESET' before change.")
            else:
                st.error("Either 'df_L0' or 'selected_key_L0' is not in session_state.")


            # Check if necessary session state variables exist
            # if 'df_L0' in st.session_state and 'selected_key_L0' in st.session_state:
            #     df_L0 = st.session_state.df_L0
            #     selected_key_L0 = st.session_state.selected_key_L0

            #     columns_to_keep = ['Market', 'Channel', selected_key_L0, 'MCV.MCV']  # Columns to keep
            #     df_L0_L0 = df_L0[columns_to_keep].copy()  # Create a copy
            #     df_L0_L0[selected_key_L0] = df_L0_L0[selected_key_L0].astype(str)  # Ensure column is string type

            #     with st.expander("Edit DataFrame"):
            #         st.write("### Edit df_L0_L0")

            #         # Define column configuration (only 'MCV.MCV' editable)
            #         column_config = {
            #             "MCV.MCV": st.column_config.NumberColumn("MCV.MCV", help="Edit this value"),
            #         }

            #         # Editable DataFrame
            #         edited_df_L0_L0 = st.data_editor(
            #             df_L0_L0,
            #             column_config=column_config,
            #             disabled=[col for col in columns_to_keep if col != "MCV.MCV"]  # Disable all except 'MCV.MCV'
            #         )

            #         col1, col2 = st.columns(2)

            #         with col1:
            #             if st.button("SAVE CHANGES"):
            #                 if not edited_df_L0_L0.equals(df_L0_L0):  # Check if edits were made
            #                     st.session_state.df_L0["MCV.MCV"] = edited_df_L0_L0["MCV.MCV"].copy()  # Save changes
            #                     st.session_state.df_L0_modified = True  # Mark as modified
            #                     st.success("Changes saved successfully!")
            #                 else:
            #                     st.info("No changes detected. Using the original DataFrame.")

            #         with col2:
            #             if st.button("RESET"):
            #                 if 'df_L0_original' in st.session_state:
            #                     st.session_state.df_L0["MCV.MCV"] = st.session_state.df_L0_original["MCV.MCV"].copy()
            #                     st.session_state.df_L0_modified = False  # Reset modification flag
            #                     st.success("DataFrame reset to original values!")
            # else:
            #     st.error("Either 'df_L0' or 'selected_key_L0' is not in session_state.")






            if 'selected_key_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0

                D0_L0['Revenue_Total'] = D0_L0.groupby(['Market', 'Channel', selected_key_L0])['SalesValue'].transform('sum')
                D0_L0['Volume_Total'] = D0_L0.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')
                D0_L0['Price_ModelPeriod'] = D0_L0['Revenue_Total']/D0_L0['Volume_Total'] 






            D0_filtered = defined_period_data(D0_L0, Reference_Period)



            if 'selected_key_L0' in st.session_state and 'df_L0_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0
                df_L0_L0=st.session_state.df_L0_L0

                D0_filtered['Revenue'] = D0_filtered.groupby(['Market', 'Channel', selected_key_L0])['SalesValue'].transform('sum')
                D0_filtered['Volume_definedperiod'] = D0_filtered.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')

                D0_filtered['Price'] = D0_filtered['Revenue']/D0_filtered['Volume_definedperiod'] 

                D0_filtered['Avg Volume'] = D0_filtered['Volume_definedperiod']/Reference_Period

                D0_filtered = D0_filtered.drop(columns=['date', 'SalesValue', 'Volume'])

                D0_filtered = D0_filtered.drop_duplicates(subset=['Market','Channel',selected_key_L0], keep='first')

                MSP_L0 = pd.merge(df_L0_L0,D0_filtered, on = ['Market','Channel',selected_key_L0], how='left')
        


            MSP_L0['Total_Sales_Per_Channel'] = MSP_L0.groupby(['Market','Channel'])['Revenue'].transform('sum')
            MSP_L0['MShare'] = MSP_L0['Revenue']/MSP_L0['Total_Sales_Per_Channel']

            MSP_L0.rename(columns={'Volume_definedperiod': 'Volume'}, inplace=True)

            MSP_L0['CS'] = 0.5 * (MSP_L0['MCV.MCV'] + MSP_L0['Price']) * MSP_L0['Avg Volume']

            MSP_L0.groupby(['Market','Channel'])['MShare'].sum()

            MSP_L0['Sum_CS_by_Channel'] = MSP_L0.groupby(['Market','Channel'])['CS'].transform('sum')
            MSP_L0['Sum_MShare_by_Channel'] = MSP_L0.groupby(['Market','Channel'])['MShare'].transform('sum')

            # Step 2: Calculate PVS for each row
            MSP_L0['PVS'] = (MSP_L0['CS'] / MSP_L0['Sum_CS_by_Channel']) * MSP_L0['Sum_MShare_by_Channel']

            MSP_L0['MSP'] = MSP_L0['PVS'] - MSP_L0['MShare']

            MSP_L0 = MSP_L0.drop(columns=['Sum_CS_by_Channel','Sum_MShare_by_Channel','PVS'])

            MSP_L0['CSF'] = MSP_L0['MCV.MCV'] / MSP_L0['Price']

            MSP_L0['NewMShare'] = MSP_L0['MShare'] + MSP_L0['MSP']

            MSP_L0['Price_elas'] = MSP_L0['Price_ModelPeriod'] /(MSP_L0['Price'] - MSP_L0['MCV.MCV'])

            MSP_L0.rename(columns={'MCV.MCV': 'MCV'}, inplace=True)

            MSP_L0['Market'] = MSP_L0['Market'].str.lower()
            MSP_L0['Category'] = MSP_L0['Category'].str.lower()


            if 'selected_key_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0
                columns_to_drop = [
                col for col in ['Region', 'Brand', 'SubCategory', 'Variant', 'PackType', 'PPG', 'PackSize']
                if col != selected_key_L0
                ]

                # Drop the remaining columns
                MSP_L0 = MSP_L0.drop(columns=columns_to_drop, errors='ignore')


            for file_name in weighted_files:
                if file_name in st.session_state:
                    try:
                        L0_Elasticity = st.session_state[file_name]

                    except Exception as e:
                        st.sidebar.error(f"Error processing {file_name}: {e}")


            result_dfs = []


            if 'selected_key_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0
                all_columns = list(L0_Elasticity.columns)

                # Ensure 'Category_elas' and 'beta0' exist
                if 'Category_elas' in all_columns and 'beta0' in all_columns:
                    # Find the positions of 'Category_elas' and 'beta0'
                    start_index = all_columns.index('Category_elas')
                    end_index = all_columns.index('beta0')

                    # Get the columns that lie between 'Category_elas' and 'beta0', excluding them
                    columns_in_range = all_columns[start_index + 1:end_index]
                else:
                    raise ValueError("'Category_elas' or 'beta0' column not found in the DataFrame.")

                # Define conditions (columns to include in the analysis)
                conditions = ['Market', 'Channel', selected_key_L0]
                valid_conditions = [cond for cond in conditions if cond in L0_Elasticity.columns]
                relevant_columns = valid_conditions + columns_in_range

                # Process each unique channel
                for channel in L0_Elasticity['Channel'].unique():
                    # Filter rows for the current channel
                    channel_df = L0_Elasticity[L0_Elasticity['Channel'] == channel]

                    # Process each unique brand within the current channel
                    for element in channel_df[selected_key_L0].unique():
                        # Filter rows for the current brand within the channel
                        element_df = channel_df[channel_df[selected_key_L0] == element].copy()

                        # # Apply multiple conditions
                        # element_df = element_df[
                        #     (element_df['method'].isin(methods_to_include)) &
                        #     (element_df['Price_pval'] == 'Yes') &
                        #     (element_df['Distribution_beta'] > 0) &
                        #     (element_df['Price_beta'] < 0)
                        # ]

                        # Ensure only relevant columns are included
                        element_df = element_df[[col for col in relevant_columns if col in element_df.columns]]

                        # Calculate the mean for the relevant columns in columns_in_range
                        for col in columns_in_range:
                            if col in element_df.columns:
                                # Calculate mean only for values greater than zero
                                element_df[col] = element_df[col][element_df[col] > 0].mean()

                        # Append the processed data to the result list
                        result_dfs.append(element_df)

                # Concatenate all processed DataFrames into a single DataFrame
                result_df = pd.concat(result_dfs, ignore_index=True)

                # Drop duplicates to retain one row per unique Market, Channel, and Brand
                result_df = result_df.drop_duplicates(subset=['Market', 'Channel', selected_key_L0], keep='first')

            

            MSP_L0 = pd.merge(MSP_L0,result_df, on = ['Market','Channel',selected_key_L0], how='left')
            MSP_L0 = MSP_L0.dropna(subset=['CSF'])

            MSP_L0.fillna('NA', inplace=True)



            if 'selected_key_L0' in st.session_state:
                selected_key_L0 = st.session_state.selected_key_L0
                # Dynamic column ordering
                unique_elements = [element for element in L0_Elasticity[selected_key_L0].unique()]

                #fixed_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand','Variant', 'PackType', 'PPG','MCV','Price_elas','Revenue', 'Volume','Price','CSF']  # Columns to place at the beginning
                fixed_columns = ['Market', 'Channel', 'Category', selected_key_L0, 'MCV','original_MCV','Price_elas','Revenue', 'Volume','Price','CSF']  # Columns to place at the beginning
                dot_columns = sorted([
                    col for col in MSP_L0.columns 
                    if '.' in col 
                    #or col.startswith('restofcategory') 
                    or col in unique_elements
                ])  # Columns with '.' in alphabetical order
                remaining_columns = ['Revenue_Total','Volume_Total', 'Price_ModelPeriod','CS','MShare','NewMShare','MSP']  # Columns to place at the end

                # Combine the desired order
                new_order = fixed_columns + dot_columns + remaining_columns

                # Reorder the DataFrame
                MSP_L0 = MSP_L0[new_order]

                # Assume `final_df` is the target DataFrame and `reference_df` is the other DataFrame.
                column_name_to_add = "Vol_Var"  # Name of the column to create in the final DataFrame
                MSP_L0 = add_vol_var_column(MSP_L0, df_L0, column_name_to_add)


    ## MSP_L0 for Type 2----------------------------------

            if "config_data" in st.session_state:
                config_df=pd.DataFrame([st.session_state.config_data])
                if (config_df['Modeling Type'].eq(2).any() and 
                    config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                    MSP_L0_T2 = MSP_L0.copy()
                    MSP_L0_T2['CsfPeriod'] = Reference_Period
                    MSP_L0_T2 = MSP_L0

                    st.session_state["MSP_L0"]=MSP_L0


                
                    # st.write('MSP_L0')

                    # st.dataframe(MSP_L0)  # Keep height limited



    # MSP_L0 fro Type 1 w/o MSP_L0L2------------------------------------------

            if "config_data" in st.session_state:
                config_df=pd.DataFrame([st.session_state.config_data])
                if (config_df['Modeling Type'].eq(1).any() and 
                    config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                    config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('no').any() and config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('no').any()) :

                    MSP_L0_T1_direct = MSP_L0.copy()
                    MSP_L0_T1_direct['CsfPeriod'] = Reference_Period
                    MSP_L0_T1_direct= MSP_L0



                    st.session_state["MSP_L0"]=MSP_L0


            # st.write('MSP L0 FILE:')
            # st.dataframe(MSP_L0)


            # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                





            # if "MSP_L0" in st.session_state and st.session_state["MSP_L0"] is not None:
            #     # Convert DataFrame to HTML with preserved index
            #     table_html = st.session_state["MSP_L0"].to_html(index=True)

            #     # Sticky and Scrollable Table
            #     st.markdown(
            #         """
            #         <style>
            #             .scrollable-table-container {
            #                 position: fixed;
            #                 top: 330px;
            #                 right: 1%;
            #                 width: 40%;
            #                 height: 400px;
            #                 overflow-x: auto !important;
            #                 overflow-y: auto !important;
            #                 z-index: 1000;
            #                 background: white;
            #                 border: 1px solid #ddd;
            #             }

            #             .scrollable-table-container table {
            #                 width: auto; /* Adjust table width dynamically */
            #                 border-collapse: collapse;
            #             }

            #             .scrollable-table-container th, 
            #             .scrollable-table-container td {
            #                 border: 1px solid #ddd;
            #                 padding: 8px;
            #                 text-align: left;
            #                 white-space: nowrap;
            #             }
            #         </style>
            #         """,
            #         unsafe_allow_html=True
            #     )

            #     # Display inside a scrollable div
            #     st.markdown(f'<div class="scrollable-table-container">{table_html}</div>', unsafe_allow_html=True)
            # else:
            #     st.warning("MSP_L0 data is not available. Please upload or initialize the data.")




        
            import plotly.graph_objects as go




            # options = ["GRAPHS"]
            # option = st.pills(f"SHOW GRAPHS!", options)

            # if option == "GRAPHS":


            # # Filter data based on selected categories, markets, and channels
            global_filtered_data = st.session_state["MSP_L0"].copy()

            # Apply filters if they exist
            # if st.session_state["selected_categories"]:
            #     global_filtered_data = global_filtered_data[global_filtered_data['Category'].isin(st.session_state["selected_categories"])]
            # if st.session_state["selected_markets"]:
            #     global_filtered_data = global_filtered_data[global_filtered_data['Market'].isin(st.session_state["selected_markets"])]
            if st.session_state["selected_channels"]:
                global_filtered_data = global_filtered_data[global_filtered_data['Channel'].isin(st.session_state["selected_channels"])]

            # Check if any data is left after filtering
            if global_filtered_data.empty:
                st.warning("No data to display after applying the filters.")
            else:

                df = global_filtered_data.copy()
                df["MSP"] = df["MSP"] * 100  
                df["MSP_label"] = df["MSP"].apply(lambda x: f"{x:.2f}%")

                df["MShare"] = df["MShare"] * 100  
                df["MShare_label"] = df["MShare"].apply(lambda x: f"{x:.2f}%")

                # Plotly chart for CSF
                fig_csff= px.bar(
                    df,
                    x='Brand',  # Ensure index is treated as string for display
                    y='CSF',
                    template="plotly_white",
                    color='Brand',
                    text_auto=True,  # Display y-axis values on top of bars
                    hover_data=["Channel"],  # Display y-axis values on top of bars
                )

                # fig_csff.update_traces(textposition="outside")  # Position labels outside bars

                # Customize hovertemplate for more detailed hover information
                fig_csff.update_traces(
                    hovertemplate=
                                'Channel: <b>%{customdata[0]}<br><br>'
                                '<b>%{x}</b><br>'  # Brand
                                # Channel
                                '<extra></extra>'  # Remove extra information like trace name
                )

                fig_csff.update_layout(
                    title="CSF",
                    xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                    yaxis=dict(title="CSF", color='black', showgrid=False, tickformat=".2f"),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='black',
                    legend=dict(
                        orientation="h",  # Horizontal orientation
                        xanchor="left",  # Anchor to the left
                        x=0,  # Position at the left edge
                        yanchor="bottom",  # Anchor to the bottom of the legend
                        y=1.02  # Position above the chart
                    )
                )

                st.plotly_chart(fig_csff, use_container_width=True)


                st.markdown('<hr class="thin">', unsafe_allow_html=True)

                



                # Plotly chart for MSP
                fig_msp = px.bar(
                    df,
                    x='Brand',  # Ensure index is treated as string for display
                    y='MSP',
                    template="plotly_white",
                    color='Brand',
                    text=df["MSP_label"],
                    hover_data=["Channel"],  # Display y-axis values on top of bars
                )

                fig_msp.update_traces(textposition="outside")  # Position labels outside bars

                # Customize hovertemplate for more detailed hover information
                fig_msp.update_traces(
                    hovertemplate=
                                'Channel: <b>%{customdata[0]}<br><br>'
                                '<b>%{x}</b><br>'  # Brand
                                # Channel
                                '<extra></extra>'  # Remove extra information like trace name
                )

                fig_msp.update_layout(
                    title="MSP",
                    xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                    yaxis=dict(title="MSP", color='black', showgrid=False, tickformat=".2f"),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='black',
                    legend=dict(
                        orientation="h",  # Horizontal orientation
                        xanchor="left",  # Anchor to the left
                        x=0,  # Position at the left edge
                        yanchor="bottom",  # Anchor to the bottom of the legend
                        y=1.02  # Position above the chart
                    )
                )

                st.plotly_chart(fig_msp, use_container_width=True)


                st.markdown('<hr class="thin">', unsafe_allow_html=True)



                # Plotly chart for MSP
                fig_ms = px.bar(
                    df,
                    x='Brand',  # Ensure index is treated as string for display
                    y='MShare',
                    template="plotly_white",
                    color='Brand',
                    text=df["MShare_label"],  # Display y-axis values on top of bars
                    hover_data=["Channel"],  # Display y-axis values on top of bars
                )

                # fig_csff.update_traces(textposition="outside")  # Position labels outside bars

                # # Customize hovertemplate for more detailed hover information
                fig_ms.update_traces(
                    hovertemplate=
                                'Channel: <b>%{customdata[0]}<br><br>'
                                '<b>%{x}</b><br>'  # Brand
                                # Channel
                                '<extra></extra>'  # Remove extra information like trace name
                )

                fig_ms.update_layout(
                    title="MSHARE",
                    xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                    yaxis=dict(title="MShare", color='black', showgrid=False, tickformat=".2f"),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='black',
                    legend=dict(
                        orientation="h",  # Horizontal orientation
                        xanchor="left",  # Anchor to the left
                        x=0,  # Position at the left edge
                        yanchor="bottom",  # Anchor to the bottom of the legend
                        y=1.02  # Position above the chart
                    )
                )

                st.plotly_chart(fig_ms, use_container_width=True)


                st.markdown('<hr class="thin">', unsafe_allow_html=True)




                # # Create figure
                # fig_mp = go.Figure()

                # # Add Price bars
                # fig_mp.add_trace(go.Bar(
                #     x=global_filtered_data['Brand'],
                #     y=global_filtered_data['Price'],
                #     name='Price',
                #     marker_color='blue'
                # ))

                # # Add MCV bars
                # fig_mp.add_trace(go.Bar(
                #     x=global_filtered_data['Brand'],
                #     y=global_filtered_data['MCV'],
                #     name='MCV',
                #     marker_color='orange'
                # ))

                # # Update layout
                # fig_mp.update_layout(
                #     barmode='group',  # Group bars next to each other
                #     title="Price & MCV",
                #     xaxis=dict(title="Brand", color='black', showgrid=False, showticklabels=True),
                #     yaxis=dict(title="Value", color='black', showgrid=False, tickformat=".2f"),
                #     plot_bgcolor='white',
                #     paper_bgcolor='white',
                #     font_color='black',
                #     legend=dict(
                #         orientation="h",
                #         xanchor="left",
                #         x=0,
                #         yanchor="bottom",
                #         y=1.02
                #     )
                # )

                # # Display in Streamlit
                # st.plotly_chart(fig_mp, use_container_width=True)\


                fig_mp = go.Figure()

                # Add Price bars with a color scale
                fig_mp.add_trace(go.Bar(
                    x=df['Brand'],
                    y=df['Price'],
                    name='Price',
                    text=global_filtered_data['Price'].round(2),  # Show values rounded to 2 decimal places
                    textposition='auto',
                    marker=dict(color=global_filtered_data['Brand'].astype('category').cat.codes,  # Convert Brand to category and map to a code
                                colorscale='Viridis')  # Apply a colorscale
                ))

                # Add MCV bars with a color scale
                fig_mp.add_trace(go.Bar(
                    x=df['Brand'],
                    y=df['MCV'],
                    text=df['MCV'].round(2),  # Show values rounded to 2 decimal places
                    textposition='auto',
                    name='MCV',
                    marker=dict(color=df['Brand'].astype('category').cat.codes,  # Convert Brand to category and map to a code
                                colorscale='Viridis')  # Apply a colorscale
                ))

                # Update layout
                fig_mp.update_layout(
                    barmode='group',  # Group bars next to each other
                    title="Price & MCV",
                    xaxis=dict(title="Brand", color='black', showgrid=False, showticklabels=True),
                    yaxis=dict(title="Value", color='black', showgrid=False, tickformat=".2f"),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='black',
                    legend=dict(
                        orientation="h",
                        xanchor="left",
                        x=0,
                        yanchor="bottom",
                        y=1.02
                    ),
                    showlegend=False
                )

                # Display in Streamlit
                st.plotly_chart(fig_mp, use_container_width=True)



                st.markdown('<hr class="thin">', unsafe_allow_html=True)




            options = ["MSPL0 FILE"]
            option = st.pills(f"SHOW MSP FILE!", options, default="MSPL0 FILE")

            if option == "MSPL0 FILE":
                st.write('MSPL0 FILE:')
                st.dataframe(st.session_state["MSP_L0"])


                st.download_button("Download the final modified data", 
                    data=st.session_state["MSP_L0"].to_csv(index=False), 
                    file_name="MSP_L0.csv", 
                    mime="csv")


            st.markdown('<hr class="thin">', unsafe_allow_html=True)




#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L2-------------------------------------------------------------------------------------------




# with tab1:
#     uploaded_file = st.sidebar.file_uploader("Upload L2: ", type="xlsx")

#     if uploaded_file:
#         try:
#             if "uploaded_file" not in st.session_state or uploaded_file != st.session_state["uploaded_file"]:
#                 # Clear session state L2 when a new file is uploaded
#                 if "L2" in st.session_state:
#                     del st.session_state["L2"]
                
#                 # Load new data
#                 L2 = load_data(uploaded_file)

#                 # Store the file name without the extension
#                 file_name = uploaded_file.name.split('.')[0]
                
#                 # Save the uploaded file reference and L2 to session state
#                 st.session_state["uploaded_file"] = uploaded_file
#                 st.session_state["L2"] = L2


#                 st.session_state["L2file_name"] = file_name

#             else:
#                 # If the file hasn't changed, load the L2 from session state
#                 L2 = st.session_state.get("L2")

#         except Exception as e:
#             # Catch and display any errors that occur
#             st.sidebar.error(f"An error occurred while processing the file: {e}")
#     else:
#         st.sidebar.info("Please upload an Excel file to proceed.")


if uploaded_file0:

    uploaded_file2 = st.sidebar.file_uploader("📄 Upload L2 file: ", type="xlsx")
    with st.sidebar.expander("🔧 Advanced Settings for L2", expanded=False):
        apply_method_filter2 = st.checkbox("Remove 'Lasso' and 'Ridge' methods", value=True, key="l2_method")
        apply_dist_elas_filter2 = st.checkbox("Apply Distribution_elas Filter (>= 0 or NaN)", value=True, key="l2_dist_elas")
        apply_cat_elas_filter2 = st.checkbox("Apply Category_elas Filter (>= 0 or NaN)", value=True, key="l2_cat_elas")
        apply_pval_filter2 = st.checkbox("Filter Price_pval == 'Yes'", value=True, key="l2_pval")

    st.sidebar.markdown('<hr class="thin">', unsafe_allow_html=True)



    if uploaded_file2:

        # tab2, =st.tabs(["MSP L0L2"])
        with tab2:

        # uploaded_file2 = st.sidebar.file_uploader("Upload L2 file: ", type="xlsx")

            if uploaded_file2:
                try:
                    # if "uploaded_file2" not in st.session_state or uploaded_file2 != st.session_state["uploaded_file2"]:
                    #     # Clear session state L2 when a new file is uploaded
                    #     if "L2" in st.session_state:
                    #         del st.session_state["L2"]
                        
                    #     # Load new data
                    #     L2 = load_data(uploaded_file2)

                        # # Store the file name without the extension
                        # file_name = uploaded_file2.name.split('.')[0]

                        # # Extract the ending after "Wtd_avg_MCV_" in the file name
                        # if "Wtd_avg_MCV_" in file_name:
                        #     L2_name = file_name.split("Wtd_avg_MCV_")[-1]
                        # else:
                        #     L2_name = None  # Set to None if "Wtd_avg_MCV_" is not in the filename

                        # # Save the uploaded file reference and L2 to session state
                        # st.session_state["uploaded_file2"] = uploaded_file2
                        # st.session_state["L2"] = L2
                        # st.session_state["L2file_name"] = file_name
                        # st.session_state["L2_name"] = L2_name  # Store extracted name


                    if ("uploaded_file2" not in st.session_state or uploaded_file2 != st.session_state["uploaded_file2"]) or \
                        (apply_method_filter2 != st.session_state.get("apply_method_filter2", True) or
                        apply_dist_elas_filter2 != st.session_state.get("apply_dist_elas_filter2", True) or
                        apply_cat_elas_filter2 != st.session_state.get("apply_cat_elas_filter2", True) or
                        apply_pval_filter2 != st.session_state.get("apply_pval_filter2", True)):
                        
                        # Clear session state data when a new file is uploaded
                        if "L2" in st.session_state:
                            del st.session_state["L2"]

                    
                        
                        # Load new data
                        L2 = load_data(uploaded_file2,apply_method_filter2, apply_dist_elas_filter2, apply_cat_elas_filter2, apply_pval_filter2)


                        # Save to session state
                        st.session_state["uploaded_file2"] = uploaded_file2
                        st.session_state["L2"] = L2
                        st.session_state["apply_method_filter2"] = apply_method_filter2
                        st.session_state["apply_dist_elas_filter2"] = apply_dist_elas_filter2
                        st.session_state["apply_cat_elas_filter2"] = apply_cat_elas_filter2
                        st.session_state["apply_pval_filter2"] = apply_pval_filter2



                        # Store the file name without the extension
                        file_name = uploaded_file2.name.split('.')[0]

                        # Extract the ending after "Wtd_avg_MCV_" in the file name
                        if "Wtd_avg_MCV_" in file_name:
                            L2_name = file_name.split("Wtd_avg_MCV_")[-1]
                        else:
                            L2_name = None  # Set to None if "Wtd_avg_MCV_" is not in the filename

                        # Save the uploaded file reference and L2 to session state
                        st.session_state["uploaded_file2"] = uploaded_file2
                        st.session_state["L2"] = L2
                        st.session_state["L2file_name"] = file_name
                        st.session_state["L2_name"] = L2_name  # Store extracted name


                    else:
                        # If the file hasn't changed, load the L2 from session state
                        L2 = st.session_state.get("L2")

                except Exception as e:
                    # Catch and display any errors that occur
                    st.sidebar.error(f"An error occurred while processing the file: {e}")
            else:
                st.sidebar.info("Please upload an Excel file to proceed.")








            col1, col2 = st.columns(2)

            with col1:
                
                # Streamlit app

                if "L2" not in st.session_state:
                    st.session_state["L2"] = None

                if "L2_name" not in st.session_state:
                    st.session_state["L2_name"] = None

                if "L2file_name" not in st.session_state:
                    st.session_state["L2file_name"] = None

                # if "selected_categories2" not in st.session_state:
                #     st.session_state["selected_categories2"] = []

                # if "selected_markets2" not in st.session_state:
                #     st.session_state["selected_markets2"] = []

                # if "selected_channels2" not in st.session_state:
                #     st.session_state["selected_channels2"] = []

                # if "apply_filter" not in st.session_state:
                #     st.session_state.apply_filter = False  # Flag to track button 



                # File uploader
                # uploaded_file = st.file_uploader("Upload L0: Wtd_avg_MCV_Brand Excel file", type="xlsx")



                # if uploaded_file2:
                    # Check if the uploaded file is different from the previous one

                file_name2 = st.session_state["L2file_name"]
                st.session_state[file_name2] = st.session_state["L2"].copy()
                
                # Ensure `st.session_state["L2"]` is not None before further processing
                if st.session_state["L2"] is not None:
                    global_filtered_data = st.session_state["L2"].copy()

                    # st.markdown('<hr class="thick">', unsafe_allow_html=True)  # Add thick line after global filters

                    # st.write("Global Filters")
                    # col3, col4, col5 = st.columns(3)

                    # Initialize stateful filter selections for global filters
                    # selected_categories2 = st.session_state.get("selected_categories2", [])
                    # selected_markets2 = st.session_state.get("selected_markets2", [])
                    selected_channels2 = st.session_state.get("selected_channels2", [])

                    # Filter L2 dynamically based on global filter selections
                    filtered_global_data = L2.copy()
                    # if selected_categories2:
                    #     filtered_global_data = filtered_global_data[filtered_global_data['Category'].isin(selected_categories2)]
                    # if selected_markets2:
                    #     filtered_global_data = filtered_global_data[filtered_global_data['Market'].isin(selected_markets2)]
                    if selected_channels2:
                        filtered_global_data = filtered_global_data[filtered_global_data['Channel'].isin(selected_channels2)]

                    # Update global filter options dynamically
                    # updated_categories2 = filtered_global_data['Category'].unique()
                    # updated_markets2 = filtered_global_data['Market'].unique()
                    updated_channels2 = filtered_global_data['Channel'].unique()
                    
                    
                    # # Render global filters
                    # with col3:
                    #     selected_categories = st.multiselect(
                    #         "Select Category",
                    #         options=updated_categories2,
                    #         default=selected_categories2 if selected_categories2 else updated_categories2[:1],
                    #         key="selected_categories2",
        
                    #     )

                    # with col4:
                    #     selected_markets = st.multiselect(
                    #         "Select Market",
                    #         options=updated_markets2,
                    #         default=selected_markets2 if selected_markets2 else updated_markets2[:1],
                    #         key="selected_markets2",
            
                    #     )

                    # with col5:
                    selected_channels = st.multiselect(
                        "Select Channel",
                        options=updated_channels2,
                        default=st.session_state.get("selected_channels2", []),
                        key="selected_channels2",
            
                    )

                    # if st.button('APPLY FILTERS L0L2'):
                    #     st.write("")


                    # # Apply filter button
                    # if st.button("Apply Filter"):
                    #     st.session_state["selected_categories"] = updated_categories
                    #     st.session_state["selected_markets"] = updated_markets
                    #     st.session_state["selected_channels"] = updated_channels



                    
                    st.markdown('<hr class="thick">', unsafe_allow_html=True)



                    if 'L2_name' in st.session_state and st.session_state['L2_name']:
                        L2_name_column = st.session_state['L2_name']



                        # Display each brand's L2 and filters
                        unique_L2_values  = filtered_global_data[L2_name_column].unique()

                        for unique_value  in unique_L2_values:
                            st.header(f"{L2_name_column} : {unique_value}")
                            L2_name_column_data = filtered_global_data[filtered_global_data[L2_name_column] == unique_value]

                            # Brand-specific filters beside graphs
                            col6, col7 = st.columns([3, 1])  # Make graphs column wider

                            with col7:
                                # Initialize filters dynamically
                                methods = st.multiselect(
                                    f"Select Method for {unique_value}",
                                    options=brand_data['method'].unique(),
                                    key=f"method_{unique_value}"
                                )

                                # Filter brand_data dynamically based on the selected method
                                filtered_data_by_method = L2_name_column_data.copy()
                                if methods:
                                    filtered_data_by_method = filtered_data_by_method[filtered_data_by_method['method'].isin(methods)]

                                
                                rpito_options = filtered_data_by_method['RPIto'].unique()
                                rpito_filter = st.multiselect(
                                    f"RPIto for {unique_value}",
                                    options=rpito_options,
                                    key=f"rpito_{unique_value}"
                                )

                                # Further filter based on selected RPIto
                                filtered_data_by_rpito = filtered_data_by_method.copy()
                                if rpito_filter:
                                    filtered_data_by_rpito = filtered_data_by_rpito[filtered_data_by_rpito['RPIto'].isin(rpito_filter)]



                                actualdistvar_options = filtered_data_by_rpito['actualdistvar'].unique()
                                actualdistvar_filter = st.multiselect(
                                    f"actualdistvar for {unique_value}",
                                    options=actualdistvar_options,
                                    key=f"actualdistvar_{unique_value}"
                                )

                                # Further filter based on selected RPIto
                                filtered_data_by_actualdistvar = filtered_data_by_rpito.copy()
                                if actualdistvar_filter:
                                    filtered_data_by_actualdistvar = filtered_data_by_actualdistvar[filtered_data_by_actualdistvar['actualdistvar'].isin(actualdistvar_filter)]
            

                                # Handle sliders with identical min and max
                                adjrsq_min = float(filtered_data_by_actualdistvar['Adj.Rsq'].min())
                                adjrsq_max = float(filtered_data_by_actualdistvar['Adj.Rsq'].max())
                                if adjrsq_min == adjrsq_max:
                                    adjrsq_min -= 0.01
                                    adjrsq_max += 0.01
                                adjrsquare_filter = st.slider(
                                    f"Adj Rsquare for {unique_value}",
                                    min_value=adjrsq_min,
                                    max_value=adjrsq_max,
                                    value=(adjrsq_min, adjrsq_max),
                                    key=f"adjrsq_{unique_value}"
                                )

                                # Further filter based on Adj.Rsq
                                filtered_data_by_adjrsq = filtered_data_by_actualdistvar.copy()
                                filtered_data_by_adjrsq = filtered_data_by_adjrsq[
                                    (filtered_data_by_adjrsq['Adj.Rsq'] >= adjrsquare_filter[0]) & 
                                    (filtered_data_by_adjrsq['Adj.Rsq'] <= adjrsquare_filter[1])
                                ]

                                aic_min = float(filtered_data_by_adjrsq['AIC'].min())
                                aic_max = float(filtered_data_by_adjrsq['AIC'].max())
                                if aic_min == aic_max:
                                    aic_min -= 0.01
                                    aic_max += 0.01
                                aic_filter = st.slider(
                                    f"AIC for {unique_value}",
                                    min_value=aic_min,
                                    max_value=aic_max,
                                    value=(aic_min, aic_max),
                                    key=f"aic_{unique_value}"
                                )

                                # Further filter based on AIC
                                filtered_data_by_aic = filtered_data_by_adjrsq.copy()
                                filtered_data_by_aic = filtered_data_by_aic[
                                    (filtered_data_by_aic['AIC'] >= aic_filter[0]) & 
                                    (filtered_data_by_aic['AIC'] <= aic_filter[1])
                                ]

                                csf_min = float(filtered_data_by_aic['CSF.CSF'].min())
                                csf_max = float(filtered_data_by_aic['CSF.CSF'].max())
                                if csf_min == csf_max:
                                    csf_min -= 0.01
                                    csf_max += 0.01
                                csf_filter = st.slider(
                                    f"CSF for {unique_value}",
                                    min_value=csf_min,
                                    max_value=csf_max,
                                    value=(csf_min, csf_max),
                                    key=f"csf_{unique_value}"
                                )

                                # Final filtered data based on CSF
                                filtered_data = filtered_data_by_aic.copy()
                                filtered_data = filtered_data[
                                    (filtered_data['CSF.CSF'] >= csf_filter[0]) & 
                                    (filtered_data['CSF.CSF'] <= csf_filter[1])
                                ]




                            # Ensure 'selectedmodels' column exists
                            filtered_data["selectedmodels"] = "No"
    
                            # Initialize session state if not present
                            if "saved_selections" not in st.session_state:
                                st.session_state["saved_selections"] = {}
    
                            if "saved_buttons" not in st.session_state:
                                st.session_state["saved_buttons"] = {}
    
                            if "reset_buttons" not in st.session_state:
                                st.session_state["reset_buttons"] = {}
    
    
                            # Function to update 'selectedmodels' column ensuring one "Yes" per (Channel, unique_value)
                            def update_selectedmodels(group, unique_value):
                                channel = group["Channel"].iloc[0]  # Get the Channel for this group
    
                                # Check if selection was saved for this (Channel, unique_value)
                                key = (channel, unique_value)
                                if key in st.session_state["saved_selections"]:
                                    saved_index = st.session_state["saved_selections"][key]
                                    if saved_index in group.index:
                                        group["selectedmodels"] = "No"
                                        group.loc[saved_index, "selectedmodels"] = "Yes"
                                        return group  # Return saved selection
    
                                # Otherwise, select the row closest to the median CSF.CSF
                                if not group.empty:
                                    median_csf = group["CSF.CSF"].median()
                                    closest_index = (group["CSF.CSF"] - median_csf).abs().idxmin()
                                    group["selectedmodels"] = "No"
                                    group.loc[closest_index, "selectedmodels"] = "Yes"
    
                                return group
    
                            # Apply selection logic per (Channel, unique_value)
                            filtered_data = filtered_data.groupby(["Channel", L2_name_column], group_keys=False).apply(update_selectedmodels, unique_value=unique_value)
    
                            # UI for selecting models
                            for (channel, unique_value), group in filtered_data.groupby(["Channel", L2_name_column]):
                                available_indices = group.index.tolist()
                                
                                default_index = (
                                    group[group["selectedmodels"] == "Yes"].index[0]
                                    if "Yes" in group["selectedmodels"].values
                                    else available_indices[0]
                                )
    
                                selected_index = st.selectbox(
                                    f"Select Model Index for {channel} - {unique_value}",
                                    options=available_indices,
                                    index=available_indices.index(default_index),
                                    key=f"selectbox_{channel}_{unique_value}"
                                )
    
                                # Ensure selectedmodels is locked once saved
                                if selected_index in group.index:
                                    key = (channel, unique_value)
                                    if key in st.session_state["saved_selections"]:
                                        saved_index = st.session_state["saved_selections"][key]
                                        group["selectedmodels"] = "No"
                                        group.loc[saved_index, "selectedmodels"] = "Yes"
                                    else:
                                        group["selectedmodels"] = "No"
                                        group.loc[selected_index, "selectedmodels"] = "Yes"
    
                                col3,col4=st.columns([3,2])
    
                                # Dynamically set button text based on session state
                                save_button_text = st.session_state["saved_buttons"].get((channel, unique_value), f"SAVE Selection {channel} - {unique_value}")
                                reset_button_text = st.session_state["reset_buttons"].get((channel, unique_value), f"RESET Selection {channel} - {unique_value}")
    
                                with col3:
                                    # Button to save selection
                                    if st.button(save_button_text, key=f"save_{channel}_{unique_value}"):
                                        if (channel, unique_value) in st.session_state["saved_selections"]:
                                            saved_index = st.session_state["saved_selections"][(channel, unique_value)]
                                            if saved_index != selected_index:
                                                st.error(f"Selection already saved for {channel} - {unique_value} at index {saved_index}.\n\n Please 'RESET' first before changing.")
                                            else:
                                                st.success(f"Selection already saved for {channel} - {unique_value} at index {saved_index}.")
                                        else:
                                            st.session_state["saved_selections"][(channel, unique_value)] = selected_index
                                            st.session_state["saved_buttons"][(channel, unique_value)] = f"SAVED ✅ ({channel} - {unique_value})"
                                            st.session_state["reset_buttons"][(channel, unique_value)] = f"RESET Selection {channel} - {unique_value}"
                                            st.success(f"Selection saved for {channel} - {unique_value} at index {selected_index}.")
    
    
                                with col4:
                                    # Button to reset selection
                                    if st.button(reset_button_text, key=f"reset_{channel}_{unique_value}"):
                                        if (channel, unique_value) in st.session_state["saved_selections"]:
                                            del st.session_state["saved_selections"][(channel, unique_value)]
                                        st.session_state["reset_buttons"][(channel, unique_value)] = f"RESET 🔄 ({channel} - {unique_value})"
                                        st.session_state["saved_buttons"][(channel, unique_value)] = f"SAVE Selection {channel} - {unique_value}"
                                        st.success(f"Selection reset for {channel} - {unique_value}.\n\n Now updates dynamically.")
    
                                # Store updates in session state
                                st.session_state["filtered_data"] = filtered_data
    
                                # Ensure L2 is updated properly with "Yes" for each (Channel, unique_value)
                                if "L2" in st.session_state:
                                    L2 = st.session_state["L2"]
                                    
                                    # Reset all selectedmodels to "No" for this Channel-Brand pair in L2
                                    L2.loc[(L2["Channel"] == channel) & (L2[L2_name_column] == unique_value), "selectedmodels"] = "No"
    
                                    # Apply the saved selection or median-based selection
                                    selected_row = group[group["selectedmodels"] == "Yes"]
                                    if not selected_row.empty:
                                        selected_idx = selected_row.index[0]
                                        L2.loc[selected_idx, "selectedmodels"] = "Yes"
    
                                    st.session_state["L2"] = L2.copy()
    
                                    # Save updates to the session state file
                                    file_name2 = st.session_state.get("L2file_name", "L2")
                                    st.session_state[file_name2] = L2.copy()






















                            with col6:
                                # graph_option = st.radio(f"Choose graph type for {brand}", options=["MCV", "CSF"], key=f"graph_option_{brand}")
                                options=["CSF","MCV"]
                                graph_option = st.pills(f"Choose graph type for {unique_value}",options,selection_mode="single",default="CSF",key=f"graph_option_{unique_value}")

                                if graph_option == "MCV":

                                    sort_by = st.radio(
                                        "SORT BY:", ["MCV.MCV", "Adj.Rsq", "AIC"], index=0, horizontal=True, key=f"sort_option_mcv_{unique_value}"
                                    )
                                    # Sort data by 'MCV.MCV' in ascending order
                                    sorted_data = filtered_data.sort_values(by=sort_by, ascending=True)
                                    sorted_data['Index'] = sorted_data.index.astype(str)
                                                        
                                    # Plotly chart for MCV with a better color palette
                                    fig_mcv = px.bar(
                                        sorted_data,
                                        x='Index',  # Use the 'Index' column for x-axis
                                        y='MCV.MCV',
                                        template="plotly_white",
                                        color='RPIto',  # Color bars based on the 'RPIto' column
                                        text_auto=True,  # Display y-axis values on top of bars
                                        category_orders={"Index": sorted_data['Index'].tolist()},  # Ensure bars follow the sorted order
                                        color_discrete_sequence=px.colors.qualitative.Set3,  # Use a better color palette
                                        hover_data={
                                            "RPIto":False,
                                            "Index": True, 
                                            "Adj.Rsq": ':.2f', 
                                            "AIC": ':.2f'
                                        }
                                    )

                                    # Update layout to remove title and position legend at top-left
                                    fig_mcv.update_layout(
                                        title="",  # Set title to empty string to avoid 'undefined'
                                        xaxis=dict(title="Index", color='black', showgrid=False, showticklabels=False),
                                        yaxis=dict(title="MCV", color='black', showgrid=False, tickformat=".1f"),
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        font_color='black',
                                        legend=dict(
                                            orientation="h",  # Horizontal orientation
                                            xanchor="left",  # Anchor to the left
                                            x=0,  # Position at the left edge
                                            yanchor="bottom",  # Anchor to the bottom of the legend
                                            y=1.02  # Position above the chart
                                        )
                                    )

                                    # Display the chart
                                    st.plotly_chart(fig_mcv, use_container_width=True)


                                elif graph_option == "CSF":

                                    sort_by = st.radio(
                                            "SORT BY:", ["CSF.CSF", "Adj.Rsq", "AIC"], index=0, horizontal=True, key=f"sort_option_csf_{unique_value}"
                                        )
                                    # Sort data by 'CSF.CSF' in ascending order
                                    sorted_data = filtered_data.sort_values(by=sort_by, ascending=True)
                                    sorted_data['Index'] = sorted_data.index.astype(str)
                                    
                                    # Plotly chart for CSF
                                    fig_csf = px.bar(
                                        sorted_data,
                                        x='Index',  # Ensure index is treated as string for display
                                        y='CSF.CSF',
                                        template="plotly_white",
                                        # color_discrete_sequence=["#FFD700"],
                                        color='RPIto',
                                        text_auto=True,  # Display y-axis values on top of bars
                                        category_orders={"Index": sorted_data['Index'].tolist()},
                                        color_discrete_sequence=px.colors.qualitative.Set3,  # Use a better color palette
                                        hover_data={
                                            "RPIto":False,
                                            "Index": True, 
                                            "Adj.Rsq": ':.2f', 
                                            "AIC": ':.2f'
                                        }


                                    )
                                    fig_csf.update_layout(
                                        title="",
                                        xaxis=dict(title="Index", color='black', showgrid=False,showticklabels=False),
                                        yaxis=dict(title="CSF", color='black', showgrid=False,tickformat=".2f"),
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        font_color='black',
                                        legend=dict(
                                            orientation="h",  # Horizontal orientation
                                            xanchor="left",  # Anchor to the left
                                            x=0,  # Position at the left edge
                                            yanchor="bottom",  # Anchor to the bottom of the legend
                                            y=1.02  # Position above the chart
                                        )
                                    )
                                    st.plotly_chart(fig_csf, use_container_width=True)

                            # options=["MAKE SELECTIONS","SUBMIT SELECTIONS"]
                            # submit_option = st.pills(f"",options,selection_mode="single",default="SUBMIT SELECTIONS",key=f"show_data_{unique_value}")
                            # # Checkbox for brand-specific data
                            # # if st.checkbox(f"Show Data for {unique_value}", key=f"show_data_{unique_value}"):
                            # if submit_option=="MAKE SELECTIONS":

                            with st.expander("SHOW DATA"):
                            
                                # st.error("CLICK ON 'SUBMIT SELECTIONS' TO PERMANENTLY SAVE THE CHANGES.")
                                # Apply global filters to the full dataset
                                global_filtered_data = st.session_state["L2"].copy()
                                # if selected_categories2:
                                #     global_filtered_data = global_filtered_data[global_filtered_data['Category'].isin(selected_categories2)]
                                # if selected_markets2:
                                #     global_filtered_data = global_filtered_data[global_filtered_data['Market'].isin(selected_markets2)]
                                if selected_channels2:
                                    global_filtered_data = global_filtered_data[global_filtered_data['Channel'].isin(selected_channels2)]

                                # Extract the brand-specific data from the globally filtered data
                                L2_name_column_data = global_filtered_data.loc[global_filtered_data[L2_name_column] == unique_value].copy()

                                # Apply brand-specific filters
                                filtered_data = L2_name_column_data.copy()
                                if methods:
                                    filtered_data = filtered_data[filtered_data['method'].isin(methods)]

                            
            

                                if rpito_filter:
                                    filtered_data = filtered_data[filtered_data['RPIto'].isin(rpito_filter)]


                                
                                if actualdistvar_filter:
                                    filtered_data = filtered_data[filtered_data['actualdistvar'].isin(actualdistvar_filter)]


                                filtered_data = filtered_data[
                                    (filtered_data['Adj.Rsq'] >= adjrsquare_filter[0]) & 
                                    (filtered_data['Adj.Rsq'] <= adjrsquare_filter[1])
                                ]
                                filtered_data = filtered_data[
                                    (filtered_data['CSF.CSF'] >= csf_filter[0]) & 
                                    (filtered_data['CSF.CSF'] <= csf_filter[1])
                                ]
                                filtered_data = filtered_data[
                                    (filtered_data['AIC'] >= aic_filter[0]) & 
                                    (filtered_data['AIC'] <= aic_filter[1])
                                ]




                                # st.session_state["filtered_data"]=filtered_data



                                #     # Ensure the 'selectedmodels' column exists
                                # filtered_data['selectedmodels'] = 'No'

                                # # Function to update selectedmodels for each (Channel, Brand) group
                                # def update_selectedmodels(group):
                                #     if not group.empty:
                                #         median_csf = group['CSF.CSF'].median()  # Find the median
                                #         closest_index = (group['CSF.CSF'] - median_csf).abs().idxmin()  # Get index closest to median
                                #         group.loc[:, 'selectedmodels'] = 'No'  # Reset all to 'No'
                                #         group.loc[closest_index, 'selectedmodels'] = 'Yes'  # Set the closest row as 'Yes'
                                #     return group

                                # # Apply function to ensure only one 'Yes' per (Channel, Brand) combination
                                # filtered_data = filtered_data.groupby(['Channel',L2_name_column], group_keys=False).apply(update_selectedmodels)

                                # # Store the updated filtered_data in session state
                                # st.session_state["filtered_data"] = filtered_data

                                # # Update only the 'selectedmodels' column in L2 for rows marked as 'Yes' in filtered_data
                                # if "L2" in st.session_state:
                                #     L2 = st.session_state["L2"]

                                #     for (channel, brand), group in filtered_data.groupby(['Channel', L2_name_column]):
                                #         # Reset all 'selectedmodels' to 'No' for this (Channel, L2_name_column) in L2
                                #         L2.loc[(L2['Channel'] == channel) & (L2[L2_name_column] == unique_value), 'selectedmodels'] = 'No'

                                #         # Find the row in filtered_data where 'selectedmodels' is 'Yes'
                                #         selected_row = group[group['selectedmodels'] == 'Yes']

                                #         if not selected_row.empty:
                                #             closest_index = selected_row.index[0]  # Get the first matching index

                                #             # Update only the 'selectedmodels' column in L2
                                #             L2.loc[closest_index, 'selectedmodels'] = 'Yes'

                                # # Store the updated L2 in session state **before saving under file_name**
                                #     st.session_state["L2"] = L2.copy()

                                #     # Store updated data in session state with file name
                                #     file_name2 = st.session_state["L2file_name"]
                                #     st.session_state[file_name2] = L2.copy()







                                # # Incorporate previously saved changes, if any
                                # if f"filtered_data_{unique_value}" in st.session_state:
                                #     # Update the filtered L2 with previously saved changes
                                #     modified_rows = st.session_state[f"filtered_data_{unique_value}"]
                                #     filtered_data.update(modified_rows)

                                # # Reset 'selectedmodels' column if not present
                                # if "selectedmodels" not in filtered_data.columns:
                                #     filtered_data['selectedmodels'] = 'No'

                                # # Calculate the default selection based on the median of CSF.CSF
                                # if not filtered_data.empty:
                                #     median_csf_index = (
                                #         (filtered_data['CSF.CSF'] - filtered_data['CSF.CSF'].median())
                                #         .abs()
                                #         .idxmin()
                                #     )
                                # else:
                                #     median_csf_index = None

                                # # Display the table with a selectbox to choose a row
                                # st.write("Select a row to choose as the selected method:")
                                # selected_index = st.selectbox(
                                #     f"Select index for {unique_value}",
                                #     options=filtered_data.index,
                                #     index=filtered_data.index.get_loc(median_csf_index) if median_csf_index in filtered_data.index else 0,
                                #     format_func=lambda x: f"{x}"  # Optional: Add more clarity to selection
                                # )

                                # # Track the state for each brand selection
                                # if unique_value not in st.session_state:
                                #     st.session_state[unique_value] = {'previous_states': [], 'last_selected_index': None}  # Initialize stack and last index

                                # # If a new 'Yes' is selected, save the current state before the change
                                # if selected_index != st.session_state[unique_value].get('last_selected_index', None):
                                #     # Save the current L2 before applying the new 'Yes'
                                #     st.session_state[unique_value]['previous_states'].append(st.session_state["L2"].copy())
                                #     st.session_state[unique_value]['last_selected_index'] = selected_index  # Update the last selected index

                                # # Now handle the "Revert" button
                                # # Create two columns to place the buttons side by side
                                # col8, col9 = st.columns([5, 1])  # Adjust the column widths as needed

                                # # First button: Save Selected Method
                                # with col8:
                                #     if st.button(f"Save Selected Method for {unique_value}", key=f"save_selected_method_{unique_value}"):
                                #         # Check if there are multiple 'Yes' for the current brand under the global filters
                                #         global_brand_filter = (
                                #             (st.session_state["L2"][L2_name_column] == unique_value) &
                                #             # (st.session_state["L2"]['Market'].isin(selected_markets)) &
                                #             # (st.session_state["L2"]['Category'].isin(selected_categories)) &
                                #             (st.session_state["L2"]['Channel'].isin(selected_channels2))
                                #         )

                                #         # Reset if there is already a 'Yes' for the brand under the filters
                                #         if (st.session_state["L2"].loc[global_brand_filter, 'selectedmodels'] == 'Yes').sum() > 0:
                                #             # Reset all rows for the current brand and filters to 'No'
                                #             st.session_state["L2"].loc[global_brand_filter, 'selectedmodels'] = 'No'

                                #         # Set the selected row's 'selectedmodels' column to 'Yes'
                                #         st.session_state["L2"].loc[selected_index, 'selectedmodels'] = 'Yes'

                                #         # Update the filtered L2 for the current brand
                                #         filtered_data['selectedmodels'] = st.session_state["L2"].loc[
                                #             filtered_data.index, 'selectedmodels'
                                #         ]

                                #         st.session_state["L2"].loc[filtered_data.index, 'selectedmodels'] = filtered_data['selectedmodels']
                                #         file_name2 = st.session_state["L2file_name"]
                                #         st.session_state[file_name2] = st.session_state["L2"].copy() 

                                #         st.success(f"Selected method for {brand} saved successfully!")

                                # # Second button: Revert Selection
                                # with col9:
                                #     if st.button(f"REVERT", key=f"revert_selected_method_{unique_value}"):
                                #         if st.session_state[unique_value]['previous_states']:
                                #             # Pop the most recent state to revert to the previous selection
                                #             last_state = st.session_state[unique_value]['previous_states'].pop()
                                #             st.session_state["L2"] = last_state.copy()  # Revert to the last state

                                #             # Update the filtered L2 for the current brand with reverted selections
                                #             filtered_data['selectedmodels'] = st.session_state["L2"].loc[
                                #                 filtered_data.index, 'selectedmodels'
                                #             ]

                                #             st.session_state["L2"].loc[filtered_data.index, 'selectedmodels'] = filtered_data['selectedmodels']
                                #             file_name2 = st.session_state["L2file_name"]
                                #             st.session_state[file_name2] = st.session_state["L2"].copy()

                                #             st.success(f"Reverted!")
                                #         else:
                                #             st.warning("No previous state found to revert to!")
                                
                                # # filtered_data_reset = filtered_data.reset_index()

                                # Display the filtered data for this brand, with the updated 'selectedmodels' after save/revert
                                st.dataframe(filtered_data[['Market', 'Category', 'Channel', 'MCV.MCV', 'CSF.CSF', 'method', 'selectedmodels', 'RPIto', 'Adj.Rsq', 'AIC','actualdistvar']])

                            # elif submit_option=="SUBMIT SELECTIONS":
                            #     st.success("TO SELECT MODELS, PLEASE CLICK ON 'MAKE SELECTIONS' BUTTON.")

                            st.markdown('<hr class="thin">', unsafe_allow_html=True)  # Add thin line after each brand

                        # Show the final modified data after changes
                        if st.button("Show Final Modified Data FOR L2"):
                            # # Combine the original data with any updated rows
                            # final_modified_data = st.session_state["data"].copy()
                            
                            file_name2 = st.session_state["L2file_name"]
                            original_columns = st.session_state["L2"].columns.tolist()
                            st.session_state[file_name2].columns = original_columns

                            # st.session_state[file_name] = final_modified_data

                            download_file_name2 = f"{file_name2}.csv" if not file_name2.endswith(".csv") else file_name2

                            # st.write(f"Final Modified Data (File Name: {file_name}):")
                            st.dataframe(st.session_state[file_name2])

                            st.download_button("Download the final modified data", 
                                    data=st.session_state[file_name2].to_csv(index=False), 
                                    file_name=download_file_name2, 
                                    mime="text/csv")

                        st.markdown('<hr class="thin">', unsafe_allow_html=True)






        with col2:


            if uploaded_file2:

            # st.markdown(
            #     """
            #     <style>
            #         /* Ensure AgGrid stays fixed in the viewport */
            #         .streamlit-expanderHeader {
            #             position: sticky;
            #             top: 330px;
            #             z-index: 1000;
            #         }
            #         /* More specific targeting for AgGrid to stay fixed */
            #         div[data-testid="stAgGrid"] {
            #             position: fixed;
            #             top: 330px;
            #             right: 1%;
            #             width: 40%;
            #             z-index: 1000;
            #         }
            #     </style>
            #     """,
            #     unsafe_allow_html=True
            # )


            # Custom CSS to make only the specific DataFrame sticky
            # Apply CSS to make ONLY this specific DataFrame sticky
            # Apply CSS to make ONLY this specific DataFrame sticky
            # Apply CSS to make ONLY this specific DataFrame sticky
            # st.markdown(
            #     """
            #     <style>
            #         /* Target only the DataFrame inside the sticky-container */
            #         [data-testid="stVerticalBlock"] > div[data-testid="column"] > div#sticky-container {
            #             position: fixed;
            #             top: 100px;  /* Adjust based on navbar */
            #             right: 20px;
            #             width: 30%;
            #             z-index: 1000;
            #             background: white;
            #             box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            #         }
            #     </style>
            #     """,
            #     unsafe_allow_html=True
            # )
                # if "MSP_L0" not in st.session_state:
                #         st.session_state["MSP_L0"] = None

                if "MSP_L0L2" not in st.session_state:
                    st.session_state["MSP_L0L2"] = None


            
                # # st.markdown('<hr class="thick">', unsafe_allow_html=True)

                # options = ["CONFIGURATION"]
                # option = st.pills(f"", options,selection_mode="single",default="CONFIGURATION",)

                # if option == "CONFIGURATION":
                #     # Define available options
                #     modeling_type_options = [1, 2]
                #     msp_options = ["Yes", "No"]
                #     variable_options = ['NA', 'Market', 'Channel', 'Region', 'Category',
                #                         'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                #     periodicity_options = ["Daily", "Weekly", "Monthly"]

                #     # Function to get a valid index
                #     def get_valid_index(option_list, stored_value, default_value):
                #         return option_list.index(stored_value) if stored_value in option_list else option_list.index(default_value)

                #     # Initialize session state if not already set
                #     if "config_data" not in st.session_state:
                #         st.session_state.config_data = {
                #             "Modeling Type": 2,
                #             "MSP at L0 (Yes/No)": "Yes",
                #             "Variable to use as L0": "Brand",
                #             "MSP at L2 (Yes/No)": "No",
                #             "Variable to use as L2": "NA",
                #             "MSP at L3 (Yes/No)": "No",
                #             "Variable to use as L3": "NA",
                #             "Periodicity of Data (Daily/Weekly/Monthly)": "Weekly",
                #             "Reference Period": 1
                #         }

                #     # Get valid indices for selectbox
                #     modeling_type_index = get_valid_index(modeling_type_options, st.session_state.config_data["Modeling Type"], 2)
                #     msp_l0_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L0 (Yes/No)"], "Yes")
                #     msp_l2_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L2 (Yes/No)"], "No")
                #     msp_l3_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L3 (Yes/No)"], "No")
                #     variable_l0_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L0"], "Brand")
                #     variable_l2_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L2"], "NA")
                #     variable_l3_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L3"], "NA")
                #     periodicity_index = get_valid_index(periodicity_options, st.session_state.config_data["Periodicity of Data (Daily/Weekly/Monthly)"], "Daily")

                #     # Create a form for user input
                #     with st.form("config_form"):
                #         modeling_type = st.selectbox("Modeling Type", options=modeling_type_options, index=modeling_type_index)

                #         col10, col11 = st.columns(2)
                #         with col10:
                #             msp_l0 = st.selectbox("MSP at L0 (Yes/No)", options=msp_options, index=msp_l0_index)
                #             msp_l2 = st.selectbox("MSP at L2 (Yes/No)", options=msp_options, index=msp_l2_index)
                #             msp_l3 = st.selectbox("MSP at L3 (Yes/No)", options=msp_options, index=msp_l3_index)
                #             periodicity = st.selectbox("Periodicity of Data (Daily/Weekly/Monthly)", options=periodicity_options, index=periodicity_index)
                        
                #         with col11:
                #             variable_l0 = st.selectbox("Variable to use as L0", options=variable_options, index=variable_l0_index)
                #             variable_l2 = st.selectbox("Variable to use as L2", options=variable_options, index=variable_l2_index)
                #             variable_l3 = st.selectbox("Variable to use as L3", options=variable_options, index=variable_l3_index)
                #             reference_period = st.number_input("Reference Period", min_value=1, value=st.session_state.config_data["Reference Period"])

                #         # Submit button
                #         submit_button = st.form_submit_button("Save Configuration")

                #         # If submitted, update session state
                #         if submit_button:
                #             st.session_state.config_data = {
                #                 "Modeling Type": modeling_type,
                #                 "MSP at L0 (Yes/No)": msp_l0,
                #                 "Variable to use as L0": variable_l0,
                #                 "MSP at L2 (Yes/No)": msp_l2,
                #                 "Variable to use as L2": variable_l2,
                #                 "MSP at L3 (Yes/No)": msp_l3,
                #                 "Variable to use as L3": variable_l3,
                #                 "Periodicity of Data (Daily/Weekly/Monthly)": periodicity,
                #                 "Reference Period": reference_period
                #             }
                #             st.write("Configuration saved successfully!")

                #     # Display current configuration
                #     config_df = pd.DataFrame([st.session_state.config_data])
                #     st.write("Current Configuration:")
                #     st.dataframe(config_df)
                        
                # st.markdown('<hr class="thick">', unsafe_allow_html=True)

                

                # Fetch D0 file from session state
                if "D0" in st.session_state["uploaded_files"]:
                    D0 = st.session_state["uploaded_files"]["D0"]

                #     # # Rename date column if needed
                #     # if 'date' in D0.columns:
                #     #     D0.rename(columns={'date': 'Date'}, inplace=True)

                #     # Filter out unwanted brands
                #     if 'Brand' in D0.columns:
                #         D0 = D0[~D0['Brand'].str.lower().isin(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])]

                # else:
                #     st.sidebar.warning("D0 file not found in session state.")

                # Fetch Weighted Average Files from session state
                weighted_files = [
                    "Wtd_avg_MCV_Brand", "Wtd_avg_MCV_PackType", "Wtd_avg_MCV_PackSize",
                    "Wtd_avg_MCV_PPG", "Wtd_avg_MCV_Variant", "Wtd_avg_MCV_Category",
                    "Wtd_avg_MCV_SubCategory"
                ]

                df_dict = {
                    "Brand": pd.DataFrame(), "PackType": pd.DataFrame(), "PackSize": pd.DataFrame(),
                    "PPG": pd.DataFrame(), "Variant": pd.DataFrame(), "Category": pd.DataFrame(),
                    "SubCategory": pd.DataFrame()
                }

                for file_name2 in weighted_files:
                    if file_name2 in st.session_state:
                        try:
                            df = st.session_state[file_name2]  # ✅ Directly use stored DataFrame
                            
                            # # Print columns for debugging
                            # st.write(f"Columns in {file_name2}:", df.columns.tolist())


                            # Check if 'selectedmodels' exists (lowercase)
                            if 'selectedmodels' in df.columns:
                                df = df[df['selectedmodels'].str.lower() == 'yes']
                                key = file_name2.replace("Wtd_avg_MCV_", "").replace(".xlsx", "")
                                df_dict[key] = df  # ✅ Store filtered DataFrame
                                # st.write(df)
                                # if df_L0 is not None:
                                #     st.session_state.df_L0 = df_L0


                            else:
                                st.sidebar.warning(f"'{file_name2}' does not contain 'selectedmodels' column after processing.")

                        except Exception as e:
                            st.sidebar.error(f"Error processing {file_name2}: {e}")
                    # else:
                    #     st.sidebar.warning(f"{file_name2} not found in session state.")

                # st.markdown('<hr class="thin">', unsafe_allow_html=True)


            #-----------------------------------------------------------------------------------------------------------------------------------------
                # # Display processed datasets
                # st.subheader("Processed D0 Data")
                # if "D0" in st.session_state["uploaded_files"]:
                #     st.dataframe(D0)
                # else:
                #     st.write("No D0 data available.")

                # st.subheader("Processed Weighted Average Files")
                # for key, df in df_dict.items():
                #     st.write(f"### {key}")
                #     if not df.empty:
                #         st.dataframe(df)
                #     else:
                #         st.write("No data available.")


            # User Input for Time Period (Weekly data)

            #     if "config_data" in st.session_state:
            #         config_df=pd.DataFrame([st.session_state.config_data])

            #         # Check the conditions
            #         if (config_df['Periodicity of Data (Daily/Weekly/Monthly)']
            #             .astype(str)  # Ensure it's a string Series
            #             .str.strip()  # Remove extra spaces
            #             .str.lower()  # Convert to lowercase
            #             .eq('weekly') # Check for 'weekly'
            #             .any()):

            #             from datetime import timedelta
            #             import pandas as pd

            #             def defined_period_data(dataframe, weeks):
            #                 """
            #                 Filters the DataFrame to include rows from the most recent weeks based on the Date column.
            #                 Converts all dates to short format (YYYY-MM-DD) for consistency.
            #                 Assumes the data is strictly weekly.

            #                 Parameters:
            #                 - dataframe (pd.DataFrame): The input DataFrame with a date column.
            #                 - weeks (int): Number of weeks of data to retain.

            #                 Returns:
            #                 - pd.DataFrame: Filtered DataFrame with data from the specified period.
            #                 """
            #                 # Detect the date column (case-insensitive search for "date")
            #                 date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
            #                 if not date_column:
            #                     raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                            
            #                 # Function to handle multiple date formats and remove time if present
            #                 def parse_date(date):
            #                     for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
            #                                 '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):  # Supported formats
            #                         try:
            #                             # Convert to datetime
            #                             parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
            #                             if pd.notnull(parsed_date):
            #                                 return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
            #                         except ValueError:
            #                             continue
            #                     raise ValueError(f"Date '{date}' does not match any supported formats.")
                            
            #                 # Apply parsing and conversion to short format to the date column
            #                 dataframe[date_column] = dataframe[date_column].apply(parse_date)
                            
            #                 # Convert the date column back to datetime for further processing
            #                 dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                            
            #                 # Sort the DataFrame by date
            #                 dataframe = dataframe.sort_values(by=date_column)
                            
            #                 # Calculate cutoff date based on weeks
            #                 recent_date = dataframe[date_column].max()
            #                 cutoff_date = recent_date - timedelta(weeks=int(weeks))
                            
            #                 # Filter the DataFrame
            #                 filtered_df = dataframe[dataframe[date_column] > cutoff_date]
                            
            #                 return filtered_df


            #             # Example Usage:
            #             Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

            #         else:
            #             print("Conditions not met. Skipping execution.")




            # #--------------------------------------------------------

            # # User Input for Time Period (Monthly data)

            #     if "config_data" in st.session_state:
            #         config_df=pd.DataFrame([st.session_state.config_data])
            #         # Check the conditions
            #         if (config_df['Periodicity of Data (Daily/Weekly/Monthly)'].str.strip().str.lower().eq('monthly').any()):

            #             from datetime import timedelta
            #             import pandas as pd

            #             def defined_period_data(dataframe, months):
            #                 """
            #                 Filters the DataFrame to include rows from the most recent months based on the Date column.
            #                 Converts all dates to short format (YYYY-MM-DD) for consistency.
            #                 Assumes the data is strictly monthly.

            #                 Parameters:
            #                 - dataframe (pd.DataFrame): The input DataFrame with a date column.
            #                 - months (int): Number of months of data to retain.

            #                 Returns:
            #                 - pd.DataFrame: Filtered DataFrame with data from the specified period.
            #                 """
            #                 # Detect the date column (case-insensitive search for "date")
            #                 date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
            #                 if not date_column:
            #                     raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                            
            #                 # Function to handle multiple date formats and remove time if present
            #                 def parse_date(date):
            #                     for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
            #                                 '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):  # Supported formats
            #                         try:
            #                             # Convert to datetime
            #                             parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
            #                             if pd.notnull(parsed_date):
            #                                 return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
            #                         except ValueError:
            #                             continue
            #                     raise ValueError(f"Date '{date}' does not match any supported formats.")
                            
            #                 # Apply parsing and conversion to short format to the date column
            #                 dataframe[date_column] = dataframe[date_column].apply(parse_date)
                            
            #                 # Convert the date column back to datetime for further processing
            #                 dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                            
            #                 # Sort the DataFrame by date
            #                 dataframe = dataframe.sort_values(by=date_column)
                            
            #                 # Convert the date column to periods (monthly)
            #                 dataframe['Month_Sorting'] = dataframe[date_column].dt.to_period('M')
                            
            #                 # Calculate the cutoff month
            #                 recent_month = dataframe['Month_Sorting'].max()
            #                 cutoff_month = recent_month - months
                            
            #                 # Filter the DataFrame based on the cutoff month
            #                 filtered_df = dataframe[dataframe['Month_Sorting'] > cutoff_month]
                            
            #                 # Drop the temporary 'Month' column
            #                 filtered_df = filtered_df.drop(columns=['Month_Sorting'], errors='ignore')
                            
            #                 return filtered_df


            #             # Example Usage:
            #             Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

            #         else:
            #             print("Conditions not met. Skipping execution.")

            # #---------------------------------------------------------------------------

            # # User Input for Time Period (Daily data)

            #     if "config_data" in st.session_state:
            #         config_df=pd.DataFrame([st.session_state.config_data])

            #         # Check the conditions
            #         if (config_df['Periodicity of Data (Daily/Weekly/Monthly)'].str.strip().str.lower().eq('daily').any()):

            #             from datetime import timedelta
            #             import pandas as pd

            #             def defined_period_data(dataframe, days):
            #                 """
            #                 Filters the DataFrame to include rows from the most recent days based on the Date column.
            #                 Converts all dates to short format (YYYY-MM-DD) for consistency.
            #                 Assumes the data is strictly daily.

            #                 Parameters:
            #                 - dataframe (pd.DataFrame): The input DataFrame with a date column.
            #                 - days (int): Number of days of data to retain.

            #                 Returns:
            #                 - pd.DataFrame: Filtered DataFrame with data from the specified period.
            #                 """
            #                 # Detect the date column (case-insensitive search for "date")
            #                 date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
            #                 if not date_column:
            #                     raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                            
            #                 # Function to handle multiple date formats and remove time if present
            #                 def parse_date(date):
            #                     for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
            #                                 '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):   # Supported formats
            #                         try:
            #                             # Convert to datetime
            #                             parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
            #                             if pd.notnull(parsed_date):
            #                                 return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
            #                         except ValueError:
            #                             continue
            #                     raise ValueError(f"Date '{date}' does not match any supported formats.")
                            
            #                 # Apply parsing and conversion to short format to the date column
            #                 dataframe[date_column] = dataframe[date_column].apply(parse_date)
                            
            #                 # Convert the date column back to datetime for further processing
            #                 dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                            
            #                 # Sort the DataFrame by date
            #                 dataframe = dataframe.sort_values(by=date_column)
                            
            #                 # Calculate the cutoff date based on days
            #                 recent_date = dataframe[date_column].max()
            #                 cutoff_date = recent_date - timedelta(days=days)
                            
            #                 # Filter the DataFrame based on the cutoff date
            #                 filtered_df = dataframe[dataframe[date_column] > cutoff_date]
                            
            #                 return filtered_df 


            #             # Example Usage:
            #             Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

            #         else:
            #             print("Conditions not met. Skipping execution.")


            # #---------------------------------------------------------------------------

            # # # Taking User Input for Brand Names

            # #     if "config_data" in st.session_state:
            # #         config_df=pd.DataFrame([st.session_state.config_data])
            # #         # Check the conditions
            # #         if (config_df['Modeling Type'].eq(1).any() and 
            # #             config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
            # #             config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):
                        
            # #             # Take user input as a single string
            # #             user_input = input("Enter Elements of Variable which is used as L0, separated by commas: ")

            # #             # Split the input string into a list and convert each item to lowercase
            # #             brand_names = [name.strip().lower() for name in user_input.split(',')]

            # #             print(brand_names)
            # #         else:
            # #             print("Conditions not met. Skipping execution.")



            # #--------------------------------------------------------------

            # # User Input for L0 Variable

                def select_defined_dataframe(df_dict, selected_key):
                    """
                    Selects a single DataFrame from a dictionary based on user input,
                    ignoring those that are not defined.

                    Args:
                        df_dict (dict): A dictionary where keys are DataFrame names (strings)
                                        and values are the DataFrame objects.
                        selected_key (str): The name of the DataFrame to select.


                    Returns:
                        pd.DataFrame: The selected DataFrame, or None if the selection is invalid.
                    """
                    # Filter the dictionary to include only defined DataFrames
                    defined_dfs = {key: df for key, df in df_dict.items() if isinstance(df, pd.DataFrame)}

                    if not defined_dfs:
                        print("No DataFrames are defined!")
                        return None

                    print("Available DataFrames:")
                    for key in defined_dfs.keys():
                        print(f"- {key}")

                    # Validate the input key
                    if selected_key in defined_dfs:
                        print(f"\nSelected DataFrame: {selected_key}")
                        return defined_dfs[selected_key]
                    else:
                        print("Invalid selection! Please try again.")
                        return None

                # if "df_L0" not in st.session_state:
                #     st.session_state.df_L0 = None

                # if "config_data" in st.session_state:
                #     config_df=pd.DataFrame([st.session_state.config_data])

                #     if config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any():

                #         # Get user input outside the function
                #         selected_key_L0 = config_df.loc[0, 'Variable to use as L0']
                #         st.session_state.selected_key_L0 = selected_key_L0

                #         # Call the function with the user input
                #         df_L0 = select_defined_dataframe(df_dict, selected_key_L0)
                        
                #         if df_L0 is not None:
                #             st.session_state.df_L0 = df_L0
                #             print("\nHere is the selected DataFrame:")
                #             print(df_L0)
                #         else:
                #             st.error("df_L0 not found in session state!")
                        

                #     else:
                #         print("The value in 'MSP at L0 (Yes/No)' is not 'Yes'. Skipping execution.")




            #---------------------------------------------------------------

            # User Input for L2 Variable

                if "df_L0" not in st.session_state:
                    st.session_state.df_L0 = None


                if "df_L2" not in st.session_state:
                    st.session_state.df_L2 = None

                if "config_data" in st.session_state:
                    config_df=pd.DataFrame([st.session_state.config_data])
                    if (config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and 
                        config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                        # Get user input outside the function
                        selected_key_L2 = config_df.loc[0, 'Variable to use as L2']
                        st.session_state.selected_key_L2 = selected_key_L2
                        # st.write(selected_key_L2)

                        # Call the function with the user input
                        df_L2 = select_defined_dataframe(df_dict, selected_key_L2)
                        # st.write(df_L2)

                        if df_L2 is not None:
                            st.session_state.df_L2 = df_L2
                            print("\nHere is the selected DataFrame:")
                            # st.write('df_L0')

                            # st.write(df_L0)

                            # st.write('df_L2')
                            # st.write(df_L2)
                            
                    else:
                        print("The value in 'MSP at L2 (Yes/No)' is not 'Yes'. Skipping execution.")



            #------------------------------------------------------

                # if "config_data" in st.session_state:
                #     config_df=pd.DataFrame([st.session_state.config_data])
                #     if (config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and 
                #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                #         config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                #         # Get user input outside the function
                #         selected_key_L3 = config_df.loc[0, 'Variable to use as L3']

                #         # Call the function with the user input
                #         df_L3 = select_defined_dataframe(df_dict, selected_key_L3)

                #         if df_L3 is not None:
                #             st.session_state.df_L3 = df_L3
                #             print("\nHere is the selected DataFrame:")
                #             print(df_L3)

                #     else:
                #         print("The value in 'MSP at L3 (Yes/No)' is not 'Yes'. Skipping execution.")



            #---------------------------------------------------------------

            # Selecting desired Volume Units Column from the DF

                def adjust_volume_based_on_other_df(original_df, other_df):
                    """
                    Adjusts the 'Volume' or 'VolumeUnits' columns in the original DataFrame based on the 'Vol_Var' column in another DataFrame.

                    Parameters:
                        original_df (pd.DataFrame): The original DataFrame containing 'Volume' and 'VolumeUnits' columns.
                        other_df (pd.DataFrame): The other DataFrame containing the 'Vol_Var' column.

                    Returns:
                        pd.DataFrame: The adjusted original DataFrame.

                    Raises:
                        ValueError: If 'Vol_Var' is missing in other_df or contains invalid/multiple unique values.
                    """
                    # Ensure 'Vol_Var' column exists in the other DataFrame
                    if 'Vol_Var' not in other_df.columns:
                        raise ValueError("The other DataFrame must contain a 'Vol_Var' column.")
                    
                    # Get unique values in 'Vol_Var'
                    vol_var_values = other_df['Vol_Var'].unique()
                    
                    # Ensure 'Vol_Var' contains only one unique value
                    if len(vol_var_values) != 1:
                        raise ValueError("The 'Vol_Var' column must contain a single unique value across the DataFrame.")
                    
                    vol_var_value = vol_var_values[0]
                    
                    # Adjust original_df based on 'Vol_Var' value
                    if vol_var_value == 'Volume':
                        # Drop 'VolumeUnits' and rename 'Volume' in the original DataFrame
                        if 'VolumeUnits' in original_df.columns:
                            original_df = original_df.drop(columns=['VolumeUnits'])
                        original_df = original_df.rename(columns={'Volume': 'Volume'})
                    
                    elif vol_var_value == 'VolumeUnits':
                        # Drop 'Volume' and rename 'VolumeUnits' in the original DataFrame
                        if 'Volume' in original_df.columns:
                            original_df = original_df.drop(columns=['Volume'])
                        original_df = original_df.rename(columns={'VolumeUnits': 'Volume'})
                    
                    else:
                        raise ValueError("Invalid value in 'Vol_Var' column. Expected 'Volume' or 'VolumeUnits'.")

                    return original_df




                def add_vol_var_column(target_df, reference_df, column_name):
                    """
                    Adds a new column to the target DataFrame based on the unique value in the reference DataFrame's Vol_Var column.
                    
                    Parameters:
                    - target_df (pd.DataFrame): The target DataFrame where the column will be added.
                    - reference_df (pd.DataFrame): The reference DataFrame containing the Vol_Var column.
                    - column_name (str): The name of the column to be added to the target DataFrame.
                    
                    Returns:
                    - pd.DataFrame: The updated target DataFrame with the new column added.
                    """
                    # Ensure the Vol_Var column exists in the reference DataFrame
                    if "Vol_Var" not in reference_df.columns:
                        raise ValueError("The reference DataFrame must contain the 'Vol_Var' column.")
                    
                    # Check if all values in the Vol_Var column are the same
                    unique_values = reference_df["Vol_Var"].unique()
                    if len(unique_values) != 1:
                        raise ValueError("The 'Vol_Var' column in the reference DataFrame must have a single unique value.")
                    
                    # Get the unique value
                    vol_var_value = unique_values[0]
                    
                    # Add the new column to the target DataFrame with the same value
                    target_df[column_name] = vol_var_value
                    
                    return target_df









                D0['Channel'] = D0['Channel'].str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.replace("'", '', regex=False).str.lower()
                D0['Brand'] = D0['Brand'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['Market'] = D0['Market'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['PPG'] = D0['PPG'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['Variant'] = D0['Variant'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['Category'] = D0['Category'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['PackType'] = D0['PackType'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['PackSize'] = D0['PackSize'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                D0['SubCategory'] = D0['SubCategory'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()




                # if 'df_L0' in st.session_state:
                #     df_L0 = st.session_state.df_L0
                #     D0 = adjust_volume_based_on_other_df(D0, df_L0)
            
                # else:
                #     st.error("df_L0 not found in session state!")

                # D0_L0 = D0.copy()
                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0
                #     D0_L0[selected_key_L0] = D0_L0[selected_key_L0].astype('str')





                # # Drop columns dynamically
                # D0_L0 = D0_L0.drop(columns=[
                #         'Year', 'Month', 'Week', 'BrCatId'
                #     ])
                


                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0
                # # Group the DataFrame
                #     D0_L0 = (
                #         D0_L0.groupby(['Market', 'Channel','Category',selected_key_L0, 'date'], as_index=False)
                #         .agg({
                #             'SalesValue': 'sum',  # Sum of SalesValue
                #             'Volume': 'sum',      # Sum of Volume 
                #         })
                #     )



                # if 'df_L0' in st.session_state and 'selected_key_L0' in st.session_state:
                #     df_L0 = st.session_state.df_L0
                #     selected_key_L0 = st.session_state.selected_key_L0

                #     columns_to_keep = ['Market', 'Channel', selected_key_L0, 'MCV.MCV']  # Replace with your actual column names
                #     df_L0_L0 = df_L0[columns_to_keep]
                #     df_L0_L0[selected_key_L0] = df_L0_L0[selected_key_L0].astype('str')
                # else:
                #     st.error("Either 'df_L0' or 'selected_key_L0' is not in session_state.")


                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0

                #     D0_L0['Revenue_Total'] = D0_L0.groupby(['Market', 'Channel', selected_key_L0])['SalesValue'].transform('sum')
                #     D0_L0['Volume_Total'] = D0_L0.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')
                #     D0_L0['Price_ModelPeriod'] = D0_L0['Revenue_Total']/D0_L0['Volume_Total'] 




                # D0_filtered = defined_period_data(D0_L0, Reference_Period)



                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0

                #     D0_filtered['Revenue'] = D0_filtered.groupby(['Market', 'Channel', selected_key_L0])['SalesValue'].transform('sum')
                #     D0_filtered['Volume_definedperiod'] = D0_filtered.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')

                #     D0_filtered['Price'] = D0_filtered['Revenue']/D0_filtered['Volume_definedperiod'] 

                #     D0_filtered['Avg Volume'] = D0_filtered['Volume_definedperiod']/Reference_Period

                #     D0_filtered = D0_filtered.drop(columns=['date', 'SalesValue', 'Volume'])

                #     D0_filtered = D0_filtered.drop_duplicates(subset=['Market','Channel',selected_key_L0], keep='first')

                #     MSP_L0 = pd.merge(df_L0_L0,D0_filtered, on = ['Market','Channel',selected_key_L0], how='left')
            


                # MSP_L0['Total_Sales_Per_Channel'] = MSP_L0.groupby(['Market','Channel'])['Revenue'].transform('sum')
                # MSP_L0['MShare'] = MSP_L0['Revenue']/MSP_L0['Total_Sales_Per_Channel']

                # MSP_L0.rename(columns={'Volume_definedperiod': 'Volume'}, inplace=True)

                # MSP_L0['CS'] = 0.5 * (MSP_L0['MCV.MCV'] + MSP_L0['Price']) * MSP_L0['Avg Volume']

                # MSP_L0.groupby(['Market','Channel'])['MShare'].sum()

                # MSP_L0['Sum_CS_by_Channel'] = MSP_L0.groupby(['Market','Channel'])['CS'].transform('sum')
                # MSP_L0['Sum_MShare_by_Channel'] = MSP_L0.groupby(['Market','Channel'])['MShare'].transform('sum')

                # # Step 2: Calculate PVS for each row
                # MSP_L0['PVS'] = (MSP_L0['CS'] / MSP_L0['Sum_CS_by_Channel']) * MSP_L0['Sum_MShare_by_Channel']

                # MSP_L0['MSP'] = MSP_L0['PVS'] - MSP_L0['MShare']

                # MSP_L0 = MSP_L0.drop(columns=['Sum_CS_by_Channel','Sum_MShare_by_Channel','PVS'])

                # MSP_L0['CSF'] = MSP_L0['MCV.MCV'] / MSP_L0['Price']

                # MSP_L0['NewMShare'] = MSP_L0['MShare'] + MSP_L0['MSP']

                # MSP_L0['Price_elas'] = MSP_L0['Price_ModelPeriod'] /(MSP_L0['Price'] - MSP_L0['MCV.MCV'])

                # MSP_L0.rename(columns={'MCV.MCV': 'MCV'}, inplace=True)

                # MSP_L0['Market'] = MSP_L0['Market'].str.lower()
                # MSP_L0['Category'] = MSP_L0['Category'].str.lower()


                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0
                #     columns_to_drop = [
                #     col for col in ['Region', 'Brand', 'SubCategory', 'Variant', 'PackType', 'PPG', 'PackSize']
                #     if col != selected_key_L0
                #     ]

                #     # Drop the remaining columns
                #     MSP_L0 = MSP_L0.drop(columns=columns_to_drop, errors='ignore')


                # for file_name in weighted_files:
                #     if file_name in st.session_state:
                #         try:
                #             L0_Elasticity = st.session_state[file_name]

                #         except Exception as e:
                #             st.sidebar.error(f"Error processing {file_name}: {e}")


                # result_dfs = []


                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0
                #     all_columns = list(L0_Elasticity.columns)

                #     # Ensure 'Category_elas' and 'beta0' exist
                #     if 'Category_elas' in all_columns and 'beta0' in all_columns:
                #         # Find the positions of 'Category_elas' and 'beta0'
                #         start_index = all_columns.index('Category_elas')
                #         end_index = all_columns.index('beta0')

                #         # Get the columns that lie between 'Category_elas' and 'beta0', excluding them
                #         columns_in_range = all_columns[start_index + 1:end_index]
                #     else:
                #         raise ValueError("'Category_elas' or 'beta0' column not found in the DataFrame.")

                #     # Define conditions (columns to include in the analysis)
                #     conditions = ['Market', 'Channel', selected_key_L0]
                #     valid_conditions = [cond for cond in conditions if cond in L0_Elasticity.columns]
                #     relevant_columns = valid_conditions + columns_in_range

                #     # Process each unique channel
                #     for channel in L0_Elasticity['Channel'].unique():
                #         # Filter rows for the current channel
                #         channel_df = L0_Elasticity[L0_Elasticity['Channel'] == channel]

                #         # Process each unique brand within the current channel
                #         for element in channel_df[selected_key_L0].unique():
                #             # Filter rows for the current brand within the channel
                #             element_df = channel_df[channel_df[selected_key_L0] == element].copy()

                #             # # Apply multiple conditions
                #             # element_df = element_df[
                #             #     (element_df['method'].isin(methods_to_include)) &
                #             #     (element_df['Price_pval'] == 'Yes') &
                #             #     (element_df['Distribution_beta'] > 0) &
                #             #     (element_df['Price_beta'] < 0)
                #             # ]

                #             # Ensure only relevant columns are included
                #             element_df = element_df[[col for col in relevant_columns if col in element_df.columns]]

                #             # Calculate the mean for the relevant columns in columns_in_range
                #             for col in columns_in_range:
                #                 if col in element_df.columns:
                #                     # Calculate mean only for values greater than zero
                #                     element_df[col] = element_df[col][element_df[col] > 0].mean()

                #             # Append the processed data to the result list
                #             result_dfs.append(element_df)

                #     # Concatenate all processed DataFrames into a single DataFrame
                #     result_df = pd.concat(result_dfs, ignore_index=True)

                #     # Drop duplicates to retain one row per unique Market, Channel, and Brand
                #     result_df = result_df.drop_duplicates(subset=['Market', 'Channel', selected_key_L0], keep='first')

                

                # MSP_L0 = pd.merge(MSP_L0,result_df, on = ['Market','Channel',selected_key_L0], how='left')
                # MSP_L0 = MSP_L0.dropna(subset=['CSF'])

                # MSP_L0.fillna('NA', inplace=True)



                # if 'selected_key_L0' in st.session_state:
                #     selected_key_L0 = st.session_state.selected_key_L0
                #     # Dynamic column ordering
                #     unique_elements = [element for element in L0_Elasticity[selected_key_L0].unique()]

                #     #fixed_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand','Variant', 'PackType', 'PPG','MCV','Price_elas','Revenue', 'Volume','Price','CSF']  # Columns to place at the beginning
                #     fixed_columns = ['Market', 'Channel', 'Category', selected_key_L0, 'MCV','Price_elas','Revenue', 'Volume','Price','CSF']  # Columns to place at the beginning
                #     dot_columns = sorted([
                #         col for col in MSP_L0.columns 
                #         if '.' in col 
                #         #or col.startswith('restofcategory') 
                #         or col in unique_elements
                #     ])  # Columns with '.' in alphabetical order
                #     remaining_columns = ['Revenue_Total','Volume_Total', 'Price_ModelPeriod','CS','MShare','NewMShare','MSP']  # Columns to place at the end

                #     # Combine the desired order
                #     new_order = fixed_columns + dot_columns + remaining_columns

                #     # Reorder the DataFrame
                #     MSP_L0 = MSP_L0[new_order]

                #     # Assume `final_df` is the target DataFrame and `reference_df` is the other DataFrame.
                #     column_name_to_add = "Vol_Var"  # Name of the column to create in the final DataFrame
                #     MSP_L0 = add_vol_var_column(MSP_L0, df_L0, column_name_to_add)


        ## MSP_L0 for Type 2----------------------------------

                # if "config_data" in st.session_state:
                #     config_df=pd.DataFrame([st.session_state.config_data])
                #     if (config_df['Modeling Type'].eq(2).any() and 
                #         config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                #         MSP_L0_T2 = MSP_L0.copy()
                #         MSP_L0_T2['CsfPeriod'] = Reference_Period
                #         MSP_L0_T2 = MSP_L0

                #         st.session_state["MSP_L0"]=MSP_L0


                    
                #         # st.write('MSP_L0')

                #         # st.dataframe(MSP_L0)  # Keep height limited



        # MSP_L0 fro Type 1 w/o MSP_L0L2------------------------------------------

                # if "config_data" in st.session_state:
                #     config_df=pd.DataFrame([st.session_state.config_data])
                #     if (config_df['Modeling Type'].eq(1).any() and 
                #         config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('no').any() and config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('no').any()) :

                #         MSP_L0_T1_direct = MSP_L0.copy()
                #         MSP_L0_T1_direct['CsfPeriod'] = Reference_Period
                #         MSP_L0_T1_direct= MSP_L0



                #         st.session_state["MSP_L0"]=MSP_L0


                # st.write('MSP L0 FILE:')
                # st.dataframe(MSP_L0)


                # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    





                # if "MSP_L0" in st.session_state and st.session_state["MSP_L0"] is not None:
                #     # Convert DataFrame to HTML with preserved index
                #     table_html = st.session_state["MSP_L0"].to_html(index=True)

                #     # Sticky and Scrollable Table
                #     st.markdown(
                #         """
                #         <style>
                #             .scrollable-table-container {
                #                 position: fixed;
                #                 top: 330px;
                #                 right: 1%;
                #                 width: 40%;
                #                 height: 400px;
                #                 overflow-x: auto !important;
                #                 overflow-y: auto !important;
                #                 z-index: 1000;
                #                 background: white;
                #                 border: 1px solid #ddd;
                #             }

                #             .scrollable-table-container table {
                #                 width: auto; /* Adjust table width dynamically */
                #                 border-collapse: collapse;
                #             }

                #             .scrollable-table-container th, 
                #             .scrollable-table-container td {
                #                 border: 1px solid #ddd;
                #                 padding: 8px;
                #                 text-align: left;
                #                 white-space: nowrap;
                #             }
                #         </style>
                #         """,
                #         unsafe_allow_html=True
                #     )

                #     # Display inside a scrollable div
                #     st.markdown(f'<div class="scrollable-table-container">{table_html}</div>', unsafe_allow_html=True)
                # else:
                #     st.warning("MSP_L0 data is not available. Please upload or initialize the data.")

    #----------------------------------------------------------------------------------

                # MSP_L0L2 T2


                #  # Ensure the session state variables exist before using them
                # if "D0_L2_grouped_original" not in st.session_state:
                #     st.session_state.D0_L2_grouped_original = None  # Store original values
                #     st.session_state.D0_L2_grouped_modified = False  # Track modification state
                #     st.session_state.D0_L2_grouped = None  # Store modified version


                if "config_data" in st.session_state:
                    config_df=pd.DataFrame([st.session_state.config_data])
                
                    if (config_df['Modeling Type'].eq(2).any() and config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                        config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):


                        if 'df_L2' in st.session_state and 'selected_key_L2' in st.session_state and 'df_L0' in st.session_state:
                            df_L2 = st.session_state.df_L2

                            df_L0 = st.session_state.df_L0
                            selected_key_L2 = st.session_state.selected_key_L2
                            selected_key_L0 = st.session_state.selected_key_L0


                            

                            # df_L2['Channel'] = df_L2['Channel'].str.strip()
                            # df_L2[selected_key_L2] = df_L2[selected_key_L2].str.strip()
                            # df_L2['Market'] = df_L2['Market'].str.strip()

                            # df_L2
                            # df_L2.columns
                            # Now, select specific columns from the DataFrame
                            columns_to_keep = ['Market','Channel', 'MCV.MCV']  # Replace with your actual column names
                            df_L2_L0L2 = df_L2[columns_to_keep + [selected_key_L2]]
                            df_L2_L0L2[selected_key_L2] = df_L2_L0L2[selected_key_L2].astype('str')
                            
                            # #Check if all values in the 'Channel' column are the same
                            # if df_L2_L0L2['Channel'].nunique() == 1:  # If there is only one unique value
                            #     df_L2_L0L2['Channel'] = 'all'

                            # df_L2_L0L2.columns
                            # Now, select specific columns from the DataFrame

                            

                            columns_to_keep = ['Market','Channel','Category', selected_key_L0, 'MCV.MCV']  # Replace with your actual column names
                            df_L0_L2 = df_L0[columns_to_keep]

                            # st.write(df_L0)

                            df_L0_L2[selected_key_L0] = df_L0_L2[selected_key_L0].astype('str')

                            D0_L2 = D0.copy()
                            

                            # st.write(D0_L2)
                            D0_L2[selected_key_L0] = D0_L2[selected_key_L0].astype('str')
                            D0_L2[selected_key_L2] = D0_L2[selected_key_L2].astype('str')
                            
                            # Drop columns dynamically
                            D0_L2 = D0_L2.drop(columns=[
                                    'Year', 'Month', 'Week', 'BrCatId'
                                ])
                            
                                # Group the DataFrame
                            D0_L2 = (
                                D0_L2.groupby(['Market', 'Channel','Category', selected_key_L0,selected_key_L2, 'date'], as_index=False)
                                .agg({
                                    'SalesValue': 'sum',  # Sum of SalesValue
                                    'Volume': 'sum',      # Sum of Volume 
                                })
                            )

                            
                            
                            # D0_L2
                            D0_L2_grouped_SalesValue = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['SalesValue'].sum().reset_index()
                            D0_L2_grouped_Volume = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].sum().reset_index()
                            D0_L2_grouped = pd.merge(D0_L2_grouped_SalesValue,D0_L2_grouped_Volume,on=['Market', 'Channel', selected_key_L0,selected_key_L2],how='left')

                            
                            # D0_L2_grouped
                            D0_L2_grouped = pd.merge(D0_L2_grouped,df_L0_L2,on=['Market', 'Channel', selected_key_L0],how='left')
                            # D0_L2_grouped


                            D0_L2_grouped = D0_L2_grouped.dropna(subset=['MCV.MCV'])

                            D0_L2_grouped.rename(columns={'MCV.MCV': 'MCV_L0'}, inplace=True)
                            D0_L2_grouped = pd.merge(D0_L2_grouped,df_L2_L0L2,on=['Market', 'Channel',selected_key_L2],how='left')

                            D0_L2_grouped = D0_L2_grouped.dropna(subset=['MCV.MCV'])

                            D0_L2_grouped.rename(columns={'MCV.MCV': 'MCV_L2'}, inplace=True)
                            D0_L2_grouped['Total_Volume_ByChannel'] = D0_L2_grouped.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')
                            # Calculate the product of MCV_PPG and Volume
                            D0_L2_grouped['Sum_Product_ByChannel'] = (
                                D0_L2_grouped['MCV_L2'] * D0_L2_grouped['Volume']
                            ).groupby([D0_L2_grouped['Market'], D0_L2_grouped['Channel'], D0_L2_grouped[selected_key_L0]]).transform('sum')
                            # D0_L2_grouped
                            D0_L2_grouped['Wtd_MCV_ByBrand_using_L2MCV'] = D0_L2_grouped['Sum_Product_ByChannel'] / D0_L2_grouped['Total_Volume_ByChannel']
                            D0_L2_grouped['Correction_Factor'] = D0_L2_grouped['MCV_L0'] / D0_L2_grouped['Wtd_MCV_ByBrand_using_L2MCV']
                            D0_L2_grouped['MCV'] = D0_L2_grouped['Correction_Factor'] * D0_L2_grouped['MCV_L2']

                            # D0_L2_grouped
                            # Calculating Price for Whole Period Data
                            D0_L2['Revenue_Total'] = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['SalesValue'].transform('sum')
                            D0_L2['Volume_Total'] = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].transform('sum')
                            D0_L2['Price_ModelPeriod'] = D0_L2['Revenue_Total']/D0_L2['Volume_Total']
                            # Calculating Revenue, Volume and Price for defined time Period
                            D0_L2_filtered = defined_period_data(D0_L2, Reference_Period)
                            D0_L2_filtered['Revenue'] = D0_L2_filtered.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['SalesValue'].transform('sum')
                            D0_L2_filtered['Volume_definedperiod'] = D0_L2_filtered.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].transform('sum')
                            D0_L2_filtered['Price'] = D0_L2_filtered['Revenue']/D0_L2_filtered['Volume_definedperiod'] 
                            # D0_L2_filtered

                            D0_L2_filtered['Avg Volume'] = D0_L2_filtered['Volume_definedperiod']/Reference_Period


                            # Drop columns dynamically
                            D0_L2_filtered = D0_L2_filtered.drop(columns=['date', 'SalesValue', 'Volume'])


                            # Create a DataFrame with unique brands, keeping the first occurrence of other columns
                            D0_L2_filtered = D0_L2_filtered.drop_duplicates(subset=['Market', 'Channel', selected_key_L0,selected_key_L2], keep='first')
                        


                            # columns_to_keep = ['Market', 'Channel', selected_key_L0,selected_key_L2,'MCV']
                            # # # Filter the DataFrame to include only these columns
                            # Ensure D0_L2_grouped is not None before filtering columns




                            if "D0_L2_grouped" not in st.session_state:
                                st.session_state.D0_L2_grouped = None

                            columns_to_keep = ['Market', 'Channel', selected_key_L0,selected_key_L2,'MCV']# Adjust this list to keep necessary columns
                            
                            
                            D0_L2_grouped = D0_L2_grouped[columns_to_keep]
                            
                            # st.session_state.D0_L2_grouped=D0_L2_grouped


                            # Ensure the session state variables exist before using them
                            if "D0_L2_grouped_original" not in st.session_state:
                                st.session_state.D0_L2_grouped_original = None  # Store original values
                                st.session_state.D0_L2_grouped_modified = False  # Track modification state
                                # st.session_state.D0_L2_grouped = None  # Store modified version


                            # # Check if data exists
                            # if D0_L2_grouped is not None:
                            #     columns_to_keep = ['Market', 'Channel', selected_key_L0,selected_key_L2,'MCV']# Adjust this list to keep necessary columns
                            #     D0_L2_grouped = D0_L2_grouped[columns_to_keep]

                            #     # Store in session state if it's the first time loading
                            #     if st.session_state.D0_L2_grouped_original is None:
                            #         st.session_state.D0_L2_grouped_original = D0_L2_grouped.copy()
                            #         st.session_state.D0_L2_grouped = D0_L2_grouped.copy()
                            #         st.session_state.D0_L2_grouped_modified = False
                            # else:
                            #     st.error("D0_L2_grouped is None. Please check your data source.")
                            #     st.stop()



                            # # Use modified dataset if edits were made, otherwise, use the original
                            # if st.session_state.D0_L2_grouped_original is not None:
                            #     D0_L2_grouped = (
                            #         st.session_state.D0_L2_grouped
                            #         if st.session_state.get("D0_L2_grouped_modified", False)
                            #         else st.session_state.D0_L2_grouped_original.copy()
                            #     )
                            # else:
                            #     st.error("D0_L2_grouped_original is None. Please check your data initialization.")
                            #     st.stop()

                            if st.session_state.D0_L2_grouped_modified:
                                D0_L2_grouped = st.session_state.D0_L2_grouped  # Keep the saved edited version
                            else:
                                D0_L2_grouped = D0_L2_grouped[columns_to_keep].copy()  # Generate fresh copy from df_L0
                                D0_L2_grouped["original_MCV"] = D0_L2_grouped["MCV"]

                                st.session_state.D0_L2_grouped_original = D0_L2_grouped.copy()


                            st.session_state.D0_L2_grouped=D0_L2_grouped


                            


                            # Expandable editor in Streamlit
                            with st.expander("✏️EDIT MCV"):

                                # Define column configuration (e.g., making specific columns editable)
                                column_config = {
                                    "MCV": st.column_config.NumberColumn("MCV", help="Edit this value"),
                                }

                                # Editable DataFrame
                                edited_D0_L2_grouped = st.data_editor(
                                    D0_L2_grouped,
                                    column_config=column_config,
                                    disabled=[col for col in D0_L2_grouped.columns if col != "MCV"]+["original_MCV"],
                                    use_container_width=True
                                )

                                col1, col2 = st.columns(2)

                                with col1:
                                    if st.button("SAVE CHANGES", key="save_changes_D0_L2"):
                                        if not edited_D0_L2_grouped.equals(st.session_state.D0_L2_grouped_original):  # Check if edits were made
                                            st.session_state.D0_L2_grouped = edited_D0_L2_grouped.copy()
                                            st.session_state.D0_L2_grouped_modified = True  # Mark as modified
                                            st.success("Changes saved! They will persist until reset.")
                                            # st.rerun()

                                        else:
                                            st.info("No changes detected. Using the original DataFrame.")

                                with col2:
                                    if st.button("RESET", key="reset_D0_L2"):
                                        st.session_state.D0_L2_grouped = st.session_state.D0_L2_grouped_original.copy()
                                        st.session_state.D0_L2_grouped_modified = False  # Reset modification flag
                                        st.success("Data reset to original values!")
                                        # st.rerun()

                                st.info("The above edits will be considered as final and will not change even if the models are reselected.\n\nIf you want to change selected models, please click on 'RESET' before change.")






                            df_L2_L0L2 = df_L2_L0L2.drop(columns='MCV.MCV')


                            if 'D0_L2_grouped' in st.session_state:
                                D0_L2_grouped=st.session_state.D0_L2_grouped

                                # df_L2_L0L2.columns
                                MSP_L2 = pd.merge(df_L2_L0L2,D0_L2_grouped, on = ['Market', 'Channel',selected_key_L2], how='left')

                                # MSP_L2
                                MSP_L0L2 = pd.merge(MSP_L2,D0_L2_filtered, on = ['Market', 'Channel', 'Brand',selected_key_L2], how='left')
                                # MSP_L0L2
                        
                            MSP_L0L2['Total_Revenue_Per_Channel'] = MSP_L0L2.groupby(['Market','Channel'])['Revenue'].transform('sum')
                            MSP_L0L2['MShare'] = MSP_L0L2['Revenue']/MSP_L0L2['Total_Revenue_Per_Channel']
                            MSP_L0L2 = MSP_L0L2.drop(columns='Total_Revenue_Per_Channel')
                            MSP_L0L2.rename(columns={'Volume_definedperiod': 'Volume'}, inplace=True)
                            # MSP_L0L2.columns
                            MSP_L0L2['CS'] = 0.5 * (MSP_L0L2['MCV'] + MSP_L0L2['Price']) * MSP_L0L2['Avg Volume']
                            MSP_L0L2.groupby(['Market','Channel'])['MShare'].sum()
                            #Calculating PVS

                            # Step 1: Calculate the sum of TPV and MShare within each channel
                            MSP_L0L2['Sum_CS_by_Channel'] = MSP_L0L2.groupby(['Market','Channel'])['CS'].transform('sum')
                            MSP_L0L2['Sum_MShare_by_Channel'] = MSP_L0L2.groupby(['Market','Channel'])['MShare'].transform('sum')

                            # Step 2: Calculate PVS for each row
                            MSP_L0L2['PVS'] = (MSP_L0L2['CS'] / MSP_L0L2['Sum_CS_by_Channel']) * MSP_L0L2['Sum_MShare_by_Channel']
                            MSP_L0L2['MSP'] = MSP_L0L2['PVS'] - MSP_L0L2['MShare']
                            MSP_L0L2 = MSP_L0L2.drop(columns=['Sum_CS_by_Channel','Sum_MShare_by_Channel','PVS'])

                            MSP_L0L2['MCV'] = MSP_L0L2.apply(
                            lambda row: max(row['MCV'], 1.05 * row['Price']) if row['MCV'] <= row['Price'] else row['MCV'],
                            axis=1
                            )

                            MSP_L0L2['CSF'] = MSP_L0L2['MCV'] / MSP_L0L2['Price']
                            MSP_L0L2['NewMShare'] = MSP_L0L2['MShare'] + MSP_L0L2['MSP']
                            MSP_L0L2['Price_elas'] = MSP_L0L2['Price_ModelPeriod'] /(MSP_L0L2['Price'] - MSP_L0L2['MCV'])
                            MSP_L0L2['Market'] = MSP_L0L2['Market'].str.lower()
                            MSP_L0L2['Category'] = MSP_L0L2['Category'].str.lower()

                            # List of columns to replace values
                            columns_to_replace = ['Region','SubCategory', 'Variant', 'PackType', 'PPG', 'PackSize']

                            # Exclude selected_key_L2 if it exists in columns_to_replace
                            columns_to_replace = [col for col in columns_to_replace if col != selected_key_L2 and col != selected_key_L0]

                            # Replace all values in these columns with 'all'
                            MSP_L0L2[columns_to_replace] = 'all'

                            MSP_L0L2 = MSP_L0L2.loc[:, ~(MSP_L0L2 == 'all').all() | (MSP_L0L2.columns == 'Channel')]


                            # MSP_L0L2
                            # Drop rows where the 'CSF' column has NaN values
                            MSP_L0L2 = MSP_L0L2.dropna(subset=['CSF'])

                            # Reset the index (optional, for cleaner DataFrame)
                            MSP_L0L2.reset_index(drop=True, inplace=True)
                            # MSP_L0L2
                            # MSP_L0L2.shape
                            MSP_L0L2 = MSP_L0L2.drop(columns =['Avg Volume'])
                            # MSP_L0L2.columns

                            L0L2_order = ['Market','Category', 'Channel', selected_key_L0, selected_key_L2, 'MCV','original_MCV', 'Revenue', 'Volume', 'Price', 'CSF', 'Price_elas','Revenue_Total', 'Volume_Total',
                                        'Price_ModelPeriod', 'CS', 'MShare', 'NewMShare', 'MSP' ]

                            # Reorder the DataFrame
                            MSP_L0L2 = MSP_L0L2[L0L2_order]

                            # Assume `final_df` is the target DataFrame and `reference_df` is the other DataFrame.
                            column_name_to_add = "Vol_Var"  # Name of the column to create in the final DataFrame
                            MSP_L0L2 = add_vol_var_column(MSP_L0L2, df_L0, column_name_to_add)

                            MSP_L0L2['CsfPeriod'] = Reference_Period

                            
                            st.session_state["MSP_L0L2"]=MSP_L0L2

                    else:
                        st.error("FILL THE CONFUGURATION FILE FOR L2 LEVEL!")






            
                import plotly.graph_objects as go




                # options = ["GRAPHS"]
                # option = st.pills(f"SHOW GRAPHS!", options)

                # if option == "GRAPHS":


                # # Filter data based on selected categories, markets, and channels
                global_filtered_data = st.session_state["MSP_L0L2"].copy()

                # Apply filters if they exist
                # if st.session_state["selected_categories2"]:
                #     global_filtered_data = global_filtered_data[global_filtered_data['Category'].isin(st.session_state["selected_categories2"])]
                # if st.session_state["selected_markets2"]:
                #     global_filtered_data = global_filtered_data[global_filtered_data['Market'].isin(st.session_state["selected_markets2"])]
                if st.session_state["selected_channels2"]:
                    global_filtered_data = global_filtered_data[global_filtered_data['Channel'].isin(st.session_state["selected_channels2"])]

                if 'L2_name' in st.session_state and st.session_state['L2_name']:
                    L2_name_column = st.session_state['L2_name']

                    L2_name_column_options = global_filtered_data[L2_name_column].unique()
                    L2_name_column_filter = st.multiselect(
                        f"Select {L2_name_column}",
                        options=L2_name_column_options,
                        key=f"{L2_name_column}"
                    )

                    # Further filter based on selected RPIto
                    filtered_data_by_L2_name_column = global_filtered_data.copy()
                    if L2_name_column_filter:
                        filtered_data_by_L2_name_column = filtered_data_by_L2_name_column[filtered_data_by_L2_name_column[L2_name_column].isin(L2_name_column_filter)]


                

                # Check if any data is left after filtering
                if filtered_data_by_L2_name_column.empty:
                    st.warning("No data to display after applying the filters.")
                else:

                    df = filtered_data_by_L2_name_column.copy()
                    df["MSP"] = df["MSP"] * 100  
                    df["MSP_label"] = df["MSP"].apply(lambda x: f"{x:.2f}%")

                    df["MShare"] = df["MShare"] * 100  
                    df["MShare_label"] = df["MShare"].apply(lambda x: f"{x:.2f}%")

                    # Plotly chart for CSF
                    fig_csff= px.bar(
                        df,
                        x='Brand',  # Ensure index is treated as string for display
                        y='CSF',
                        template="plotly_white",
                        color='Brand',
                        text_auto=True,  # Display y-axis values on top of bars
                        hover_data=["Channel",L2_name_column],  # Display y-axis values on top of bars
                    )

                    # fig_csff.update_traces(textposition="outside")  # Position labels outside bars

                    # Customize hovertemplate for more detailed hover information
                    fig_csff.update_traces(
                        hovertemplate=
                                    'Channel: <b>%{customdata[0]}<br><br>'
                                    f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'
                                    '<b>%{x}</b><br>'  # Brand
                                    # Channel
                                    '<extra></extra>'  # Remove extra information like trace name
                    )

                    fig_csff.update_layout(
                        title="CSF",
                        xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                        yaxis=dict(title="CSF", color='black', showgrid=False, tickformat=".2f"),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        legend=dict(
                            orientation="h",  # Horizontal orientation
                            xanchor="left",  # Anchor to the left
                            x=0,  # Position at the left edge
                            yanchor="bottom",  # Anchor to the bottom of the legend
                            y=1.02  # Position above the chart
                        )
                    )

                    st.plotly_chart(fig_csff, use_container_width=True)


                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    



                    # Plotly chart for MSP
                    fig_msp = px.bar(
                        df,
                        x='Brand',  # Ensure index is treated as string for display
                        y='MSP',
                        template="plotly_white",
                        color='Brand',
                        text=df["MSP_label"],
                        hover_data=["Channel",L2_name_column],  # Display y-axis values on top of bars
                    )

                    fig_msp.update_traces(textposition="outside")  # Position labels outside bars

                    # Customize hovertemplate for more detailed hover information
                    fig_msp.update_traces(
                        hovertemplate=
                                    'Channel: <b>%{customdata[0]}<br><br>'
                                    f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'
                                    '<b>%{x}</b><br>'  # Brand
                                    # Channel
                                    '<extra></extra>'  # Remove extra information like trace name
                    )

                    fig_msp.update_layout(
                        title="MSP",
                        xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                        yaxis=dict(title="MSP", color='black', showgrid=False, tickformat=".2f"),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        legend=dict(
                            orientation="h",  # Horizontal orientation
                            xanchor="left",  # Anchor to the left
                            x=0,  # Position at the left edge
                            yanchor="bottom",  # Anchor to the bottom of the legend
                            y=1.02  # Position above the chart
                        )
                    )

                    st.plotly_chart(fig_msp, use_container_width=True)


                    st.markdown('<hr class="thin">', unsafe_allow_html=True)



                    # Plotly chart for MSP
                    fig_ms = px.bar(
                        df,
                        x='Brand',  # Ensure index is treated as string for display
                        y='MShare',
                        template="plotly_white",
                        color='Brand',
                        text=df["MShare_label"],  # Display y-axis values on top of bars
                        hover_data=["Channel",L2_name_column],  # Display y-axis values on top of bars
                    )

                    # fig_csff.update_traces(textposition="outside")  # Position labels outside bars

                    # # Customize hovertemplate for more detailed hover information
                    fig_ms.update_traces(
                        hovertemplate=
                                    'Channel: <b>%{customdata[0]}<br><br>'
                                    f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'
                                    '<b>%{x}</b><br>'  # Brand
                                    # Channel
                                    '<extra></extra>'  # Remove extra information like trace name
                    )

                    fig_ms.update_layout(
                        title="MSHARE",
                        xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                        yaxis=dict(title="MShare", color='black', showgrid=False, tickformat=".2f"),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        legend=dict(
                            orientation="h",  # Horizontal orientation
                            xanchor="left",  # Anchor to the left
                            x=0,  # Position at the left edge
                            yanchor="bottom",  # Anchor to the bottom of the legend
                            y=1.02  # Position above the chart
                        )
                    )

                    st.plotly_chart(fig_ms, use_container_width=True)


                    st.markdown('<hr class="thin">', unsafe_allow_html=True)




                    # # Create figure
                    # fig_mp = go.Figure()

                    # # Add Price bars
                    # fig_mp.add_trace(go.Bar(
                    #     x=global_filtered_data['Brand'],
                    #     y=global_filtered_data['Price'],
                    #     name='Price',
                    #     marker_color='blue'
                    # ))

                    # # Add MCV bars
                    # fig_mp.add_trace(go.Bar(
                    #     x=global_filtered_data['Brand'],
                    #     y=global_filtered_data['MCV'],
                    #     name='MCV',
                    #     marker_color='orange'
                    # ))

                    # # Update layout
                    # fig_mp.update_layout(
                    #     barmode='group',  # Group bars next to each other
                    #     title="Price & MCV",
                    #     xaxis=dict(title="Brand", color='black', showgrid=False, showticklabels=True),
                    #     yaxis=dict(title="Value", color='black', showgrid=False, tickformat=".2f"),
                    #     plot_bgcolor='white',
                    #     paper_bgcolor='white',
                    #     font_color='black',
                    #     legend=dict(
                    #         orientation="h",
                    #         xanchor="left",
                    #         x=0,
                    #         yanchor="bottom",
                    #         y=1.02
                    #     )
                    # )

                    # # Display in Streamlit
                    # st.plotly_chart(fig_mp, use_container_width=True)\


                    fig_mp = go.Figure()

                    # Add Price bars with a color scale
                    fig_mp.add_trace(go.Bar(
                        x=df['Brand'],
                        y=df['Price'],
                        name='Price',
                        text=df['Price'].round(2),  # Show values rounded to 2 decimal places
                        textposition='auto',
                        marker=dict(color=df['Brand'].astype('category').cat.codes,  # Convert Brand to category and map to a code
                                    colorscale='Viridis'),  # Apply a colorscale

                        hovertemplate=
                                    '<b>Channel:</b> %{customdata[0]}<br>' +
                                    f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'+
                                    '<b>Brand:</b> %{x}<br>' +
                                    '<b>Price:</b> %{y:.2f}<br>' ,

                        customdata=df[['Channel', L2_name_column]].values
                            
                    ))

                    # Add MCV bars with a color scale
                    fig_mp.add_trace(go.Bar(
                        x=df['Brand'],
                        y=df['MCV'],
                        text=df['MCV'].round(2),  # Show values rounded to 2 decimal places
                        textposition='auto',
                        name='MCV',
                        marker=dict(color=df['Brand'].astype('category').cat.codes,  # Convert Brand to category and map to a code
                                    colorscale='Viridis'),  # Apply a colorscale

                        hovertemplate=
                                    '<b>Channel:</b> %{customdata[0]}<br>' +
                                    f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'+
                                    '<b>Brand:</b> %{x}<br>' +
                                    '<b>Price:</b> %{y:.2f}<br>',

                        customdata=df[['Channel', L2_name_column]].values
                    ))

                    # Update layout
                    fig_mp.update_layout(
                        barmode='group',  # Group bars next to each other
                        title="Price & MCV",
                        xaxis=dict(title="Brand", color='black', showgrid=False, showticklabels=True),
                        yaxis=dict(title="Value", color='black', showgrid=False, tickformat=".2f"),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        legend=dict(
                            orientation="h",
                            xanchor="left",
                            x=0,
                            yanchor="bottom",
                            y=1.02
                        ),
                        showlegend=False
                    )

                    # Display in Streamlit
                    st.plotly_chart(fig_mp, use_container_width=True)



                    st.markdown('<hr class="thin">', unsafe_allow_html=True)




                options = ["MSPL0L2 FILE"]
                option = st.pills(f"SHOW MSP_L0L2 FILE!", options, default='MSPL0L2 FILE')

                if option == "MSPL0L2 FILE":
                    st.write('MSPL0L2 FILE:')
                    st.dataframe(st.session_state["MSP_L0L2"])


                    st.download_button("Download the final modified data MSP_L0L2", 
                        data=st.session_state["MSP_L0L2"].to_csv(index=False), 
                        file_name="MSP_L0L2.csv", 
                        mime="csv")


                st.markdown('<hr class="thin">', unsafe_allow_html=True)







#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------MSP L0L3-------------------------------------------------------------------------------------------




# with tab1:
#     uploaded_file = st.sidebar.file_uploader("Upload L2: ", type="xlsx")

#     if uploaded_file:
#         try:
#             if "uploaded_file" not in st.session_state or uploaded_file != st.session_state["uploaded_file"]:
#                 # Clear session state L2 when a new file is uploaded
#                 if "L2" in st.session_state:
#                     del st.session_state["L2"]
                
#                 # Load new data
#                 L2 = load_data(uploaded_file)

#                 # Store the file name without the extension
#                 file_name = uploaded_file.name.split('.')[0]
                
#                 # Save the uploaded file reference and L2 to session state
#                 st.session_state["uploaded_file"] = uploaded_file
#                 st.session_state["L2"] = L2


#                 st.session_state["L2file_name"] = file_name

#             else:
#                 # If the file hasn't changed, load the L2 from session state
#                 L2 = st.session_state.get("L2")

#         except Exception as e:
#             # Catch and display any errors that occur
#             st.sidebar.error(f"An error occurred while processing the file: {e}")
#     else:
#         st.sidebar.info("Please upload an Excel file to proceed.")



    if uploaded_file2:
        uploaded_file3 = st.sidebar.file_uploader("📄 Upload L3 file: ", type="xlsx")

        with st.sidebar.expander("🔧 Advanced Settings for L3", expanded=False):
            apply_method_filter3= st.checkbox("Remove 'Lasso' and 'Ridge' methods", value=True, key="l3_method")
            apply_dist_elas_filter3 = st.checkbox("Apply Distribution_elas Filter (>= 0 or NaN)", value=True, key="l3_dist_elas")
            apply_cat_elas_filter3 = st.checkbox("Apply Category_elas Filter (>= 0 or NaN)", value=True, key="l3_cat_elas")
            apply_pval_filter3 = st.checkbox("Filter Price_pval == 'Yes'", value=True, key="l3_pval")

        st.sidebar.markdown('<hr class="thin">', unsafe_allow_html=True)



        if uploaded_file3:

            # tab2, =st.tabs(["MSP L0L2"])
            with tab3:

            # uploaded_file2 = st.sidebar.file_uploader("Upload L2 file: ", type="xlsx")

                if uploaded_file3:
                    try:
                        # if "uploaded_file3" not in st.session_state or uploaded_file3 != st.session_state["uploaded_file3"]:
                        #     # Clear session state L3 when a new file is uploaded
                        #     if "L3" in st.session_state:
                        #         del st.session_state["L3"]
                            
                        #     # Load new data
                        #     L3 = load_data(uploaded_file3)

                            # # Store the file name without the extension
                            # file_name = uploaded_file3.name.split('.')[0]

                            # # Extract the ending after "Wtd_avg_MCV_" in the file name
                            # if "Wtd_avg_MCV_" in file_name:
                            #     L3_name = file_name.split("Wtd_avg_MCV_")[-1]
                            # else:
                            #     L3_name = None  # Set to None if "Wtd_avg_MCV_" is not in the filename

                            # # Save the uploaded file reference and L2 to session state
                            # st.session_state["uploaded_file3"] = uploaded_file3
                            # st.session_state["L3"] = L3
                            # st.session_state["L3file_name"] = file_name
                            # st.session_state["L3_name"] = L3_name  # Store extracted name

                        if ("uploaded_file3" not in st.session_state or uploaded_file3 != st.session_state["uploaded_file3"]) or \
                            (apply_method_filter3 != st.session_state.get("apply_method_filter3", True) or
                            apply_dist_elas_filter3 != st.session_state.get("apply_dist_elas_filter3", True) or
                            apply_cat_elas_filter3 != st.session_state.get("apply_cat_elas_filter3", True) or
                            apply_pval_filter3 != st.session_state.get("apply_pval_filter3", True)):
                            
                            # Clear session state data when a new file is uploaded
                            if "L3" in st.session_state:
                                del st.session_state["L3"]

                        
                            
                            # Load new data
                            L3 = load_data(uploaded_file3,apply_method_filter3, apply_dist_elas_filter3, apply_cat_elas_filter3, apply_pval_filter3)




                            # Save to session state
                            st.session_state["uploaded_file3"] = uploaded_file3
                            st.session_state["L3"] = L3
                            st.session_state["apply_method_filter3"] = apply_method_filter3
                            st.session_state["apply_dist_elas_filter3"] = apply_dist_elas_filter3
                            st.session_state["apply_cat_elas_filter3"] = apply_cat_elas_filter3
                            st.session_state["apply_pval_filter3"] = apply_pval_filter3

                                # Store the file name without the extension
                            file_name = uploaded_file3.name.split('.')[0]

                            # Extract the ending after "Wtd_avg_MCV_" in the file name
                            if "Wtd_avg_MCV_" in file_name:
                                L3_name = file_name.split("Wtd_avg_MCV_")[-1]
                            else:
                                L3_name = None  # Set to None if "Wtd_avg_MCV_" is not in the filename

                            # Save the uploaded file reference and L2 to session state
                            st.session_state["uploaded_file3"] = uploaded_file3
                            st.session_state["L3"] = L3
                            st.session_state["L3file_name"] = file_name
                            st.session_state["L3_name"] = L3_name  # Store extracted name

                        else:
                            # If the file hasn't changed, load the L3 from session state
                            L3 = st.session_state.get("L3")

                    except Exception as e:
                        # Catch and display any errors that occur
                        st.sidebar.error(f"An error occurred while processing the file: {e}")
                else:
                    st.sidebar.info("Please upload an Excel file to proceed.")









                col1, col2 = st.columns(2)

                with col1:
                    
                    # Streamlit app

                    if "L3" not in st.session_state:
                        st.session_state["L3"] = None

                    if "L3_name" not in st.session_state:
                        st.session_state["L3_name"] = None

                    if "L3file_name" not in st.session_state:
                        st.session_state["L3file_name"] = None

                    # if "selected_categories3" not in st.session_state:
                    #     st.session_state["selected_categories3"] = []

                    # if "selected_markets3" not in st.session_state:
                    #     st.session_state["selected_markets3"] = []

                    # if "selected_channels3" not in st.session_state:
                    #     st.session_state["selected_channels3"] = []

                    # if "apply_filter" not in st.session_state:
                    #     st.session_state.apply_filter = False  # Flag to track button 



                    # File uploader
                    # uploaded_file = st.file_uploader("Upload L0: Wtd_avg_MCV_Brand Excel file", type="xlsx")



                    # if uploaded_file2:
                        # Check if the uploaded file is different from the previous one

                    file_name3 = st.session_state["L3file_name"]
                    st.session_state[file_name3] = st.session_state["L3"].copy()
                    
                    # Ensure `st.session_state["L2"]` is not None before further processing
                    if st.session_state["L3"] is not None:
                        global_filtered_data = st.session_state["L3"].copy()

                        # st.markdown('<hr class="thick">', unsafe_allow_html=True)  # Add thick line after global filters

                        # st.write("Global Filters")
                        # col3, col4, col5 = st.columns(3)

                        # Initialize stateful filter selections for global filters
                        # selected_categories3 = st.session_state.get("selected_categories3", [])
                        # selected_markets3 = st.session_state.get("selected_market3", [])
                        selected_channels3 = st.session_state.get("selected_channels3", [])

                        # Filter L3 dynamically based on global filter selections
                        filtered_global_data = L3.copy()
                        # if selected_categories3:
                        #     filtered_global_data = filtered_global_data[filtered_global_data['Category'].isin(selected_categories3)]
                        # if selected_markets3:
                        #     filtered_global_data = filtered_global_data[filtered_global_data['Market'].isin(selected_markets3)]
                        if selected_channels3:
                            filtered_global_data = filtered_global_data[filtered_global_data['Channel'].isin(selected_channels3)]

                        # Update global filter options dynamically
                        # updated_categories3 = filtered_global_data['Category'].unique()
                        # updated_markets3 = filtered_global_data['Market'].unique()
                        updated_channels3 = filtered_global_data['Channel'].unique()
                        
                        
                        # Render global filters
                        # with col3:
                        #     selected_categories = st.multiselect(
                        #         "Select Category",
                        #         options=updated_categories3,
                        #         default=selected_categories3 if selected_categories3 else updated_categories3[:1],
                        #         key="selected_categories3",
            
                        #     )

                        # with col4:
                        #     selected_markets = st.multiselect(
                        #         "Select Market",
                        #         options=updated_markets3,
                        #         default=selected_markets3 if selected_markets3 else updated_markets3[:1],
                        #         key="selected_markets3",
                
                        #     )

                        # with col5:
                        selected_channels = st.multiselect(
                            "Select Channel",
                            options=updated_channels3,
                            default=st.session_state.get("selected_channels3", []),
                            key="selected_channels3",
                
                        )

                        # if st.button('APPLY FILTERS L0L2L3'):
                        #     st.write("")


                        # # Apply filter button
                        # if st.button("Apply Filter"):
                        #     st.session_state["selected_categories"] = updated_categories
                        #     st.session_state["selected_markets"] = updated_markets
                        #     st.session_state["selected_channels"] = updated_channels



                        
                        st.markdown('<hr class="thick">', unsafe_allow_html=True)



                        if 'L3_name' in st.session_state and st.session_state['L3_name']:
                            L3_name_column = st.session_state['L3_name']



                            # Display each brand's L3 and filters
                            unique_L3_values  = filtered_global_data[L3_name_column].unique()

                            for unique_value  in unique_L3_values:
                                st.header(f"{L3_name_column} : {unique_value}")
                                L3_name_column_data = filtered_global_data[filtered_global_data[L3_name_column] == unique_value]

                                # Brand-specific filters beside graphs
                                col6, col7 = st.columns([3, 1])  # Make graphs column wider

                                with col7:
                                    # Initialize filters dynamically
                                    methods = st.multiselect(
                                        f"Select Method for {unique_value}",
                                        options=brand_data['method'].unique(),
                                        key=f"method_{unique_value}"
                                    )

                                    # Filter brand_data dynamically based on the selected method
                                    filtered_data_by_method = L3_name_column_data.copy()
                                    if methods:
                                        filtered_data_by_method = filtered_data_by_method[filtered_data_by_method['method'].isin(methods)]

                                    
                                    rpito_options = filtered_data_by_method['RPIto'].unique()
                                    rpito_filter = st.multiselect(
                                        f"RPIto for {unique_value}",
                                        options=rpito_options,
                                        key=f"rpito_{unique_value}"
                                    )

                                    # Further filter based on selected RPIto
                                    filtered_data_by_rpito = filtered_data_by_method.copy()
                                    if rpito_filter:
                                        filtered_data_by_rpito = filtered_data_by_rpito[filtered_data_by_rpito['RPIto'].isin(rpito_filter)]


                                    
                                    actualdistvar_options = filtered_data_by_rpito['actualdistvar'].unique()
                                    actualdistvar_filter = st.multiselect(
                                        f"actualdistvar for {unique_value}",
                                        options=actualdistvar_options,
                                        key=f"actualdistvar_{unique_value}"
                                    )

                                    # Further filter based on selected RPIto
                                    filtered_data_by_actualdistvar = filtered_data_by_rpito.copy()
                                    if actualdistvar_filter:
                                        filtered_data_by_actualdistvar = filtered_data_by_actualdistvar[filtered_data_by_actualdistvar['actualdistvar'].isin(actualdistvar_filter)]
                
                                    

                                    # Handle sliders with identical min and max
                                    adjrsq_min = float(filtered_data_by_actualdistvar['Adj.Rsq'].min())
                                    adjrsq_max = float(filtered_data_by_actualdistvar['Adj.Rsq'].max())
                                    if adjrsq_min == adjrsq_max:
                                        adjrsq_min -= 0.01
                                        adjrsq_max += 0.01
                                    adjrsquare_filter = st.slider(
                                        f"Adj Rsquare for {unique_value}",
                                        min_value=adjrsq_min,
                                        max_value=adjrsq_max,
                                        value=(adjrsq_min, adjrsq_max),
                                        key=f"adjrsq_{unique_value}"
                                    )

                                    # Further filter based on Adj.Rsq
                                    filtered_data_by_adjrsq = filtered_data_by_actualdistvar.copy()
                                    filtered_data_by_adjrsq = filtered_data_by_adjrsq[
                                        (filtered_data_by_adjrsq['Adj.Rsq'] >= adjrsquare_filter[0]) & 
                                        (filtered_data_by_adjrsq['Adj.Rsq'] <= adjrsquare_filter[1])
                                    ]

                                    aic_min = float(filtered_data_by_adjrsq['AIC'].min())
                                    aic_max = float(filtered_data_by_adjrsq['AIC'].max())
                                    if aic_min == aic_max:
                                        aic_min -= 0.01
                                        aic_max += 0.01
                                    aic_filter = st.slider(
                                        f"AIC for {unique_value}",
                                        min_value=aic_min,
                                        max_value=aic_max,
                                        value=(aic_min, aic_max),
                                        key=f"aic_{unique_value}"
                                    )

                                    # Further filter based on AIC
                                    filtered_data_by_aic = filtered_data_by_adjrsq.copy()
                                    filtered_data_by_aic = filtered_data_by_aic[
                                        (filtered_data_by_aic['AIC'] >= aic_filter[0]) & 
                                        (filtered_data_by_aic['AIC'] <= aic_filter[1])
                                    ]

                                    csf_min = float(filtered_data_by_aic['CSF.CSF'].min())
                                    csf_max = float(filtered_data_by_aic['CSF.CSF'].max())
                                    if csf_min == csf_max:
                                        csf_min -= 0.01
                                        csf_max += 0.01
                                    csf_filter = st.slider(
                                        f"CSF for {unique_value}",
                                        min_value=csf_min,
                                        max_value=csf_max,
                                        value=(csf_min, csf_max),
                                        key=f"csf_{unique_value}"
                                    )

                                    # Final filtered data based on CSF
                                    filtered_data = filtered_data_by_aic.copy()
                                    filtered_data = filtered_data[
                                        (filtered_data['CSF.CSF'] >= csf_filter[0]) & 
                                        (filtered_data['CSF.CSF'] <= csf_filter[1])
                                    ]




                                # Ensure 'selectedmodels' column exists
                                filtered_data["selectedmodels"] = "No"

                                # Initialize session state if not present
                                if "saved_selections" not in st.session_state:
                                    st.session_state["saved_selections"] = {}

                                if "saved_buttons" not in st.session_state:
                                    st.session_state["saved_buttons"] = {}

                                if "reset_buttons" not in st.session_state:
                                    st.session_state["reset_buttons"] = {}

                                # Function to update 'selectedmodels' column ensuring one "Yes" per (Channel, unique_value)
                                def update_selectedmodels(group, unique_value):
                                    channel = group["Channel"].iloc[0]  # Get the Channel for this group

                                    # Check if selection was saved for this (Channel, unique_value)
                                    key = (channel, unique_value)
                                    if key in st.session_state["saved_selections"]:
                                        saved_index = st.session_state["saved_selections"][key]
                                        if saved_index in group.index:
                                            group["selectedmodels"] = "No"
                                            group.loc[saved_index, "selectedmodels"] = "Yes"
                                            return group  # Return saved selection

                                    # Otherwise, select the row closest to the median CSF.CSF
                                    if not group.empty:
                                        median_csf = group["CSF.CSF"].median()
                                        closest_index = (group["CSF.CSF"] - median_csf).abs().idxmin()
                                        group["selectedmodels"] = "No"
                                        group.loc[closest_index, "selectedmodels"] = "Yes"

                                    return group

                                # Apply selection logic per (Channel, unique_value)
                                filtered_data = filtered_data.groupby(["Channel", L3_name_column], group_keys=False).apply(update_selectedmodels, unique_value=unique_value)

                                # UI for selecting models
                                for (channel, unique_value), group in filtered_data.groupby(["Channel", L3_name_column]):
                                    available_indices = group.index.tolist()
                                    
                                    default_index = (
                                        group[group["selectedmodels"] == "Yes"].index[0]
                                        if "Yes" in group["selectedmodels"].values
                                        else available_indices[0]
                                    )

                                    selected_index = st.selectbox(
                                        f"Select Model Index for {channel} - {unique_value}",
                                        options=available_indices,
                                        index=available_indices.index(default_index),
                                        key=f"selectbox_{channel}_{unique_value}"
                                    )

                                    # Ensure selectedmodels is locked once saved
                                    if selected_index in group.index:
                                        key = (channel, unique_value)
                                        if key in st.session_state["saved_selections"]:
                                            saved_index = st.session_state["saved_selections"][key]
                                            group["selectedmodels"] = "No"
                                            group.loc[saved_index, "selectedmodels"] = "Yes"
                                        else:
                                            group["selectedmodels"] = "No"
                                            group.loc[selected_index, "selectedmodels"] = "Yes"

                                    col3,col4=st.columns([3,2])

                                    # Dynamically set button text based on session state
                                    save_button_text = st.session_state["saved_buttons"].get((channel, unique_value), f"SAVE Selection {channel} - {unique_value}")
                                    reset_button_text = st.session_state["reset_buttons"].get((channel, unique_value), f"RESET Selection {channel} - {unique_value}")

                                    with col3:
                                        # Button to save selection
                                        if st.button(save_button_text, key=f"save_{channel}_{unique_value}"):
                                            if (channel, unique_value) in st.session_state["saved_selections"]:
                                                saved_index = st.session_state["saved_selections"][(channel, unique_value)]
                                                if saved_index != selected_index:
                                                    st.warning(f"Selection already saved for {channel} - {unique_value} at index {saved_index}.\n\n Please 'RESET' first before changing.")
                                                else:
                                                    st.success(f"Selection already saved for {channel} - {unique_value} at index {saved_index}.")
                                            else:
                                                st.session_state["saved_selections"][(channel, unique_value)] = selected_index
                                                st.session_state["saved_buttons"][(channel, unique_value)] = f"SAVED ✅ ({channel} - {unique_value})"
                                                st.session_state["reset_buttons"][(channel, unique_value)] = f"RESET Selection {channel} - {unique_value}"
                                                st.success(f"Selection saved for {channel} - {unique_value} at index {selected_index}.")

                                    with col4:
                                        # Button to reset selection
                                        if st.button(reset_button_text, key=f"reset_{channel}_{unique_value}"):
                                            if (channel, unique_value) in st.session_state["saved_selections"]:
                                                del st.session_state["saved_selections"][(channel, unique_value)]
                                            st.session_state["reset_buttons"][(channel, unique_value)] = f"RESET 🔄 ({channel} - {unique_value})"
                                            st.session_state["saved_buttons"][(channel, unique_value)] = f"SAVE Selection {channel} - {unique_value}"
                                            st.success(f"Selection reset for {channel} - {unique_value}.\n\n Now updates dynamically.")

                                    # Store updates in session state
                                    st.session_state["filtered_data"] = filtered_data

                                    # Ensure L3 is updated properly with "Yes" for each (Channel, unique_value)
                                    if "L3" in st.session_state:
                                        L3 = st.session_state["L3"]
                                        
                                        # Reset all selectedmodels to "No" for this Channel-unique_value pair in L3
                                        L3.loc[(L3["Channel"] == channel) & (L3[L3_name_column] == unique_value), "selectedmodels"] = "No"

                                        # Apply the saved selection or median-based selection
                                        selected_row = group[group["selectedmodels"] == "Yes"]
                                        if not selected_row.empty:
                                            selected_idx = selected_row.index[0]
                                            L3.loc[selected_idx, "selectedmodels"] = "Yes"

                                        st.session_state["L3"] = L3.copy()

                                        # Save updates to the session state file
                                        file_name3 = st.session_state.get("L3file_name", "L3")
                                        st.session_state[file_name3] = L3.copy()





                                with col6:
                                    # graph_option = st.radio(f"Choose graph type for {brand}", options=["MCV", "CSF"], key=f"graph_option_{brand}")
                                    options=["CSF","MCV"]
                                    graph_option = st.pills(f"Choose graph type for {unique_value}",options,selection_mode="single",default="CSF",key=f"graph_option_{unique_value}")

                                    if graph_option == "MCV":

                                        sort_by = st.radio(
                                                "SORT BY:", ["MCV.MCV", "Adj.Rsq", "AIC"], index=0, horizontal=True, key=f"sort_option_mcv_{unique_value}"
                                            )
                                        # Sort data by 'MCV.MCV' in ascending order
                                        sorted_data = filtered_data.sort_values(by=sort_by, ascending=True)
                                        sorted_data['Index'] = sorted_data.index.astype(str)
                                                            
                                        # Plotly chart for MCV with a better color palette
                                        fig_mcv = px.bar(
                                            sorted_data,
                                            x='Index',  # Use the 'Index' column for x-axis
                                            y='MCV.MCV',
                                            template="plotly_white",
                                            color='RPIto',  # Color bars based on the 'RPIto' column
                                            text_auto=True,  # Display y-axis values on top of bars
                                            category_orders={"Index": sorted_data['Index'].tolist()},  # Ensure bars follow the sorted order
                                            color_discrete_sequence=px.colors.qualitative.Set3,  # Use a better color palette
                                            hover_data={
                                                "RPIto":False,
                                                "Index": True, 
                                                "Adj.Rsq": ':.2f', 
                                                "AIC": ':.2f'
                                            }
                                        )

                                        # Update layout to remove title and position legend at top-left
                                        fig_mcv.update_layout(
                                            title="",  # Set title to empty string to avoid 'undefined'
                                            xaxis=dict(title="Index", color='black', showgrid=False, showticklabels=False),
                                            yaxis=dict(title="MCV", color='black', showgrid=False, tickformat=".1f"),
                                            plot_bgcolor='white',
                                            paper_bgcolor='white',
                                            font_color='black',
                                            legend=dict(
                                                orientation="h",  # Horizontal orientation
                                                xanchor="left",  # Anchor to the left
                                                x=0,  # Position at the left edge
                                                yanchor="bottom",  # Anchor to the bottom of the legend
                                                y=1.02  # Position above the chart
                                            )
                                        )

                                        # Display the chart
                                        st.plotly_chart(fig_mcv, use_container_width=True)


                                    elif graph_option == "CSF":

                                        sort_by = st.radio(
                                                "SORT BY:", ["CSF.CSF", "Adj.Rsq", "AIC"], index=0, horizontal=True, key=f"sort_option_csf_{unique_value}"
                                            )
                                        # Sort data by 'CSF.CSF' in ascending order
                                        sorted_data = filtered_data.sort_values(by=sort_by, ascending=True)
                                        sorted_data['Index'] = sorted_data.index.astype(str)
                                        
                                        # Plotly chart for CSF
                                        fig_csf = px.bar(
                                            sorted_data,
                                            x='Index',  # Ensure index is treated as string for display
                                            y='CSF.CSF',
                                            template="plotly_white",
                                            # color_discrete_sequence=["#FFD700"],
                                            color='RPIto',
                                            text_auto=True,  # Display y-axis values on top of bars
                                            category_orders={"Index": sorted_data['Index'].tolist()},
                                            color_discrete_sequence=px.colors.qualitative.Set3,  # Use a better color palette
                                            hover_data={
                                                "RPIto":False,
                                                "Index": True, 
                                                "Adj.Rsq": ':.2f', 
                                                "AIC": ':.2f'
                                            }


                                        )
                                        fig_csf.update_layout(
                                            title="",
                                            xaxis=dict(title="Index", color='black', showgrid=False,showticklabels=False),
                                            yaxis=dict(title="CSF", color='black', showgrid=False,tickformat=".2f"),
                                            plot_bgcolor='white',
                                            paper_bgcolor='white',
                                            font_color='black',
                                            legend=dict(
                                                orientation="h",  # Horizontal orientation
                                                xanchor="left",  # Anchor to the left
                                                x=0,  # Position at the left edge
                                                yanchor="bottom",  # Anchor to the bottom of the legend
                                                y=1.02  # Position above the chart
                                            )
                                        )
                                        st.plotly_chart(fig_csf, use_container_width=True)

                                

                                # options=["MAKE SELECTIONS","SUBMIT SELECTIONS"]
                                # submit_option = st.pills(f"",options,selection_mode="single",default="SUBMIT SELECTIONS",key=f"show_data_{unique_value}")
                                # # Checkbox for brand-specific data
                                # # if st.checkbox(f"Show Data for {unique_value}", key=f"show_data_{unique_value}"):
                                # if submit_option=="MAKE SELECTIONS":
                                with st.expander("SHOW DATA"):
                                
                                    # st.error("CLICK ON 'SUBMIT SELECTIONS' TO PERMANENTLY SAVE THE CHANGES.")
                                    # Apply global filters to the full dataset
                                    global_filtered_data = st.session_state["L3"].copy()
                                    # if selected_categories3:
                                    #     global_filtered_data = global_filtered_data[global_filtered_data['Category'].isin(selected_categories3)]
                                    # if selected_markets3:
                                    #     global_filtered_data = global_filtered_data[global_filtered_data['Market'].isin(selected_markets3)]
                                    if selected_channels3:
                                        global_filtered_data = global_filtered_data[global_filtered_data['Channel'].isin(selected_channels3)]

                                    # Extract the brand-specific data from the globally filtered data
                                    L3_name_column_data = global_filtered_data.loc[global_filtered_data[L3_name_column] == unique_value].copy()

                                    # Apply brand-specific filters
                                    filtered_data = L3_name_column_data.copy()
                                    if methods:
                                        filtered_data = filtered_data[filtered_data['method'].isin(methods)]

                
                                    if rpito_filter:
                                        filtered_data = filtered_data[filtered_data['RPIto'].isin(rpito_filter)]


                                    
                                    if actualdistvar_filter:
                                        filtered_data = filtered_data[filtered_data['actualdistvar'].isin(actualdistvar_filter)]


                                    filtered_data = filtered_data[
                                        (filtered_data['Adj.Rsq'] >= adjrsquare_filter[0]) & 
                                        (filtered_data['Adj.Rsq'] <= adjrsquare_filter[1])
                                    ]
                                    filtered_data = filtered_data[
                                        (filtered_data['CSF.CSF'] >= csf_filter[0]) & 
                                        (filtered_data['CSF.CSF'] <= csf_filter[1])
                                    ]
                                    filtered_data = filtered_data[
                                        (filtered_data['AIC'] >= aic_filter[0]) & 
                                        (filtered_data['AIC'] <= aic_filter[1])
                                    ]




                                #     filtered_data['selectedmodels'] = 'No'

                                #     # Function to update selectedmodels for each (Channel, Brand) group
                                #     def update_selectedmodels(group):
                                #         if not group.empty:
                                #             median_csf = group['CSF.CSF'].median()  # Find the median
                                #             closest_index = (group['CSF.CSF'] - median_csf).abs().idxmin()  # Get index closest to median
                                #             group.loc[:, 'selectedmodels'] = 'No'  # Reset all to 'No'
                                #             group.loc[closest_index, 'selectedmodels'] = 'Yes'  # Set the closest row as 'Yes'
                                #         return group

                                #     # Apply function to ensure only one 'Yes' per (Channel, Brand) combination
                                #     filtered_data = filtered_data.groupby(['Channel', L3_name_column], group_keys=False).apply(update_selectedmodels)

                                #     # Store the updated filtered_data in session state
                                #     st.session_state["filtered_data"] = filtered_data

                                #     # Update only the 'selectedmodels' column in L0 for rows marked as 'Yes' in filtered_data
                                #     if "L3" in st.session_state:
                                #         L3 = st.session_state["L3"]

                                #         for (channel, brand), group in filtered_data.groupby(['Channel', L3_name_column]):
                                #             # Reset all 'selectedmodels' to 'No' for this (Channel, Brand) in L3
                                #             L3.loc[(L3['Channel'] == channel) & (L3[L3_name_column] == unique_value), 'selectedmodels'] = 'No'

                                #             # Find the row in filtered_data where 'selectedmodels' is 'Yes'
                                #             selected_row = group[group['selectedmodels'] == 'Yes']

                                #             if not selected_row.empty:
                                #                 closest_index = selected_row.index[0]  # Get the first matching index

                                #                 # Update only the 'selectedmodels' column in L3
                                #                 L3.loc[closest_index, 'selectedmodels'] = 'Yes'

                                #     # Store the updated L3 in session state **before saving under file_name**
                                #         st.session_state["L3"] = L3.copy()

                                #         # Store updated data in session state with file name
                                #         file_name3 = st.session_state["L3file_name"]
                                #         st.session_state[file_name3] = L3.copy()










                                #     # Incorporate previously saved changes, if any
                                #     if f"filtered_data_{unique_value}" in st.session_state:
                                #         # Update the filtered L3 with previously saved changes
                                #         modified_rows = st.session_state[f"filtered_data_{unique_value}"]
                                #         filtered_data.update(modified_rows)

                                #     # Reset 'selectedmodels' column if not present
                                #     if "selectedmodels" not in filtered_data.columns:
                                #         filtered_data['selectedmodels'] = 'No'

                                #     # Calculate the default selection based on the median of CSF.CSF
                                #     if not filtered_data.empty:
                                #         median_csf_index = (
                                #             (filtered_data['CSF.CSF'] - filtered_data['CSF.CSF'].median())
                                #             .abs()
                                #             .idxmin()
                                #         )
                                #     else:
                                #         median_csf_index = None

                                #     # Display the table with a selectbox to choose a row
                                #     st.write("Select a row to choose as the selected method:")
                                #     selected_index = st.selectbox(
                                #         f"Select index for {unique_value}",
                                #         options=filtered_data.index,
                                #         index=filtered_data.index.get_loc(median_csf_index) if median_csf_index in filtered_data.index else 0,
                                #         format_func=lambda x: f"{x}"  # Optional: Add more clarity to selection
                                #     )

                                #     # Track the state for each brand selection
                                #     if unique_value not in st.session_state:
                                #         st.session_state[unique_value] = {'previous_states': [], 'last_selected_index': None}  # Initialize stack and last index

                                #     # If a new 'Yes' is selected, save the current state before the change
                                #     if selected_index != st.session_state[unique_value].get('last_selected_index', None):
                                #         # Save the current L3 before applying the new 'Yes'
                                #         st.session_state[unique_value]['previous_states'].append(st.session_state["L3"].copy())
                                #         st.session_state[unique_value]['last_selected_index'] = selected_index  # Update the last selected index

                                #     # Now handle the "Revert" button
                                #     # Create two columns to place the buttons side by side
                                #     col8, col9 = st.columns([5, 1])  # Adjust the column widths as needed

                                #     # First button: Save Selected Method
                                #     with col8:
                                #         if st.button(f"Save Selected Method for {unique_value}", key=f"save_selected_method_{unique_value}"):
                                #             # Check if there are multiple 'Yes' for the current brand under the global filters
                                #             global_brand_filter = (
                                #                 (st.session_state["L3"][L3_name_column] == unique_value) &
                                #                 # (st.session_state["L3"]['Market'].isin(selected_markets)) &
                                #                 # (st.session_state["L3"]['Category'].isin(selected_categories)) &
                                #                 (st.session_state["L3"]['Channel'].isin(selected_channels3))
                                #             )

                                #             # Reset if there is already a 'Yes' for the brand under the filters
                                #             if (st.session_state["L3"].loc[global_brand_filter, 'selectedmodels'] == 'Yes').sum() > 0:
                                #                 # Reset all rows for the current brand and filters to 'No'
                                #                 st.session_state["L3"].loc[global_brand_filter, 'selectedmodels'] = 'No'

                                #             # Set the selected row's 'selectedmodels' column to 'Yes'
                                #             st.session_state["L3"].loc[selected_index, 'selectedmodels'] = 'Yes'

                                #             # Update the filtered L3 for the current brand
                                #             filtered_data['selectedmodels'] = st.session_state["L3"].loc[
                                #                 filtered_data.index, 'selectedmodels'
                                #             ]

                                #             st.session_state["L3"].loc[filtered_data.index, 'selectedmodels'] = filtered_data['selectedmodels']
                                #             file_name3 = st.session_state["L3file_name"]
                                #             st.session_state[file_name3] = st.session_state["L3"].copy() 

                                #             st.success(f"Selected method for {brand} saved successfully!")

                                #     # Second button: Revert Selection
                                #     with col9:
                                #         if st.button(f"REVERT", key=f"revert_selected_method_{unique_value}"):
                                #             if st.session_state[unique_value]['previous_states']:
                                #                 # Pop the most recent state to revert to the previous selection
                                #                 last_state = st.session_state[unique_value]['previous_states'].pop()
                                #                 st.session_state["L3"] = last_state.copy()  # Revert to the last state

                                #                 # Update the filtered L3 for the current brand with reverted selections
                                #                 filtered_data['selectedmodels'] = st.session_state["L3"].loc[
                                #                     filtered_data.index, 'selectedmodels'
                                #                 ]

                                #                 st.session_state["L3"].loc[filtered_data.index, 'selectedmodels'] = filtered_data['selectedmodels']
                                #                 file_name3 = st.session_state["L3file_name"]
                                #                 st.session_state[file_name3] = st.session_state["L3"].copy()

                                #                 st.success(f"Reverted!")
                                #             else:
                                #                 st.warning("No previous state found to revert to!")
                                    
                                #     # filtered_data_reset = filtered_data.reset_index()

                                #     # Display the filtered data for this brand, with the updated 'selectedmodels' after save/revert
                                    st.dataframe(filtered_data[['Market', 'Category', 'Channel', 'MCV.MCV', 'CSF.CSF', 'method', 'selectedmodels', 'RPIto', 'Adj.Rsq', 'AIC','actualdistvar']])

                                # elif submit_option=="SUBMIT SELECTIONS":
                                #     st.success("TO SELECT MODELS, PLEASE CLICK ON 'MAKE SELECTIONS' BUTTON.")

                                st.markdown('<hr class="thin">', unsafe_allow_html=True)  # Add thin line after each brand

                            # Show the final modified data after changes
                            if st.button("Show Final Modified Data FOR L3"):
                                # # Combine the original data with any updated rows
                                # final_modified_data = st.session_state["data"].copy()
                                
                                file_name3 = st.session_state["L3file_name"]
                                original_columns = st.session_state["L3"].columns.tolist()
                                st.session_state[file_name3].columns = original_columns

                                # st.session_state[file_name] = final_modified_data

                                download_file_name3 = f"{file_name3}.csv" if not file_name3.endswith(".csv") else file_name3

                                # st.write(f"Final Modified Data (File Name: {file_name}):")
                                st.dataframe(st.session_state[file_name3])

                                st.download_button("Download the final modified data", 
                                        data=st.session_state[file_name3].to_csv(index=False), 
                                        file_name=download_file_name3, 
                                        mime="text/csv")

                            st.markdown('<hr class="thin">', unsafe_allow_html=True)






            with col2:


                if uploaded_file3:

                # st.markdown(
                #     """
                #     <style>
                #         /* Ensure AgGrid stays fixed in the viewport */
                #         .streamlit-expanderHeader {
                #             position: sticky;
                #             top: 330px;
                #             z-index: 1000;
                #         }
                #         /* More specific targeting for AgGrid to stay fixed */
                #         div[data-testid="stAgGrid"] {
                #             position: fixed;
                #             top: 330px;
                #             right: 1%;
                #             width: 40%;
                #             z-index: 1000;
                #         }
                #     </style>
                #     """,
                #     unsafe_allow_html=True
                # )


                # Custom CSS to make only the specific DataFrame sticky
                # Apply CSS to make ONLY this specific DataFrame sticky
                # Apply CSS to make ONLY this specific DataFrame sticky
                # Apply CSS to make ONLY this specific DataFrame sticky
                # st.markdown(
                #     """
                #     <style>
                #         /* Target only the DataFrame inside the sticky-container */
                #         [data-testid="stVerticalBlock"] > div[data-testid="column"] > div#sticky-container {
                #             position: fixed;
                #             top: 100px;  /* Adjust based on navbar */
                #             right: 20px;
                #             width: 30%;
                #             z-index: 1000;
                #             background: white;
                #             box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                #         }
                #     </style>
                #     """,
                #     unsafe_allow_html=True
                # )
                    # if "MSP_L0" not in st.session_state:
                    #         st.session_state["MSP_L0"] = None

                    if "MSP_L0L2" not in st.session_state:
                        st.session_state["MSP_L0L2"] = None


                
                    # # st.markdown('<hr class="thick">', unsafe_allow_html=True)

                    # options = ["CONFIGURATION"]
                    # option = st.pills(f"", options,selection_mode="single",default="CONFIGURATION",)

                    # if option == "CONFIGURATION":
                    #     # Define available options
                    #     modeling_type_options = [1, 2]
                    #     msp_options = ["Yes", "No"]
                    #     variable_options = ['NA', 'Market', 'Channel', 'Region', 'Category',
                    #                         'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                    #     periodicity_options = ["Daily", "Weekly", "Monthly"]

                    #     # Function to get a valid index
                    #     def get_valid_index(option_list, stored_value, default_value):
                    #         return option_list.index(stored_value) if stored_value in option_list else option_list.index(default_value)

                    #     # Initialize session state if not already set
                    #     if "config_data" not in st.session_state:
                    #         st.session_state.config_data = {
                    #             "Modeling Type": 2,
                    #             "MSP at L0 (Yes/No)": "Yes",
                    #             "Variable to use as L0": "Brand",
                    #             "MSP at L2 (Yes/No)": "No",
                    #             "Variable to use as L2": "NA",
                    #             "MSP at L3 (Yes/No)": "No",
                    #             "Variable to use as L3": "NA",
                    #             "Periodicity of Data (Daily/Weekly/Monthly)": "Weekly",
                    #             "Reference Period": 1
                    #         }

                    #     # Get valid indices for selectbox
                    #     modeling_type_index = get_valid_index(modeling_type_options, st.session_state.config_data["Modeling Type"], 2)
                    #     msp_l0_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L0 (Yes/No)"], "Yes")
                    #     msp_l2_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L2 (Yes/No)"], "No")
                    #     msp_l3_index = get_valid_index(msp_options, st.session_state.config_data["MSP at L3 (Yes/No)"], "No")
                    #     variable_l0_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L0"], "Brand")
                    #     variable_l2_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L2"], "NA")
                    #     variable_l3_index = get_valid_index(variable_options, st.session_state.config_data["Variable to use as L3"], "NA")
                    #     periodicity_index = get_valid_index(periodicity_options, st.session_state.config_data["Periodicity of Data (Daily/Weekly/Monthly)"], "Daily")

                    #     # Create a form for user input
                    #     with st.form("config_form"):
                    #         modeling_type = st.selectbox("Modeling Type", options=modeling_type_options, index=modeling_type_index)

                    #         col10, col11 = st.columns(2)
                    #         with col10:
                    #             msp_l0 = st.selectbox("MSP at L0 (Yes/No)", options=msp_options, index=msp_l0_index)
                    #             msp_l2 = st.selectbox("MSP at L2 (Yes/No)", options=msp_options, index=msp_l2_index)
                    #             msp_l3 = st.selectbox("MSP at L3 (Yes/No)", options=msp_options, index=msp_l3_index)
                    #             periodicity = st.selectbox("Periodicity of Data (Daily/Weekly/Monthly)", options=periodicity_options, index=periodicity_index)
                            
                    #         with col11:
                    #             variable_l0 = st.selectbox("Variable to use as L0", options=variable_options, index=variable_l0_index)
                    #             variable_l2 = st.selectbox("Variable to use as L2", options=variable_options, index=variable_l2_index)
                    #             variable_l3 = st.selectbox("Variable to use as L3", options=variable_options, index=variable_l3_index)
                    #             reference_period = st.number_input("Reference Period", min_value=1, value=st.session_state.config_data["Reference Period"])

                    #         # Submit button
                    #         submit_button = st.form_submit_button("Save Configuration")

                    #         # If submitted, update session state
                    #         if submit_button:
                    #             st.session_state.config_data = {
                    #                 "Modeling Type": modeling_type,
                    #                 "MSP at L0 (Yes/No)": msp_l0,
                    #                 "Variable to use as L0": variable_l0,
                    #                 "MSP at L2 (Yes/No)": msp_l2,
                    #                 "Variable to use as L2": variable_l2,
                    #                 "MSP at L3 (Yes/No)": msp_l3,
                    #                 "Variable to use as L3": variable_l3,
                    #                 "Periodicity of Data (Daily/Weekly/Monthly)": periodicity,
                    #                 "Reference Period": reference_period
                    #             }
                    #             st.write("Configuration saved successfully!")

                    #     # Display current configuration
                    #     config_df = pd.DataFrame([st.session_state.config_data])
                    #     st.write("Current Configuration:")
                    #     st.dataframe(config_df)
                            
                    # st.markdown('<hr class="thick">', unsafe_allow_html=True)

                    

                    # Fetch D0 file from session state
                    if "D0" in st.session_state["uploaded_files"]:
                        D0 = st.session_state["uploaded_files"]["D0"]

                    #     # # Rename date column if needed
                    #     # if 'date' in D0.columns:
                    #     #     D0.rename(columns={'date': 'Date'}, inplace=True)

                    #     # Filter out unwanted brands
                    #     if 'Brand' in D0.columns:
                    #         D0 = D0[~D0['Brand'].str.lower().isin(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])]

                    # else:
                    #     st.sidebar.warning("D0 file not found in session state.")

                    # Fetch Weighted Average Files from session state
                    weighted_files = [
                        "Wtd_avg_MCV_Brand", "Wtd_avg_MCV_PackType", "Wtd_avg_MCV_PackSize",
                        "Wtd_avg_MCV_PPG", "Wtd_avg_MCV_Variant", "Wtd_avg_MCV_Category",
                        "Wtd_avg_MCV_SubCategory"
                    ]

                    df_dict = {
                        "Brand": pd.DataFrame(), "PackType": pd.DataFrame(), "PackSize": pd.DataFrame(),
                        "PPG": pd.DataFrame(), "Variant": pd.DataFrame(), "Category": pd.DataFrame(),
                        "SubCategory": pd.DataFrame()
                    }

                    for file_name3 in weighted_files:
                        if file_name3 in st.session_state:
                            try:
                                df = st.session_state[file_name3]  # ✅ Directly use stored DataFrame
                                
                                # # Print columns for debugging
                                # st.write(f"Columns in {file_name3}:", df.columns.tolist())


                                # Check if 'selectedmodels' exists (lowercase)
                                if 'selectedmodels' in df.columns:
                                    df = df[df['selectedmodels'].str.lower() == 'yes']
                                    key = file_name3.replace("Wtd_avg_MCV_", "").replace(".xlsx", "")
                                    df_dict[key] = df  # ✅ Store filtered DataFrame
                                    # st.write(df)
                                    # if df_L0 is not None:
                                    #     st.session_state.df_L0 = df_L0


                                else:
                                    st.sidebar.warning(f"'{file_name3}' does not contain 'selectedmodels' column after processing.")

                            except Exception as e:
                                st.sidebar.error(f"Error processing {file_name3}: {e}")
                        # else:
                        #     st.sidebar.warning(f"{file_name2} not found in session state.")

                    # st.markdown('<hr class="thin">', unsafe_allow_html=True)


                #-----------------------------------------------------------------------------------------------------------------------------------------
                    # # Display processed datasets
                    # st.subheader("Processed D0 Data")
                    # if "D0" in st.session_state["uploaded_files"]:
                    #     st.dataframe(D0)
                    # else:
                    #     st.write("No D0 data available.")

                    # st.subheader("Processed Weighted Average Files")
                    # for key, df in df_dict.items():
                    #     st.write(f"### {key}")
                    #     if not df.empty:
                    #         st.dataframe(df)
                    #     else:
                    #         st.write("No data available.")


                # User Input for Time Period (Weekly data)

                #     if "config_data" in st.session_state:
                #         config_df=pd.DataFrame([st.session_state.config_data])

                #         # Check the conditions
                #         if (config_df['Periodicity of Data (Daily/Weekly/Monthly)']
                #             .astype(str)  # Ensure it's a string Series
                #             .str.strip()  # Remove extra spaces
                #             .str.lower()  # Convert to lowercase
                #             .eq('weekly') # Check for 'weekly'
                #             .any()):

                #             from datetime import timedelta
                #             import pandas as pd

                #             def defined_period_data(dataframe, weeks):
                #                 """
                #                 Filters the DataFrame to include rows from the most recent weeks based on the Date column.
                #                 Converts all dates to short format (YYYY-MM-DD) for consistency.
                #                 Assumes the data is strictly weekly.

                #                 Parameters:
                #                 - dataframe (pd.DataFrame): The input DataFrame with a date column.
                #                 - weeks (int): Number of weeks of data to retain.

                #                 Returns:
                #                 - pd.DataFrame: Filtered DataFrame with data from the specified period.
                #                 """
                #                 # Detect the date column (case-insensitive search for "date")
                #                 date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
                #                 if not date_column:
                #                     raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                                
                #                 # Function to handle multiple date formats and remove time if present
                #                 def parse_date(date):
                #                     for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
                #                                 '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):  # Supported formats
                #                         try:
                #                             # Convert to datetime
                #                             parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
                #                             if pd.notnull(parsed_date):
                #                                 return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
                #                         except ValueError:
                #                             continue
                #                     raise ValueError(f"Date '{date}' does not match any supported formats.")
                                
                #                 # Apply parsing and conversion to short format to the date column
                #                 dataframe[date_column] = dataframe[date_column].apply(parse_date)
                                
                #                 # Convert the date column back to datetime for further processing
                #                 dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                                
                #                 # Sort the DataFrame by date
                #                 dataframe = dataframe.sort_values(by=date_column)
                                
                #                 # Calculate cutoff date based on weeks
                #                 recent_date = dataframe[date_column].max()
                #                 cutoff_date = recent_date - timedelta(weeks=int(weeks))
                                
                #                 # Filter the DataFrame
                #                 filtered_df = dataframe[dataframe[date_column] > cutoff_date]
                                
                #                 return filtered_df


                #             # Example Usage:
                #             Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

                #         else:
                #             print("Conditions not met. Skipping execution.")




                # #--------------------------------------------------------

                # # User Input for Time Period (Monthly data)

                #     if "config_data" in st.session_state:
                #         config_df=pd.DataFrame([st.session_state.config_data])
                #         # Check the conditions
                #         if (config_df['Periodicity of Data (Daily/Weekly/Monthly)'].str.strip().str.lower().eq('monthly').any()):

                #             from datetime import timedelta
                #             import pandas as pd

                #             def defined_period_data(dataframe, months):
                #                 """
                #                 Filters the DataFrame to include rows from the most recent months based on the Date column.
                #                 Converts all dates to short format (YYYY-MM-DD) for consistency.
                #                 Assumes the data is strictly monthly.

                #                 Parameters:
                #                 - dataframe (pd.DataFrame): The input DataFrame with a date column.
                #                 - months (int): Number of months of data to retain.

                #                 Returns:
                #                 - pd.DataFrame: Filtered DataFrame with data from the specified period.
                #                 """
                #                 # Detect the date column (case-insensitive search for "date")
                #                 date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
                #                 if not date_column:
                #                     raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                                
                #                 # Function to handle multiple date formats and remove time if present
                #                 def parse_date(date):
                #                     for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
                #                                 '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):  # Supported formats
                #                         try:
                #                             # Convert to datetime
                #                             parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
                #                             if pd.notnull(parsed_date):
                #                                 return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
                #                         except ValueError:
                #                             continue
                #                     raise ValueError(f"Date '{date}' does not match any supported formats.")
                                
                #                 # Apply parsing and conversion to short format to the date column
                #                 dataframe[date_column] = dataframe[date_column].apply(parse_date)
                                
                #                 # Convert the date column back to datetime for further processing
                #                 dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                                
                #                 # Sort the DataFrame by date
                #                 dataframe = dataframe.sort_values(by=date_column)
                                
                #                 # Convert the date column to periods (monthly)
                #                 dataframe['Month_Sorting'] = dataframe[date_column].dt.to_period('M')
                                
                #                 # Calculate the cutoff month
                #                 recent_month = dataframe['Month_Sorting'].max()
                #                 cutoff_month = recent_month - months
                                
                #                 # Filter the DataFrame based on the cutoff month
                #                 filtered_df = dataframe[dataframe['Month_Sorting'] > cutoff_month]
                                
                #                 # Drop the temporary 'Month' column
                #                 filtered_df = filtered_df.drop(columns=['Month_Sorting'], errors='ignore')
                                
                #                 return filtered_df


                #             # Example Usage:
                #             Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

                #         else:
                #             print("Conditions not met. Skipping execution.")

                # #---------------------------------------------------------------------------

                # # User Input for Time Period (Daily data)

                #     if "config_data" in st.session_state:
                #         config_df=pd.DataFrame([st.session_state.config_data])

                #         # Check the conditions
                #         if (config_df['Periodicity of Data (Daily/Weekly/Monthly)'].str.strip().str.lower().eq('daily').any()):

                #             from datetime import timedelta
                #             import pandas as pd

                #             def defined_period_data(dataframe, days):
                #                 """
                #                 Filters the DataFrame to include rows from the most recent days based on the Date column.
                #                 Converts all dates to short format (YYYY-MM-DD) for consistency.
                #                 Assumes the data is strictly daily.

                #                 Parameters:
                #                 - dataframe (pd.DataFrame): The input DataFrame with a date column.
                #                 - days (int): Number of days of data to retain.

                #                 Returns:
                #                 - pd.DataFrame: Filtered DataFrame with data from the specified period.
                #                 """
                #                 # Detect the date column (case-insensitive search for "date")
                #                 date_column = next((col for col in dataframe.columns if col.lower() == 'date'), None)
                #                 if not date_column:
                #                     raise ValueError("The DataFrame must have a 'date' or 'Date' column.")
                                
                #                 # Function to handle multiple date formats and remove time if present
                #                 def parse_date(date):
                #                     for fmt in ('%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M' ,'%d-%m-%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M','%Y-%m-%d',  '%m-%d-%Y %H:%M:%S', '%m-%d-%Y %H:%M' ,'%m-%d-%Y',
                #                                 '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M' ,'%d/%m/%Y', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M','%Y/%m/%d',  '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M','%m/%d/%Y'):   # Supported formats
                #                         try:
                #                             # Convert to datetime
                #                             parsed_date = pd.to_datetime(date, format=fmt, errors='coerce')
                #                             if pd.notnull(parsed_date):
                #                                 return parsed_date.strftime('%Y-%m-%d')  # Return short format (YYYY-MM-DD)
                #                         except ValueError:
                #                             continue
                #                     raise ValueError(f"Date '{date}' does not match any supported formats.")
                                
                #                 # Apply parsing and conversion to short format to the date column
                #                 dataframe[date_column] = dataframe[date_column].apply(parse_date)
                                
                #                 # Convert the date column back to datetime for further processing
                #                 dataframe[date_column] = pd.to_datetime(dataframe[date_column])
                                
                #                 # Sort the DataFrame by date
                #                 dataframe = dataframe.sort_values(by=date_column)
                                
                #                 # Calculate the cutoff date based on days
                #                 recent_date = dataframe[date_column].max()
                #                 cutoff_date = recent_date - timedelta(days=days)
                                
                #                 # Filter the DataFrame based on the cutoff date
                #                 filtered_df = dataframe[dataframe[date_column] > cutoff_date]
                                
                #                 return filtered_df 


                #             # Example Usage:
                #             Reference_Period = int(config_df.loc[0, 'Reference Period'])  # Ensure this is an integer

                #         else:
                #             print("Conditions not met. Skipping execution.")


                # #---------------------------------------------------------------------------

                # # # Taking User Input for Brand Names

                # #     if "config_data" in st.session_state:
                # #         config_df=pd.DataFrame([st.session_state.config_data])
                # #         # Check the conditions
                # #         if (config_df['Modeling Type'].eq(1).any() and 
                # #             config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                # #             config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):
                            
                # #             # Take user input as a single string
                # #             user_input = input("Enter Elements of Variable which is used as L0, separated by commas: ")

                # #             # Split the input string into a list and convert each item to lowercase
                # #             brand_names = [name.strip().lower() for name in user_input.split(',')]

                # #             print(brand_names)
                # #         else:
                # #             print("Conditions not met. Skipping execution.")



                # #--------------------------------------------------------------

                # # User Input for L0 Variable

                    def select_defined_dataframe(df_dict, selected_key):
                        """
                        Selects a single DataFrame from a dictionary based on user input,
                        ignoring those that are not defined.

                        Args:
                            df_dict (dict): A dictionary where keys are DataFrame names (strings)
                                            and values are the DataFrame objects.
                            selected_key (str): The name of the DataFrame to select.


                        Returns:
                            pd.DataFrame: The selected DataFrame, or None if the selection is invalid.
                        """
                        # Filter the dictionary to include only defined DataFrames
                        defined_dfs = {key: df for key, df in df_dict.items() if isinstance(df, pd.DataFrame)}

                        if not defined_dfs:
                            print("No DataFrames are defined!")
                            return None

                        print("Available DataFrames:")
                        for key in defined_dfs.keys():
                            print(f"- {key}")

                        # Validate the input key
                        if selected_key in defined_dfs:
                            print(f"\nSelected DataFrame: {selected_key}")
                            return defined_dfs[selected_key]
                        else:
                            print("Invalid selection! Please try again.")
                            return None

                    # if "df_L0" not in st.session_state:
                    #     st.session_state.df_L0 = None

                    # if "config_data" in st.session_state:
                    #     config_df=pd.DataFrame([st.session_state.config_data])

                    #     if config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any():

                    #         # Get user input outside the function
                    #         selected_key_L0 = config_df.loc[0, 'Variable to use as L0']
                    #         st.session_state.selected_key_L0 = selected_key_L0

                    #         # Call the function with the user input
                    #         df_L0 = select_defined_dataframe(df_dict, selected_key_L0)
                            
                    #         if df_L0 is not None:
                    #             st.session_state.df_L0 = df_L0
                    #             print("\nHere is the selected DataFrame:")
                    #             print(df_L0)
                    #         else:
                    #             st.error("df_L0 not found in session state!")
                            

                    #     else:
                    #         print("The value in 'MSP at L0 (Yes/No)' is not 'Yes'. Skipping execution.")




                #---------------------------------------------------------------

                # User Input for L2 Variable

                    # if "df_L0" not in st.session_state:
                    #     st.session_state.df_L0 = None


                    # if "df_L2" not in st.session_state:
                    #     st.session_state.df_L2 = None

                    # if "config_data" in st.session_state:
                    #     config_df=pd.DataFrame([st.session_state.config_data])
                    #     if (config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and 
                    #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                    #         # Get user input outside the function
                    #         selected_key_L2 = config_df.loc[0, 'Variable to use as L2']
                    #         st.session_state.selected_key_L2 = selected_key_L2
                    #         # st.write(selected_key_L2)

                    #         # Call the function with the user input
                    #         df_L2 = select_defined_dataframe(df_dict, selected_key_L2)
                    #         # st.write(df_L2)

                    #         if df_L2 is not None:
                    #             st.session_state.df_L2 = df_L2
                    #             print("\nHere is the selected DataFrame:")
                    #             # st.write(df_L2)
                                
                    #     else:
                    #         print("The value in 'MSP at L2 (Yes/No)' is not 'Yes'. Skipping execution.")



                #------------------------------------------------------


                    if "df_L3" not in st.session_state:
                        st.session_state.df_L3 = None

                    if "config_data" in st.session_state:
                        config_df=pd.DataFrame([st.session_state.config_data])
                        if (config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and 
                            config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                            config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                            # Get user input outside the function
                            selected_key_L3 = config_df.loc[0, 'Variable to use as L3']
                            st.session_state.selected_key_L3 = selected_key_L3

                            # Call the function with the user input
                            df_L3 = select_defined_dataframe(df_dict, selected_key_L3)
                            

                            if df_L3 is not None:
                                st.session_state.df_L3 = df_L3
                                print("\nHere is the selected DataFrame:")
                                # st.write(df_L3)

                        else:
                            print("The value in 'MSP at L3 (Yes/No)' is not 'Yes'. Skipping execution.")



                #---------------------------------------------------------------

                # Selecting desired Volume Units Column from the DF

                    def adjust_volume_based_on_other_df(original_df, other_df):
                        """
                        Adjusts the 'Volume' or 'VolumeUnits' columns in the original DataFrame based on the 'Vol_Var' column in another DataFrame.

                        Parameters:
                            original_df (pd.DataFrame): The original DataFrame containing 'Volume' and 'VolumeUnits' columns.
                            other_df (pd.DataFrame): The other DataFrame containing the 'Vol_Var' column.

                        Returns:
                            pd.DataFrame: The adjusted original DataFrame.

                        Raises:
                            ValueError: If 'Vol_Var' is missing in other_df or contains invalid/multiple unique values.
                        """
                        # Ensure 'Vol_Var' column exists in the other DataFrame
                        if 'Vol_Var' not in other_df.columns:
                            raise ValueError("The other DataFrame must contain a 'Vol_Var' column.")
                        
                        # Get unique values in 'Vol_Var'
                        vol_var_values = other_df['Vol_Var'].unique()
                        
                        # Ensure 'Vol_Var' contains only one unique value
                        if len(vol_var_values) != 1:
                            raise ValueError("The 'Vol_Var' column must contain a single unique value across the DataFrame.")
                        
                        vol_var_value = vol_var_values[0]
                        
                        # Adjust original_df based on 'Vol_Var' value
                        if vol_var_value == 'Volume':
                            # Drop 'VolumeUnits' and rename 'Volume' in the original DataFrame
                            if 'VolumeUnits' in original_df.columns:
                                original_df = original_df.drop(columns=['VolumeUnits'])
                            original_df = original_df.rename(columns={'Volume': 'Volume'})
                        
                        elif vol_var_value == 'VolumeUnits':
                            # Drop 'Volume' and rename 'VolumeUnits' in the original DataFrame
                            if 'Volume' in original_df.columns:
                                original_df = original_df.drop(columns=['Volume'])
                            original_df = original_df.rename(columns={'VolumeUnits': 'Volume'})
                        
                        else:
                            raise ValueError("Invalid value in 'Vol_Var' column. Expected 'Volume' or 'VolumeUnits'.")

                        return original_df




                    def add_vol_var_column(target_df, reference_df, column_name):
                        """
                        Adds a new column to the target DataFrame based on the unique value in the reference DataFrame's Vol_Var column.
                        
                        Parameters:
                        - target_df (pd.DataFrame): The target DataFrame where the column will be added.
                        - reference_df (pd.DataFrame): The reference DataFrame containing the Vol_Var column.
                        - column_name (str): The name of the column to be added to the target DataFrame.
                        
                        Returns:
                        - pd.DataFrame: The updated target DataFrame with the new column added.
                        """
                        # Ensure the Vol_Var column exists in the reference DataFrame
                        if "Vol_Var" not in reference_df.columns:
                            raise ValueError("The reference DataFrame must contain the 'Vol_Var' column.")
                        
                        # Check if all values in the Vol_Var column are the same
                        unique_values = reference_df["Vol_Var"].unique()
                        if len(unique_values) != 1:
                            raise ValueError("The 'Vol_Var' column in the reference DataFrame must have a single unique value.")
                        
                        # Get the unique value
                        vol_var_value = unique_values[0]
                        
                        # Add the new column to the target DataFrame with the same value
                        target_df[column_name] = vol_var_value
                        
                        return target_df









                    D0['Channel'] = D0['Channel'].str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.replace("'", '', regex=False).str.lower()
                    D0['Brand'] = D0['Brand'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['Market'] = D0['Market'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['PPG'] = D0['PPG'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['Variant'] = D0['Variant'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['Category'] = D0['Category'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['PackType'] = D0['PackType'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['PackSize'] = D0['PackSize'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()
                    D0['SubCategory'] = D0['SubCategory'].str.replace("'", '', regex=False).str.replace('.', '', regex=False).str.replace('&', 'and', regex=False).str.lower()




                    # if 'df_L0' in st.session_state:
                    #     df_L0 = st.session_state.df_L0
                    #     D0 = adjust_volume_based_on_other_df(D0, df_L0)
                
                    # else:
                    #     st.error("df_L0 not found in session state!")

                    # D0_L0 = D0.copy()
                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0
                    #     D0_L0[selected_key_L0] = D0_L0[selected_key_L0].astype('str')





                    # # Drop columns dynamically
                    # D0_L0 = D0_L0.drop(columns=[
                    #         'Year', 'Month', 'Week', 'BrCatId'
                    #     ])
                    


                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0
                    # # Group the DataFrame
                    #     D0_L0 = (
                    #         D0_L0.groupby(['Market', 'Channel','Category',selected_key_L0, 'date'], as_index=False)
                    #         .agg({
                    #             'SalesValue': 'sum',  # Sum of SalesValue
                    #             'Volume': 'sum',      # Sum of Volume 
                    #         })
                    #     )



                    # if 'df_L0' in st.session_state and 'selected_key_L0' in st.session_state:
                    #     df_L0 = st.session_state.df_L0
                    #     selected_key_L0 = st.session_state.selected_key_L0

                    #     columns_to_keep = ['Market', 'Channel', selected_key_L0, 'MCV.MCV']  # Replace with your actual column names
                    #     df_L0_L0 = df_L0[columns_to_keep]
                    #     df_L0_L0[selected_key_L0] = df_L0_L0[selected_key_L0].astype('str')
                    # else:
                    #     st.error("Either 'df_L0' or 'selected_key_L0' is not in session_state.")


                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0

                    #     D0_L0['Revenue_Total'] = D0_L0.groupby(['Market', 'Channel', selected_key_L0])['SalesValue'].transform('sum')
                    #     D0_L0['Volume_Total'] = D0_L0.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')
                    #     D0_L0['Price_ModelPeriod'] = D0_L0['Revenue_Total']/D0_L0['Volume_Total'] 




                    # D0_filtered = defined_period_data(D0_L0, Reference_Period)



                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0

                    #     D0_filtered['Revenue'] = D0_filtered.groupby(['Market', 'Channel', selected_key_L0])['SalesValue'].transform('sum')
                    #     D0_filtered['Volume_definedperiod'] = D0_filtered.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')

                    #     D0_filtered['Price'] = D0_filtered['Revenue']/D0_filtered['Volume_definedperiod'] 

                    #     D0_filtered['Avg Volume'] = D0_filtered['Volume_definedperiod']/Reference_Period

                    #     D0_filtered = D0_filtered.drop(columns=['date', 'SalesValue', 'Volume'])

                    #     D0_filtered = D0_filtered.drop_duplicates(subset=['Market','Channel',selected_key_L0], keep='first')

                    #     MSP_L0 = pd.merge(df_L0_L0,D0_filtered, on = ['Market','Channel',selected_key_L0], how='left')
                


                    # MSP_L0['Total_Sales_Per_Channel'] = MSP_L0.groupby(['Market','Channel'])['Revenue'].transform('sum')
                    # MSP_L0['MShare'] = MSP_L0['Revenue']/MSP_L0['Total_Sales_Per_Channel']

                    # MSP_L0.rename(columns={'Volume_definedperiod': 'Volume'}, inplace=True)

                    # MSP_L0['CS'] = 0.5 * (MSP_L0['MCV.MCV'] + MSP_L0['Price']) * MSP_L0['Avg Volume']

                    # MSP_L0.groupby(['Market','Channel'])['MShare'].sum()

                    # MSP_L0['Sum_CS_by_Channel'] = MSP_L0.groupby(['Market','Channel'])['CS'].transform('sum')
                    # MSP_L0['Sum_MShare_by_Channel'] = MSP_L0.groupby(['Market','Channel'])['MShare'].transform('sum')

                    # # Step 2: Calculate PVS for each row
                    # MSP_L0['PVS'] = (MSP_L0['CS'] / MSP_L0['Sum_CS_by_Channel']) * MSP_L0['Sum_MShare_by_Channel']

                    # MSP_L0['MSP'] = MSP_L0['PVS'] - MSP_L0['MShare']

                    # MSP_L0 = MSP_L0.drop(columns=['Sum_CS_by_Channel','Sum_MShare_by_Channel','PVS'])

                    # MSP_L0['CSF'] = MSP_L0['MCV.MCV'] / MSP_L0['Price']

                    # MSP_L0['NewMShare'] = MSP_L0['MShare'] + MSP_L0['MSP']

                    # MSP_L0['Price_elas'] = MSP_L0['Price_ModelPeriod'] /(MSP_L0['Price'] - MSP_L0['MCV.MCV'])

                    # MSP_L0.rename(columns={'MCV.MCV': 'MCV'}, inplace=True)

                    # MSP_L0['Market'] = MSP_L0['Market'].str.lower()
                    # MSP_L0['Category'] = MSP_L0['Category'].str.lower()


                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0
                    #     columns_to_drop = [
                    #     col for col in ['Region', 'Brand', 'SubCategory', 'Variant', 'PackType', 'PPG', 'PackSize']
                    #     if col != selected_key_L0
                    #     ]

                    #     # Drop the remaining columns
                    #     MSP_L0 = MSP_L0.drop(columns=columns_to_drop, errors='ignore')


                    # for file_name in weighted_files:
                    #     if file_name in st.session_state:
                    #         try:
                    #             L0_Elasticity = st.session_state[file_name]

                    #         except Exception as e:
                    #             st.sidebar.error(f"Error processing {file_name}: {e}")


                    # result_dfs = []


                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0
                    #     all_columns = list(L0_Elasticity.columns)

                    #     # Ensure 'Category_elas' and 'beta0' exist
                    #     if 'Category_elas' in all_columns and 'beta0' in all_columns:
                    #         # Find the positions of 'Category_elas' and 'beta0'
                    #         start_index = all_columns.index('Category_elas')
                    #         end_index = all_columns.index('beta0')

                    #         # Get the columns that lie between 'Category_elas' and 'beta0', excluding them
                    #         columns_in_range = all_columns[start_index + 1:end_index]
                    #     else:
                    #         raise ValueError("'Category_elas' or 'beta0' column not found in the DataFrame.")

                    #     # Define conditions (columns to include in the analysis)
                    #     conditions = ['Market', 'Channel', selected_key_L0]
                    #     valid_conditions = [cond for cond in conditions if cond in L0_Elasticity.columns]
                    #     relevant_columns = valid_conditions + columns_in_range

                    #     # Process each unique channel
                    #     for channel in L0_Elasticity['Channel'].unique():
                    #         # Filter rows for the current channel
                    #         channel_df = L0_Elasticity[L0_Elasticity['Channel'] == channel]

                    #         # Process each unique brand within the current channel
                    #         for element in channel_df[selected_key_L0].unique():
                    #             # Filter rows for the current brand within the channel
                    #             element_df = channel_df[channel_df[selected_key_L0] == element].copy()

                    #             # # Apply multiple conditions
                    #             # element_df = element_df[
                    #             #     (element_df['method'].isin(methods_to_include)) &
                    #             #     (element_df['Price_pval'] == 'Yes') &
                    #             #     (element_df['Distribution_beta'] > 0) &
                    #             #     (element_df['Price_beta'] < 0)
                    #             # ]

                    #             # Ensure only relevant columns are included
                    #             element_df = element_df[[col for col in relevant_columns if col in element_df.columns]]

                    #             # Calculate the mean for the relevant columns in columns_in_range
                    #             for col in columns_in_range:
                    #                 if col in element_df.columns:
                    #                     # Calculate mean only for values greater than zero
                    #                     element_df[col] = element_df[col][element_df[col] > 0].mean()

                    #             # Append the processed data to the result list
                    #             result_dfs.append(element_df)

                    #     # Concatenate all processed DataFrames into a single DataFrame
                    #     result_df = pd.concat(result_dfs, ignore_index=True)

                    #     # Drop duplicates to retain one row per unique Market, Channel, and Brand
                    #     result_df = result_df.drop_duplicates(subset=['Market', 'Channel', selected_key_L0], keep='first')

                    

                    # MSP_L0 = pd.merge(MSP_L0,result_df, on = ['Market','Channel',selected_key_L0], how='left')
                    # MSP_L0 = MSP_L0.dropna(subset=['CSF'])

                    # MSP_L0.fillna('NA', inplace=True)



                    # if 'selected_key_L0' in st.session_state:
                    #     selected_key_L0 = st.session_state.selected_key_L0
                    #     # Dynamic column ordering
                    #     unique_elements = [element for element in L0_Elasticity[selected_key_L0].unique()]

                    #     #fixed_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand','Variant', 'PackType', 'PPG','MCV','Price_elas','Revenue', 'Volume','Price','CSF']  # Columns to place at the beginning
                    #     fixed_columns = ['Market', 'Channel', 'Category', selected_key_L0, 'MCV','Price_elas','Revenue', 'Volume','Price','CSF']  # Columns to place at the beginning
                    #     dot_columns = sorted([
                    #         col for col in MSP_L0.columns 
                    #         if '.' in col 
                    #         #or col.startswith('restofcategory') 
                    #         or col in unique_elements
                    #     ])  # Columns with '.' in alphabetical order
                    #     remaining_columns = ['Revenue_Total','Volume_Total', 'Price_ModelPeriod','CS','MShare','NewMShare','MSP']  # Columns to place at the end

                    #     # Combine the desired order
                    #     new_order = fixed_columns + dot_columns + remaining_columns

                    #     # Reorder the DataFrame
                    #     MSP_L0 = MSP_L0[new_order]

                    #     # Assume `final_df` is the target DataFrame and `reference_df` is the other DataFrame.
                    #     column_name_to_add = "Vol_Var"  # Name of the column to create in the final DataFrame
                    #     MSP_L0 = add_vol_var_column(MSP_L0, df_L0, column_name_to_add)


            ## MSP_L0 for Type 2----------------------------------

                    # if "config_data" in st.session_state:
                    #     config_df=pd.DataFrame([st.session_state.config_data])
                    #     if (config_df['Modeling Type'].eq(2).any() and 
                    #         config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any()):

                    #         MSP_L0_T2 = MSP_L0.copy()
                    #         MSP_L0_T2['CsfPeriod'] = Reference_Period
                    #         MSP_L0_T2 = MSP_L0

                    #         st.session_state["MSP_L0"]=MSP_L0


                        
                    #         # st.write('MSP_L0')

                    #         # st.dataframe(MSP_L0)  # Keep height limited



            # MSP_L0 fro Type 1 w/o MSP_L0L2------------------------------------------

                    # if "config_data" in st.session_state:
                    #     config_df=pd.DataFrame([st.session_state.config_data])
                    #     if (config_df['Modeling Type'].eq(1).any() and 
                    #         config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                    #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('no').any() and config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('no').any()) :

                    #         MSP_L0_T1_direct = MSP_L0.copy()
                    #         MSP_L0_T1_direct['CsfPeriod'] = Reference_Period
                    #         MSP_L0_T1_direct= MSP_L0



                    #         st.session_state["MSP_L0"]=MSP_L0


                    # st.write('MSP L0 FILE:')
                    # st.dataframe(MSP_L0)


                    # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        





                    # if "MSP_L0" in st.session_state and st.session_state["MSP_L0"] is not None:
                    #     # Convert DataFrame to HTML with preserved index
                    #     table_html = st.session_state["MSP_L0"].to_html(index=True)

                    #     # Sticky and Scrollable Table
                    #     st.markdown(
                    #         """
                    #         <style>
                    #             .scrollable-table-container {
                    #                 position: fixed;
                    #                 top: 330px;
                    #                 right: 1%;
                    #                 width: 40%;
                    #                 height: 400px;
                    #                 overflow-x: auto !important;
                    #                 overflow-y: auto !important;
                    #                 z-index: 1000;
                    #                 background: white;
                    #                 border: 1px solid #ddd;
                    #             }

                    #             .scrollable-table-container table {
                    #                 width: auto; /* Adjust table width dynamically */
                    #                 border-collapse: collapse;
                    #             }

                    #             .scrollable-table-container th, 
                    #             .scrollable-table-container td {
                    #                 border: 1px solid #ddd;
                    #                 padding: 8px;
                    #                 text-align: left;
                    #                 white-space: nowrap;
                    #             }
                    #         </style>
                    #         """,
                    #         unsafe_allow_html=True
                    #     )

                    #     # Display inside a scrollable div
                    #     st.markdown(f'<div class="scrollable-table-container">{table_html}</div>', unsafe_allow_html=True)
                    # else:
                    #     st.warning("MSP_L0 data is not available. Please upload or initialize the data.")

        #----------------------------------------------------------------------------------

                    # MSP_L0L2 T2


                    # if "config_data" in st.session_state:
                    #     config_df=pd.DataFrame([st.session_state.config_data])
                    
                    #     if (config_df['Modeling Type'].eq(2).any() and config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                    #         config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any()):


                    #         if 'df_L2' in st.session_state and 'selected_key_L2' in st.session_state and 'df_L0' in st.session_state:
                    #             df_L2 = st.session_state.df_L2

                    #             df_L0 = st.session_state.df_L0
                    #             selected_key_L2 = st.session_state.selected_key_L2
                    #             selected_key_L0 = st.session_state.selected_key_L0


                                

                    #             # df_L2['Channel'] = df_L2['Channel'].str.strip()
                    #             # df_L2[selected_key_L2] = df_L2[selected_key_L2].str.strip()
                    #             # df_L2['Market'] = df_L2['Market'].str.strip()

                    #             # df_L2
                    #             # df_L2.columns
                    #             # Now, select specific columns from the DataFrame
                    #             columns_to_keep = ['Market','Channel', 'MCV.MCV']  # Replace with your actual column names
                    #             df_L2_L0L2 = df_L2[columns_to_keep + [selected_key_L2]]
                    #             df_L2_L0L2[selected_key_L2] = df_L2_L0L2[selected_key_L2].astype('str')
                                
                    #             # #Check if all values in the 'Channel' column are the same
                    #             # if df_L2_L0L2['Channel'].nunique() == 1:  # If there is only one unique value
                    #             #     df_L2_L0L2['Channel'] = 'all'

                    #             # df_L2_L0L2.columns
                    #             # Now, select specific columns from the DataFrame

                                

                    #             columns_to_keep = ['Market','Channel','Category', selected_key_L0, 'MCV.MCV']  # Replace with your actual column names
                    #             df_L0_L2 = df_L0[columns_to_keep]

                    #             # st.write(df_L0)

                    #             df_L0_L2[selected_key_L0] = df_L0_L2[selected_key_L0].astype('str')

                    #             D0_L2 = D0.copy()
                                

                    #             # st.write(D0_L2)
                    #             D0_L2[selected_key_L0] = D0_L2[selected_key_L0].astype('str')
                    #             D0_L2[selected_key_L2] = D0_L2[selected_key_L2].astype('str')
                                
                    #             # Drop columns dynamically
                    #             D0_L2 = D0_L2.drop(columns=[
                    #                     'Year', 'Month', 'Week', 'BrCatId'
                    #                 ])
                                
                    #                 # Group the DataFrame
                    #             D0_L2 = (
                    #                 D0_L2.groupby(['Market', 'Channel','Category', selected_key_L0,selected_key_L2, 'date'], as_index=False)
                    #                 .agg({
                    #                     'SalesValue': 'sum',  # Sum of SalesValue
                    #                     'Volume': 'sum',      # Sum of Volume 
                    #                 })
                    #             )

                                
                                
                    #             # D0_L2
                    #             D0_L2_grouped_SalesValue = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['SalesValue'].sum().reset_index()
                    #             D0_L2_grouped_Volume = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].sum().reset_index()
                    #             D0_L2_grouped = pd.merge(D0_L2_grouped_SalesValue,D0_L2_grouped_Volume,on=['Market', 'Channel', selected_key_L0,selected_key_L2],how='left')

                                
                    #             # D0_L2_grouped
                    #             D0_L2_grouped = pd.merge(D0_L2_grouped,df_L0_L2,on=['Market', 'Channel', selected_key_L0],how='left')
                    #             # D0_L2_grouped


                    #             D0_L2_grouped = D0_L2_grouped.dropna(subset=['MCV.MCV'])

                    #             D0_L2_grouped.rename(columns={'MCV.MCV': 'MCV_L0'}, inplace=True)
                    #             D0_L2_grouped = pd.merge(D0_L2_grouped,df_L2_L0L2,on=['Market', 'Channel',selected_key_L2],how='left')

                    #             D0_L2_grouped = D0_L2_grouped.dropna(subset=['MCV.MCV'])

                    #             D0_L2_grouped.rename(columns={'MCV.MCV': 'MCV_L2'}, inplace=True)
                    #             D0_L2_grouped['Total_Volume_ByChannel'] = D0_L2_grouped.groupby(['Market', 'Channel', selected_key_L0])['Volume'].transform('sum')
                    #             # Calculate the product of MCV_PPG and Volume
                    #             D0_L2_grouped['Sum_Product_ByChannel'] = (
                    #                 D0_L2_grouped['MCV_L2'] * D0_L2_grouped['Volume']
                    #             ).groupby([D0_L2_grouped['Market'], D0_L2_grouped['Channel'], D0_L2_grouped[selected_key_L0]]).transform('sum')
                    #             # D0_L2_grouped
                    #             D0_L2_grouped['Wtd_MCV_ByBrand_using_L2MCV'] = D0_L2_grouped['Sum_Product_ByChannel'] / D0_L2_grouped['Total_Volume_ByChannel']
                    #             D0_L2_grouped['Correction_Factor'] = D0_L2_grouped['MCV_L0'] / D0_L2_grouped['Wtd_MCV_ByBrand_using_L2MCV']
                    #             D0_L2_grouped['MCV'] = D0_L2_grouped['Correction_Factor'] * D0_L2_grouped['MCV_L2']
                    #             # D0_L2_grouped
                    #             # Calculating Price for Whole Period Data
                    #             D0_L2['Revenue_Total'] = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['SalesValue'].transform('sum')
                    #             D0_L2['Volume_Total'] = D0_L2.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].transform('sum')
                    #             D0_L2['Price_ModelPeriod'] = D0_L2['Revenue_Total']/D0_L2['Volume_Total']
                    #             # Calculating Revenue, Volume and Price for defined time Period
                    #             D0_L2_filtered = defined_period_data(D0_L2, Reference_Period)
                    #             D0_L2_filtered['Revenue'] = D0_L2_filtered.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['SalesValue'].transform('sum')
                    #             D0_L2_filtered['Volume_definedperiod'] = D0_L2_filtered.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].transform('sum')
                    #             D0_L2_filtered['Price'] = D0_L2_filtered['Revenue']/D0_L2_filtered['Volume_definedperiod'] 
                    #             # D0_L2_filtered

                    #             D0_L2_filtered['Avg Volume'] = D0_L2_filtered['Volume_definedperiod']/Reference_Period


                    #             # Drop columns dynamically
                    #             D0_L2_filtered = D0_L2_filtered.drop(columns=['date', 'SalesValue', 'Volume'])


                    #             # Create a DataFrame with unique brands, keeping the first occurrence of other columns
                    #             D0_L2_filtered = D0_L2_filtered.drop_duplicates(subset=['Market', 'Channel', selected_key_L0,selected_key_L2], keep='first')
                    #             # D0_L2_filtered
                    #             # D0_L2_filtered.shape
                    #             # List of columns to keep
                    #             columns_to_keep = ['Market', 'Channel', selected_key_L0,selected_key_L2,'MCV']

                    #             # Filter the DataFrame to include only these columns
                    #             D0_L2_grouped = D0_L2_grouped[columns_to_keep]
                    #             # D0_L2_grouped.shape
                    #             # D0_L2_grouped
                    #             # df_L2_L0L2.shape
                    #             # df_L2_L0L2
                    #             df_L2_L0L2 = df_L2_L0L2.drop(columns='MCV.MCV')



                    #             # df_L2_L0L2.columns
                    #             MSP_L2 = pd.merge(df_L2_L0L2,D0_L2_grouped, on = ['Market', 'Channel',selected_key_L2], how='left')
                    #             # MSP_L2
                    #             MSP_L0L2 = pd.merge(MSP_L2,D0_L2_filtered, on = ['Market', 'Channel', 'Brand',selected_key_L2], how='left')
                    #             # MSP_L0L2
                    #             MSP_L0L2['Total_Revenue_Per_Channel'] = MSP_L0L2.groupby(['Market','Channel'])['Revenue'].transform('sum')
                    #             MSP_L0L2['MShare'] = MSP_L0L2['Revenue']/MSP_L0L2['Total_Revenue_Per_Channel']
                    #             MSP_L0L2 = MSP_L0L2.drop(columns='Total_Revenue_Per_Channel')
                    #             MSP_L0L2.rename(columns={'Volume_definedperiod': 'Volume'}, inplace=True)
                    #             # MSP_L0L2.columns
                    #             MSP_L0L2['CS'] = 0.5 * (MSP_L0L2['MCV'] + MSP_L0L2['Price']) * MSP_L0L2['Avg Volume']
                    #             MSP_L0L2.groupby(['Market','Channel'])['MShare'].sum()
                    #             #Calculating PVS

                    #             # Step 1: Calculate the sum of TPV and MShare within each channel
                    #             MSP_L0L2['Sum_CS_by_Channel'] = MSP_L0L2.groupby(['Market','Channel'])['CS'].transform('sum')
                    #             MSP_L0L2['Sum_MShare_by_Channel'] = MSP_L0L2.groupby(['Market','Channel'])['MShare'].transform('sum')

                    #             # Step 2: Calculate PVS for each row
                    #             MSP_L0L2['PVS'] = (MSP_L0L2['CS'] / MSP_L0L2['Sum_CS_by_Channel']) * MSP_L0L2['Sum_MShare_by_Channel']
                    #             MSP_L0L2['MSP'] = MSP_L0L2['PVS'] - MSP_L0L2['MShare']
                    #             MSP_L0L2 = MSP_L0L2.drop(columns=['Sum_CS_by_Channel','Sum_MShare_by_Channel','PVS'])

                    #             MSP_L0L2['MCV'] = MSP_L0L2.apply(
                    #             lambda row: max(row['MCV'], 1.05 * row['Price']) if row['MCV'] <= row['Price'] else row['MCV'],
                    #             axis=1
                    #             )

                    #             MSP_L0L2['CSF'] = MSP_L0L2['MCV'] / MSP_L0L2['Price']
                    #             MSP_L0L2['NewMShare'] = MSP_L0L2['MShare'] + MSP_L0L2['MSP']
                    #             MSP_L0L2['Price_elas'] = MSP_L0L2['Price_ModelPeriod'] /(MSP_L0L2['Price'] - MSP_L0L2['MCV'])
                    #             MSP_L0L2['Market'] = MSP_L0L2['Market'].str.lower()
                    #             MSP_L0L2['Category'] = MSP_L0L2['Category'].str.lower()

                    #             # List of columns to replace values
                    #             columns_to_replace = ['Region','SubCategory', 'Variant', 'PackType', 'PPG', 'PackSize']

                    #             # Exclude selected_key_L2 if it exists in columns_to_replace
                    #             columns_to_replace = [col for col in columns_to_replace if col != selected_key_L2 and col != selected_key_L0]

                    #             # Replace all values in these columns with 'all'
                    #             MSP_L0L2[columns_to_replace] = 'all'

                    #             MSP_L0L2 = MSP_L0L2.loc[:, ~(MSP_L0L2 == 'all').all() | (MSP_L0L2.columns == 'Channel')]


                    #             # MSP_L0L2
                    #             # Drop rows where the 'CSF' column has NaN values
                    #             MSP_L0L2 = MSP_L0L2.dropna(subset=['CSF'])

                    #             # Reset the index (optional, for cleaner DataFrame)
                    #             MSP_L0L2.reset_index(drop=True, inplace=True)
                    #             # MSP_L0L2
                    #             # MSP_L0L2.shape
                    #             MSP_L0L2 = MSP_L0L2.drop(columns =['Avg Volume'])
                    #             # MSP_L0L2.columns

                    #             L0L2_order = ['Market','Category', 'Channel', selected_key_L0, selected_key_L2, 'MCV', 'Revenue', 'Volume', 'Price', 'CSF', 'Price_elas','Revenue_Total', 'Volume_Total',
                    #                         'Price_ModelPeriod', 'CS', 'MShare', 'NewMShare', 'MSP' ]

                    #             # Reorder the DataFrame
                    #             MSP_L0L2 = MSP_L0L2[L0L2_order]

                    #             # Assume `final_df` is the target DataFrame and `reference_df` is the other DataFrame.
                    #             column_name_to_add = "Vol_Var"  # Name of the column to create in the final DataFrame
                    #             MSP_L0L2 = add_vol_var_column(MSP_L0L2, df_L0, column_name_to_add)

                    #             MSP_L0L2['CsfPeriod'] = Reference_Period

                                
                    #             st.session_state["MSP_L0L2"]=MSP_L0L2

        #------------------------------------------------------------------------------------------------------------------------------

                    # MSP_L0L2L3 T2


                    if "config_data" in st.session_state:
                        config_df=pd.DataFrame([st.session_state.config_data])

                        if (config_df['Modeling Type'].eq(2).any() and config_df['MSP at L0 (Yes/No)'].str.strip().str.lower().eq('yes').any() and
                            config_df['MSP at L2 (Yes/No)'].str.strip().str.lower().eq('yes').any() and config_df['MSP at L3 (Yes/No)'].str.strip().str.lower().eq('yes').any()):


                            if 'df_L3' in st.session_state and 'selected_key_L3' in st.session_state:
                                df_L3 = st.session_state.df_L3

                                df_L2 = st.session_state.df_L2

                                df_L0 = st.session_state.df_L0

                                selected_key_L3 = st.session_state.selected_key_L3
                                selected_key_L2 = st.session_state.selected_key_L2
                                selected_key_L0 = st.session_state.selected_key_L0

                            # df_L3['Channel'] = df_L3['Channel'].str.strip()
                            # df_L3[selected_key_L3] = df_L3[selected_key_L3].str.strip()
                            # df_L3['Market'] = df_L3['Market'].str.strip()

                            # df_L3
                            # Now, select specific columns from the DataFrame
                            columns_to_keep = ['Market','Channel', 'MCV.MCV']  # Replace with your actual column names
                            df_L3_L0L2L3 = df_L3[columns_to_keep + [selected_key_L3]]
                            df_L3_L0L2L3[selected_key_L3] = df_L3_L0L2L3[selected_key_L3].astype('object')
                            # #Check if all values in the 'Channel' column are the same
                            # if df_L3_L0L2L3['Channel'].nunique() == 1:  # If there is only one unique value
                            #     df_L3_L0L2L3['Channel'] = 'all'

                            # df_L3_L0L2L3.columns
                            # Now, select specific columns from the DataFrame
                            columns_to_keep = ['Market','Channel', selected_key_L0, 'MCV']  # Replace with your actual column names
                            MSP_L0L2 = MSP_L0L2[columns_to_keep + [selected_key_L2]] 
                            MSP_L0L2[selected_key_L2] = MSP_L0L2[selected_key_L2].astype('str')
                            D0_L3 = D0.copy()
                            D0_L3[selected_key_L0] = D0_L3[selected_key_L0].astype('str')
                            D0_L3[selected_key_L2] = D0_L3[selected_key_L2].astype('str')
                            D0_L3[selected_key_L3] = D0_L3[selected_key_L3].astype('str')   
                                # Drop columns dynamically
                            D0_L3 = D0_L3.drop(columns=[
                                    'Year', 'Month', 'Week', 'BrCatId'
                                ])
                            
                                # Group the DataFrame
                            D0_L3 = (
                                D0_L3.groupby(['Market', 'Channel','Category', selected_key_L0,selected_key_L2,selected_key_L3, 'date'], as_index=False)
                                .agg({
                                    'SalesValue': 'sum',  # Sum of SalesValue
                                    'Volume': 'sum',      # Sum of Volume 
                                })
                            )

                            D0_L3_grouped_SalesValue = D0_L3.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3])['SalesValue'].sum().reset_index()
                            D0_L3_grouped_Volume = D0_L3.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3])['Volume'].sum().reset_index()
                            D0_L3_grouped = pd.merge(D0_L3_grouped_SalesValue,D0_L3_grouped_Volume,on=['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3],how='left')
                            # D0_L3_grouped
                            # MSP_L0L2.columns
                            D0_L3_grouped = pd.merge(D0_L3_grouped,MSP_L0L2,on=['Market', 'Channel', selected_key_L0,selected_key_L2],how='left')
                            # D0_L3_grouped

                            D0_L3_grouped = D0_L3_grouped.dropna(subset=['MCV'])

                            D0_L3_grouped.rename(columns={'MCV': 'MCV_of_L0L2'}, inplace=True)
                            D0_L3_grouped = pd.merge(D0_L3_grouped,df_L3_L0L2L3,on=['Market', 'Channel', selected_key_L3],how='left')

                            D0_L3_grouped = D0_L3_grouped.dropna(subset=['MCV.MCV'])

                            D0_L3_grouped.rename(columns={'MCV.MCV': 'MCV_L3'}, inplace=True)
                            # D0_L3_grouped
                            D0_L3_grouped['Total_Volume_ByChannel'] = D0_L3_grouped.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2])['Volume'].transform('sum')
                            # Calculate the product of MCV_L3 and Volume
                            D0_L3_grouped['Sum_Product_ByChannel'] = (
                                D0_L3_grouped['MCV_L3'] * D0_L3_grouped['Volume']
                            ).groupby([D0_L3_grouped['Market'], D0_L3_grouped['Channel'], D0_L3_grouped[selected_key_L0],D0_L3_grouped[selected_key_L2]]).transform('sum')
                            # D0_L3_grouped
                            D0_L3_grouped['Wtd_MCV_ByL0&L2_using_L3MCV'] = D0_L3_grouped['Sum_Product_ByChannel'] / D0_L3_grouped['Total_Volume_ByChannel']
                            D0_L3_grouped['Correction_Factor'] = D0_L3_grouped['MCV_of_L0L2'] / D0_L3_grouped['Wtd_MCV_ByL0&L2_using_L3MCV']
                            D0_L3_grouped['MCV'] = D0_L3_grouped['Correction_Factor'] * D0_L3_grouped['MCV_L3']
                            # D0_L3_grouped
                            # Calculating MShare for Whole Period Data
                            D0_L3['Revenue_Total'] = D0_L3.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3])['SalesValue'].transform('sum')
                            D0_L3['Volume_Total'] = D0_L3.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3])['Volume'].transform('sum')
                            D0_L3['Price_ModelPeriod'] = D0_L3['Revenue_Total']/D0_L3['Volume_Total']
                            # Calculating Revenue, Volume and Price for defined time Period
                            D0_L3_filtered = defined_period_data(D0_L3, Reference_Period)
                            D0_L3_filtered['Revenue'] = D0_L3_filtered.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3])['SalesValue'].transform('sum')
                            D0_L3_filtered['Volume_definedperiod'] = D0_L3_filtered.groupby(['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3])['Volume'].transform('sum')
                            D0_L3_filtered['Price'] = D0_L3_filtered['Revenue']/D0_L3_filtered['Volume_definedperiod'] 
                            # D0_L3_filtered
                            D0_L3_filtered['Avg Volume'] = D0_L3_filtered['Volume_definedperiod']/Reference_Period

                            # Drop columns dynamically
                            D0_L3_filtered = D0_L3_filtered.drop(columns=['date', 'SalesValue', 'Volume'])
                            
                            # Create a DataFrame with unique brands, keeping the first occurrence of other columns
                            D0_L3_filtered = D0_L3_filtered.drop_duplicates(subset=['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3], keep='first')
                            # D0_L3_filtered.shape
                            # List of columns to keep



                            if "D0_L3_grouped" not in st.session_state:
                                    st.session_state.D0_L3_grouped = None



                            columns_to_keep = ['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3,'MCV']

                            # Filter the DataFrame to include only these columns
                            D0_L3_grouped = D0_L3_grouped[columns_to_keep]
                            # D0_L3_grouped.shape
                            # D0_L3_grouped
                            # df_L3_L0L2L3.shape
                            # df_L3_L0L2L3


                            if "D0_L3_grouped_original" not in st.session_state:
                                    st.session_state.D0_L3_grouped_original = None  # Store original values
                                    st.session_state.D0_L3_grouped_modified = False



                            if st.session_state.D0_L3_grouped_modified:
                                D0_L3_grouped = st.session_state.D0_L3_grouped  # Keep the saved edited version
                            else:
                                D0_L3_grouped = D0_L3_grouped[columns_to_keep].copy()  # Generate fresh copy from df_L0
                                D0_L3_grouped["original_MCV"] = D0_L3_grouped["MCV"]

                                st.session_state.D0_L3_grouped_original = D0_L3_grouped.copy()


                            st.session_state.D0_L3_grouped=D0_L3_grouped



                            with st.expander("✏️EDIT MCV"):

                                # Define column configuration (e.g., making specific columns editable)
                                column_config = {
                                    "MCV": st.column_config.NumberColumn("MCV", help="Edit this value"),
                                }

                                # Editable DataFrame
                                edited_D0_L3_grouped = st.data_editor(
                                    D0_L3_grouped,
                                    column_config=column_config,
                                    disabled=[col for col in D0_L3_grouped.columns if col != "MCV"]+["original_MCV"],
                                    use_container_width=True
                                )

                                col1, col2 = st.columns(2)

                                with col1:
                                    if st.button("SAVE CHANGES", key="save_changes_D0_L3"):
                                        if not edited_D0_L3_grouped.equals(st.session_state.D0_L3_grouped_original):  # Check if edits were made
                                            st.session_state.D0_L3_grouped = edited_D0_L3_grouped.copy()
                                            st.session_state.D0_L3_grouped_modified = True  # Mark as modified
                                            st.success("Changes saved! They will persist until reset.")
                                            st.rerun()

                                        else:
                                            st.info("No changes detected. Using the original DataFrame.")

                                with col2:
                                    if st.button("RESET", key="reset_D0_L3"):
                                        st.session_state.D0_L3_grouped = st.session_state.D0_L3_grouped_original.copy()
                                        st.session_state.D0_L3_grouped_modified = False  # Reset modification flag
                                        st.success("Data reset to original values!")
                                        st.rerun()

                                st.info("The above edits will be considered as final and will not change even if the models are reselected.\n\nIf you want to change selected models, please click on 'RESET' before change.")



                        
                        
                            df_L3_L0L2L3 = df_L3_L0L2L3.drop(columns='MCV.MCV')



                            if 'D0_L3_grouped' in st.session_state:
                                D0_L3_grouped=st.session_state.D0_L3_grouped

                                # df_L3_L0L2L3.columns
                                MSP_L3 = pd.merge(df_L3_L0L2L3,D0_L3_grouped, on = ['Market', 'Channel',selected_key_L3], how='left')
                                # MSP_L3
                                MSP_L0L2L3 = pd.merge(MSP_L3,D0_L3_filtered, on = ['Market', 'Channel', selected_key_L0,selected_key_L2,selected_key_L3], how='left')


                            # MSP_L0L2L3
                            MSP_L0L2L3['Total_Revenue_Per_Channel'] = MSP_L0L2L3.groupby(['Market','Channel'])['Revenue'].transform('sum')
                            MSP_L0L2L3['MShare'] = MSP_L0L2L3['Revenue']/MSP_L0L2L3['Total_Revenue_Per_Channel']
                            MSP_L0L2L3 = MSP_L0L2L3.drop(columns='Total_Revenue_Per_Channel')
                            MSP_L0L2L3.rename(columns={'Volume_definedperiod': 'Volume'}, inplace=True)
                            # MSP_L0L2L3.columns
                            MSP_L0L2L3['CS'] = 0.5 * (MSP_L0L2L3['MCV'] + MSP_L0L2L3['Price']) * MSP_L0L2L3['Avg Volume']
                            # MSP_L0L2L3.groupby(['Market','Channel'])['MShare'].sum()
                            #Calculating PVS

                            # Step 1: Calculate the sum of TPV and MShare within each channel
                            MSP_L0L2L3['Sum_CS_by_Channel'] = MSP_L0L2L3.groupby(['Market','Channel'])['CS'].transform('sum')
                            MSP_L0L2L3['Sum_MShare_by_Channel'] = MSP_L0L2L3.groupby(['Market','Channel'])['MShare'].transform('sum')

                            # Step 2: Calculate PVS for each row
                            MSP_L0L2L3['PVS'] = (MSP_L0L2L3['CS'] / MSP_L0L2L3['Sum_CS_by_Channel']) * MSP_L0L2L3['Sum_MShare_by_Channel']
                            MSP_L0L2L3['MSP'] = MSP_L0L2L3['PVS'] - MSP_L0L2L3['MShare']
                            MSP_L0L2L3 = MSP_L0L2L3.drop(columns=['Sum_CS_by_Channel','Sum_MShare_by_Channel','PVS']) 

                            MSP_L0L2L3['MCV'] = MSP_L0L2L3.apply(
                            lambda row: max(row['MCV'], 1.05 * row['Price']) if row['MCV'] <= row['Price'] else row['MCV'],
                            axis=1
                            )
                            
                            MSP_L0L2L3['CSF'] = MSP_L0L2L3['MCV'] / MSP_L0L2L3['Price']
                            MSP_L0L2L3['NewMShare'] = MSP_L0L2L3['MShare'] + MSP_L0L2L3['MSP']
                            MSP_L0L2L3['Price_elas'] = MSP_L0L2L3['Price_ModelPeriod'] /(MSP_L0L2L3['Price'] - MSP_L0L2L3['MCV'])
                            MSP_L0L2L3['Market'] = MSP_L0L2L3['Market'].str.lower()
                            MSP_L0L2L3['Category'] = MSP_L0L2L3['Category'].str.lower()

                            # List of columns to replace values
                            columns_to_replace = ['Region', 'SubCategory', 'Variant', 'PackType', 'PPG', 'PackSize']

                            # Exclude selected_key_L2 and selected_key_L3 from columns_to_replace
                            columns_to_replace = [col for col in columns_to_replace if col not in [selected_key_L0, selected_key_L2, selected_key_L3]]


                            # Replace all values in these columns with 'all'
                            MSP_L0L2L3[columns_to_replace] = 'all'

                            MSP_L0L2L3 = MSP_L0L2L3.loc[:, ~(MSP_L0L2L3 == 'all').all() | (MSP_L0L2L3.columns == 'Channel')]


                            # MSP_L0L2L3
                            # Drop rows where the 'CSF' column has NaN values
                            MSP_L0L2L3 = MSP_L0L2L3.dropna(subset=['CSF'])

                            # Reset the index (optional, for cleaner DataFrame)
                            MSP_L0L2L3.reset_index(drop=True, inplace=True)
                            # MSP_L0L2L3
                            # MSP_L0L2L3.shape
                            MSP_L0L2L3 = MSP_L0L2L3.drop(columns =['Avg Volume'])
                            # MSP_L0L2L3.columns

                            L0L2L3_order = ['Market', 'Channel','Category', selected_key_L0, selected_key_L2, selected_key_L3,'MCV','original_MCV','Revenue', 'Volume', 'Price', 'CSF', 'Price_elas','Revenue_Total', 'Volume_Total',
                                        'Price_ModelPeriod', 'CS', 'MShare', 'NewMShare', 'MSP' ]

                            # Reorder the DataFrame
                            MSP_L0L2L3 = MSP_L0L2L3[L0L2L3_order]

                            # Assume `final_df` is the target DataFrame and `reference_df` is the other DataFrame.
                            column_name_to_add = "Vol_Var"  # Name of the column to create in the final DataFrame
                            MSP_L0L2L3 = add_vol_var_column(MSP_L0L2L3, df_L0, column_name_to_add)

                            MSP_L0L2L3['CsfPeriod'] = Reference_Period



                            st.session_state["MSP_L0L2L3"]=MSP_L0L2L3


                        else:
                            st.error("FILL THE CONFUGURATION FILE FOR L3 LEVEL!")

            












                
                    import plotly.graph_objects as go




                    # options = ["GRAPHS"]
                    # option = st.pills(f"SHOW GRAPHS!", options)

                    # if option == "GRAPHS":


                    # # Filter data based on selected categories, markets, and channels
                    global_filtered_data = st.session_state["MSP_L0L2L3"].copy()

                    # Apply filters if they exist
                    # if st.session_state["selected_categories3"]:
                    #     global_filtered_data = global_filtered_data[global_filtered_data['Category'].isin(st.session_state["selected_categories3"])]
                    # if st.session_state["selected_markets3"]:
                    #     global_filtered_data = global_filtered_data[global_filtered_data['Market'].isin(st.session_state["selected_markets3"])]
                    if st.session_state["selected_channels3"]:
                        global_filtered_data = global_filtered_data[global_filtered_data['Channel'].isin(st.session_state["selected_channels3"])]



                    col14,col15=st.columns(2)


                    with col14:
                        if 'L2_name' in st.session_state and st.session_state['L2_name']:
                            L2_name_column = st.session_state['L2_name']

                            L2_name_column_options = global_filtered_data[L2_name_column].unique()
                            L2_name_column_filter = st.multiselect(
                                f"Select {L2_name_column}",
                                options=L2_name_column_options,
                                # key=f"{L23_name_column}"
                            )

                            filtered_data_by_L2_name_column = global_filtered_data.copy()
                            if L2_name_column_filter:
                                filtered_data_by_L2_name_column = filtered_data_by_L2_name_column[filtered_data_by_L2_name_column[L2_name_column].isin(L2_name_column_filter)]


                    with col15:


                        if 'L3_name' in st.session_state and st.session_state['L3_name']:
                            L3_name_column = st.session_state['L3_name']

                            L3_name_column_options = filtered_data_by_L2_name_column[L3_name_column].unique()
                            L3_name_column_filter = st.multiselect(
                                f"Select {L3_name_column}",
                                options=L3_name_column_options,
                                key=f"{L3_name_column}"
                            )

                            # Further filter based on selected RPIto
                            filtered_data_by_L3_name_column = filtered_data_by_L2_name_column.copy()
                            if L3_name_column_filter:
                                filtered_data_by_L3_name_column = filtered_data_by_L3_name_column[filtered_data_by_L3_name_column[L3_name_column].isin(L3_name_column_filter)]


                    

                    # Check if any data is left after filtering
                    if filtered_data_by_L3_name_column.empty:
                        st.warning("No data to display after applying the filters.")
                    else:

                        df = filtered_data_by_L3_name_column.copy()
                        df["MSP"] = df["MSP"] * 100  
                        df["MSP_label"] = df["MSP"].apply(lambda x: f"{x:.2f}%")

                        df["MShare"] = df["MShare"] * 100  
                        df["MShare_label"] = df["MShare"].apply(lambda x: f"{x:.2f}%")

                        # Plotly chart for CSF
                        fig_csff= px.bar(
                            df,
                            x='Brand',  # Ensure index is treated as string for display
                            y='CSF',
                            template="plotly_white",
                            color='Brand',
                            text_auto=True,  # Display y-axis values on top of bars
                            hover_data=["Channel",L2_name_column,L3_name_column],  # Display y-axis values on top of bars
                        )

                        # fig_csff.update_traces(textposition="outside")  # Position labels outside bars

                        # Customize hovertemplate for more detailed hover information
                        fig_csff.update_traces(
                            hovertemplate=
                                        'Channel: <b>%{customdata[0]}<br><br>'
                                        f'{L3_name_column}: <b>%{{customdata[2]}}</b><br>'
                                        f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'
                                        '<b>%{x}</b><br>'  # Brand
                                        # Channel
                                        '<extra></extra>'  # Remove extra information like trace name
                        )

                        fig_csff.update_layout(
                            title="CSF",
                            xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                            yaxis=dict(title="CSF", color='black', showgrid=False, tickformat=".2f"),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font_color='black',
                            legend=dict(
                                orientation="h",  # Horizontal orientation
                                xanchor="left",  # Anchor to the left
                                x=0,  # Position at the left edge
                                yanchor="bottom",  # Anchor to the bottom of the legend
                                y=1.02  # Position above the chart
                            )
                        )

                        st.plotly_chart(fig_csff, use_container_width=True)


                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        



                        # Plotly chart for MSP
                        fig_msp = px.bar(
                            df,
                            x='Brand',  # Ensure index is treated as string for display
                            y='MSP',
                            template="plotly_white",
                            color='Brand',
                            text=df["MSP_label"],
                            hover_data=["Channel",L2_name_column,L3_name_column],  # Display y-axis values on top of bars
                        )

                        fig_msp.update_traces(textposition="outside")  # Position labels outside bars

                        # Customize hovertemplate for more detailed hover information
                        fig_msp.update_traces(
                            hovertemplate=
                                        'Channel: <b>%{customdata[0]}<br><br>'
                                        f'{L3_name_column}: <b>%{{customdata[2]}}</b><br>'
                                        f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'
                                        '<b>%{x}</b><br>'  # Brand
                                        # Channel
                                        '<extra></extra>'  # Remove extra information like trace name
                        )

                        fig_msp.update_layout(
                            title="MSP",
                            xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                            yaxis=dict(title="MSP", color='black', showgrid=False, tickformat=".2f"),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font_color='black',
                            legend=dict(
                                orientation="h",  # Horizontal orientation
                                xanchor="left",  # Anchor to the left
                                x=0,  # Position at the left edge
                                yanchor="bottom",  # Anchor to the bottom of the legend
                                y=1.02  # Position above the chart
                            )
                        )

                        st.plotly_chart(fig_msp, use_container_width=True)


                        st.markdown('<hr class="thin">', unsafe_allow_html=True)



                        # Plotly chart for MSP
                        fig_ms = px.bar(
                            df,
                            x='Brand',  # Ensure index is treated as string for display
                            y='MShare',
                            template="plotly_white",
                            color='Brand',
                            text=df["MShare_label"],  # Display y-axis values on top of bars
                            hover_data=["Channel",L2_name_column,L3_name_column],  # Display y-axis values on top of bars
                        )

                        # fig_csff.update_traces(textposition="outside")  # Position labels outside bars

                        # # Customize hovertemplate for more detailed hover information
                        fig_ms.update_traces(
                            hovertemplate=
                                        'Channel: <b>%{customdata[0]}<br><br>'
                                        f'{L3_name_column}: <b>%{{customdata[2]}}</b><br>'
                                        f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'
                                        '<b>%{x}</b><br>'  # Brand
                                        # Channel
                                        '<extra></extra>'  # Remove extra information like trace name
                        )

                        fig_ms.update_layout(
                            title="MSHARE",
                            xaxis=dict(title="", color='black', showgrid=False, showticklabels=True),
                            yaxis=dict(title="MShare", color='black', showgrid=False, tickformat=".2f"),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font_color='black',
                            legend=dict(
                                orientation="h",  # Horizontal orientation
                                xanchor="left",  # Anchor to the left
                                x=0,  # Position at the left edge
                                yanchor="bottom",  # Anchor to the bottom of the legend
                                y=1.02  # Position above the chart
                            )
                        )

                        st.plotly_chart(fig_ms, use_container_width=True)


                        st.markdown('<hr class="thin">', unsafe_allow_html=True)




                        # # Create figure
                        # fig_mp = go.Figure()

                        # # Add Price bars
                        # fig_mp.add_trace(go.Bar(
                        #     x=global_filtered_data['Brand'],
                        #     y=global_filtered_data['Price'],
                        #     name='Price',
                        #     marker_color='blue'
                        # ))

                        # # Add MCV bars
                        # fig_mp.add_trace(go.Bar(
                        #     x=global_filtered_data['Brand'],
                        #     y=global_filtered_data['MCV'],
                        #     name='MCV',
                        #     marker_color='orange'
                        # ))

                        # # Update layout
                        # fig_mp.update_layout(
                        #     barmode='group',  # Group bars next to each other
                        #     title="Price & MCV",
                        #     xaxis=dict(title="Brand", color='black', showgrid=False, showticklabels=True),
                        #     yaxis=dict(title="Value", color='black', showgrid=False, tickformat=".2f"),
                        #     plot_bgcolor='white',
                        #     paper_bgcolor='white',
                        #     font_color='black',
                        #     legend=dict(
                        #         orientation="h",
                        #         xanchor="left",
                        #         x=0,
                        #         yanchor="bottom",
                        #         y=1.02
                        #     )
                        # )

                        # # Display in Streamlit
                        # st.plotly_chart(fig_mp, use_container_width=True)\


                        fig_mp = go.Figure()

                        # Add Price bars with a color scale
                        fig_mp.add_trace(go.Bar(
                            x=df['Brand'],
                            y=df['Price'],
                            name='Price',
                            text=df['Price'].round(2),  # Show values rounded to 2 decimal places
                            textposition='auto',
                            marker=dict(color=df['Brand'].astype('category').cat.codes,  # Convert Brand to category and map to a code
                                        colorscale='Viridis'),  # Apply a colorscale

                            hovertemplate=
                                        '<b>Channel:</b> %{customdata[0]}<br>' +
                                        f'{L3_name_column}: <b>%{{customdata[2]}}</b><br>'+
                                        f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'+
                                        '<b>Brand:</b> %{x}<br>' +
                                        '<b>Price:</b> %{y:.2f}<br>' ,

                            customdata=df[['Channel', L2_name_column, L3_name_column]].values
                                
                        ))

                        # Add MCV bars with a color scale
                        fig_mp.add_trace(go.Bar(
                            x=df['Brand'],
                            y=df['MCV'],
                            text=df['MCV'].round(2),  # Show values rounded to 2 decimal places
                            textposition='auto',
                            name='MCV',
                            marker=dict(color=df['Brand'].astype('category').cat.codes,  # Convert Brand to category and map to a code
                                        colorscale='Viridis'),  # Apply a colorscale

                            hovertemplate=
                                        '<b>Channel:</b> %{customdata[0]}<br>' +
                                        f'{L3_name_column}: <b>%{{customdata[2]}}</b><br>'+
                                        f'{L2_name_column}: <b>%{{customdata[1]}}</b><br>'+
                                        '<b>Brand:</b> %{x}<br>' +
                                        '<b>Price:</b> %{y:.2f}<br>',

                            customdata=df[['Channel',L2_name_column, L3_name_column]].values
                        ))

                        # Update layout
                        fig_mp.update_layout(
                            barmode='group',  # Group bars next to each other
                            title="Price & MCV",
                            xaxis=dict(title="Brand", color='black', showgrid=False, showticklabels=True),
                            yaxis=dict(title="Value", color='black', showgrid=False, tickformat=".2f"),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font_color='black',
                            legend=dict(
                                orientation="h",
                                xanchor="left",
                                x=0,
                                yanchor="bottom",
                                y=1.02
                            ),
                            showlegend=False
                        )

                        # Display in Streamlit
                        st.plotly_chart(fig_mp, use_container_width=True)



                        st.markdown('<hr class="thin">', unsafe_allow_html=True)




                    options = ["MSPL0L2L3 FILE"]
                    option = st.pills(f"SHOW MSP_L0L2L3 FILE!", options,default='MSPL0L2L3 FILE')

                    if option == "MSPL0L2L3 FILE":
                        st.write('MSP L0L2L3 FILE:')
                        st.dataframe(st.session_state["MSP_L0L2L3"])


                        st.download_button("Download the final modified data MSP_L0L2L3", 
                            data=st.session_state["MSP_L0L2L3"].to_csv(index=False), 
                            file_name="MSP_L0L2L3.csv", 
                            mime="csv")


                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

