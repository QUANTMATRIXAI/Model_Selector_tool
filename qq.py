
import uuid 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import zscore, shapiro
from io import BytesIO
from zipfile import ZipFile
 
# Page Configuration
st.set_page_config(
    page_title="Quant Matrix AI - Enhanced EDA Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

 
# Updated CSS Styling with Black Text and Yellow Background
st.markdown(
    """
    <style>
    /* Main Header Styling */
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        color: black;
        background: yellow;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
    }

    /* Sidebar Section Styling */
    .sidebar-header {
        font-size: 1.5em;
        font-weight: bold;
        color: black;
        background: yellow;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        text-align: center;
    }

    .stSidebar .sidebar-content {
        background: #f9f9f9;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.8em;
        color: black;
        font-weight: bold;
        background: yellow;
        padding: 10px;
        border-radius: 5px;
        text-align: left;
        margin-bottom: 20px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.15);
    }

    /* Buttons */
    .btn-style {
        font-size: 1em;
        font-weight: bold;
        color: black;
        background-color: yellow;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        transition: 0.3s ease-in-out;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .btn-style:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    /* Highlight Effect */
    .highlight:hover {
        background: #ffd700; /* Slightly darker yellow for hover effect */
        color: black !important;
    }

    /* Icon Styling */
    .icon {
        font-size: 1.5em;
        margin-right: 10px;
        vertical-align: middle;
        color: black; /* Black icons */
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-header"><i class="fas fa-chart-line icon"></i> Quant Matrix AI - Enhanced EDA Tool</div>',
    unsafe_allow_html=True
)

# Initialize session state for saved charts and pinned results
if "saved_charts" not in st.session_state:
    st.session_state.saved_charts = []
if "pinned_chart" not in st.session_state:
    st.session_state.pinned_chart = None


# Sidebar - File Upload
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
 
# If the file is uploaded, load data
if uploaded_file:
    # Read data
    if uploaded_file.name.endswith('.csv'):
        dataframe = pd.read_csv(uploaded_file)
    else:
        dataframe = pd.read_excel(uploaded_file)
 
    st.success("File uploaded successfully!")
 
    # Identify columns
    numerical_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
 
    # Layout for Visualization and Saved Charts
    col1, col2, col3 = st.columns([1, 3, 1])
 
    # **Pinned Chart Section**
    if st.session_state.pinned_chart:
        with col1:
            st.markdown('<div class="pin-section">', unsafe_allow_html=True)
            st.write("📌 **Pinned Chart**")
            st.write(st.session_state.pinned_chart[0])
            st.plotly_chart(st.session_state.pinned_chart[1], use_container_width=True, key="pinned_chart")
            st.markdown('</div>', unsafe_allow_html=True)
 
# **Center Column - Data Slicing Options and Generated Charts**
with col2:
    st.header("Data Slicing Options")
    slicer_active = st.checkbox("Enable Slicer", value=False)

    if slicer_active:
        slicer_columns = ["Market", "Channel", "Region", "Category", "SubCategory", "Brand", "Variant", "PackType", "PPG", "PackSize", "Year", "Month", "Week"]
        filters = {}

        # Divide slicer columns into two rows for better layout
        row1_cols = slicer_columns[:len(slicer_columns) // 2]
        row2_cols = slicer_columns[len(slicer_columns) // 2:]

        # Create slicer UI for the first row of columns
        row1 = st.columns(len(row1_cols))
        for idx, col in enumerate(row1_cols):
            if col in dataframe.columns:
                with row1[idx]:
                    selected_values = st.multiselect(f"{col}", options=dataframe[col].unique(), key=f"slicer_{col}")
                    if selected_values:
                        filters[col] = selected_values

        # Create slicer UI for the second row of columns
        row2 = st.columns(len(row2_cols))
        for idx, col in enumerate(row2_cols):
            if col in dataframe.columns:
                with row2[idx]:
                    selected_values = st.multiselect(f"{col}", options=dataframe[col].unique(), key=f"slicer_{col}")
                    if selected_values:
                        filters[col] = selected_values

        # Apply filters to create filtered_data
        filtered_data = dataframe.copy()
        for col, selected_values in filters.items():
            filtered_data = filtered_data[filtered_data[col].isin(selected_values)]

        # Display filtered data preview
        show_data_preview = st.checkbox("Show Filtered Data Preview", value=True, key="data_preview_toggle")
        if show_data_preview:
            st.write("Filtered Data Preview:")
            st.dataframe(filtered_data)

    # Use filtered_data as the default data for the rest of the app
# Ensure `filtered_data` is returned if slicer is active, otherwise use `dataframe`
if not slicer_active:
    filtered_data = dataframe.copy()

# Visualization Options
with st.sidebar.expander("📊 Visualization Options", expanded=True):
    st.header("Visualization Options")
    enable_visualization = st.checkbox("Enable Visualization", key="enable_visualization")

    if enable_visualization:
        # X and Y Axis Selection
        x_axis = st.selectbox("Select X-axis", options=dataframe.columns, key="x_axis")
        y_axis = st.selectbox("Select Y-axis", options=numerical_columns, key="y_axis")
        chart_type = st.selectbox("Chart Type", ["Bar", "Scatter", "Pie", "Histogram"], key="chart_type")
        threshold = st.number_input("Value Threshold", value=0, help="Display only values above this threshold in the chart.")
        custom_title = st.text_input("Custom Chart Title", value="")

        # Apply Slicer Filters if Enabled
        data_to_plot = filtered_data if slicer_active else dataframe
        data_to_plot = data_to_plot if threshold == 0 else data_to_plot[data_to_plot[y_axis] > threshold]

        # Generate Charts Based on Selected Type
        fig = None
        if chart_type == "Bar":
            fig = px.bar(data_to_plot, x=x_axis, y=y_axis, title=custom_title or f"{y_axis} vs {x_axis} - Bar Chart")
        elif chart_type == "Scatter":
            fig = px.scatter(data_to_plot, x=x_axis, y=y_axis, title=custom_title or f"{y_axis} vs {x_axis} - Scatter Plot")
        elif chart_type == "Pie":
            fig = px.pie(data_to_plot, names=x_axis, values=y_axis, title=custom_title or f"{y_axis} Distribution by {x_axis}")
        elif chart_type == "Histogram":
            fig = px.histogram(data_to_plot, x=x_axis, title=custom_title or f"{x_axis} Histogram")

        # Display Chart
        with col2:
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"main_viz_chart_{chart_type}")
            else:
                st.warning("Please select chart type and variables to generate a chart.")


# **Trendline Options**
with st.sidebar.expander("📈 Time-Based Trendline", expanded=True):
    st.header("Trendline Analysis")
    time_based_y_axis = st.multiselect("Select Y-axis for Trendline", options=numerical_columns, key="trend_y_axis")
    time_column = st.selectbox("Time Column (e.g., Year, Month, Week)", options=["Year", "Month", "Week", "Date"], key="time_column")
    aggregation_type = st.selectbox("Aggregation Type", ["Sum", "Mean"], key="aggregation_type")

    if time_based_y_axis and time_column:
        trend_data = filtered_data if slicer_active else dataframe.copy()

        if time_column == "Year":
            trend_data["Time"] = pd.to_datetime(trend_data["Year"], format='%Y')
        elif time_column == "Month":
            trend_data["Time"] = pd.to_datetime(trend_data["Year"].astype(str) + '-' + trend_data["Month"].astype(str), format='%Y-%m')
        elif time_column == "Week":
            trend_data["Time"] = pd.to_datetime(trend_data["Year"].astype(str) + trend_data["Week"].astype(str) + '1', format='%Y%U%w')
        else:
            trend_data["Time"] = pd.to_datetime(trend_data["Date"])

        trend_agg = trend_data.groupby("Time")[time_based_y_axis].agg(aggregation_type.lower()).reset_index()

        trendline_fig = px.line(
            trend_agg,
            x="Time",
            y=time_based_y_axis,
            title=f"Trendline of {', '.join(time_based_y_axis)} Over {time_column}",
            markers=True
        )

        with col2:
            st.plotly_chart(trendline_fig, use_container_width=True, key="trendline_chart")

        # Market Share Analysis Section
with st.sidebar.expander("📊 Market Share Analysis", expanded=True):
    st.header("Market Share Analysis")

    # Enable/Disable Market Share Analysis
    enable_market_share = st.checkbox("Enable Market Share Analysis", key="enable_market_share")

    if enable_market_share:
        # Select Columns for Market Share
        category_column = st.selectbox(
            "Select Category Column (e.g., Region, Brand):",
            categorical_columns,
            key="market_category"
        )
        value_column = st.selectbox(
            "Select Value Column (e.g., Sales, Revenue):",
            numerical_columns,
            key="market_value"
        )

        if category_column and value_column:
            # Use filtered_data if slicer is active; otherwise, use the original dataframe
            data_for_analysis = filtered_data if slicer_active else dataframe

            # Compute Market Share
            market_share_data = data_for_analysis.groupby(category_column)[value_column].sum().reset_index()
            market_share_data["Market Share (%)"] = (
                market_share_data[value_column] / market_share_data[value_column].sum()
            ) * 100
            market_share_data = market_share_data.sort_values("Market Share (%)", ascending=False)

            # Display Results in Center
            with col2:
                st.subheader(f"Market Share by {category_column}")
                st.dataframe(market_share_data, use_container_width=True)

                # Visualize Market Share
                st.subheader(f"Market Share Visualization ({category_column})")
                fig_pie = px.pie(
                    market_share_data,
                    names=category_column,
                    values=value_column,
                    title=f"Market Share Distribution ({category_column})",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Bar Chart Option
                st.subheader(f"Market Share Bar Chart ({category_column})")
                fig_bar = px.bar(
                    market_share_data,
                    x=category_column,
                    y="Market Share (%)",
                    text="Market Share (%)",
                    title=f"Market Share (%) by {category_column}",
                    template="plotly_white"
                )
                fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)


        
        # Data Overview in Sidebar
with st.sidebar.expander("📊 Data Overview", expanded=True):
    st.header("Data Overview")

    # Describe Data Option
    show_description = st.checkbox("Describe Data", key="describe_data")

    # Handle Missing Values Option
    handle_missing = st.checkbox("Handle Missing Values", key="handle_missing")
    if handle_missing:
        missing_action = st.radio(
            "Choose an action to handle missing values:",
            options=["None", "Remove Rows", "Fill with Mean", "Fill with Median"],
            key="missing_action"
        )

    # Results in Center Column
    with col2:
        # Use filtered_data if slicer is active; otherwise, use the original dataframe
        data_for_overview = filtered_data if slicer_active else dataframe

        # Show Description
        if show_description:
            st.subheader("Data Description")
            st.write("Summary Statistics of the Dataset:")
            st.dataframe(data_for_overview.describe())

        # Handle Missing Values
        if handle_missing and missing_action != "None":
            modified_df = data_for_overview.copy()
            for col in modified_df.columns:
                if modified_df[col].isnull().any():
                    if missing_action == "Remove Rows":
                        modified_df.dropna(inplace=True)
                    elif missing_action == "Fill with Mean":
                        modified_df[col].fillna(modified_df[col].mean(), inplace=True)
                    elif missing_action == "Fill with Median":
                        modified_df[col].fillna(modified_df[col].median(), inplace=True)

            st.subheader("Dataset After Handling Missing Values")
            st.dataframe(modified_df)

            # Save updated data for further usage if needed
            st.session_state.modified_data = modified_df

        
        # Correlation Matrix in Sidebar
with st.sidebar.expander("📈 Correlation Matrix", expanded=True):
    st.header("📈 Correlation Analysis")

    # Enable Correlation Analysis
    enable_correlation = st.checkbox("Enable Correlation Matrix", key="enable_correlation")

    if enable_correlation:
        st.subheader("Correlation Configuration")
        # Option to select specific variables
        selected_variables = st.multiselect(
            "Select Numerical Variables for Correlation (Optional):",
            options=numerical_columns,
            default=numerical_columns,  # Default to all numerical columns
            key="selected_corr_vars"
        )

        # Results in Center Column
        with col2:
            st.subheader("Correlation Matrix Result")

            if len(selected_variables) > 1:
                # Filtered data compatibility
                data_to_analyze = filtered_data if slicer_active else dataframe

                # Calculate correlation matrix for selected variables
                correlation_matrix = data_to_analyze[selected_variables].corr()

                # Plot Correlation Matrix
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    color_continuous_scale="viridis",
                    title=f"Correlation Matrix for Selected Variables"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Please select at least two numerical variables for correlation analysis.")
    else:
        st.info("Enable the correlation matrix analysis to proceed.")

        
        
        # Distribution Analysis in Sidebar
with st.sidebar.expander("📊 Distribution Analysis", expanded=True):
    st.header("📊 Distribution Analysis")

    # Enable Distribution Analysis
    enable_distribution = st.checkbox("Enable Distribution Analysis", key="enable_distribution")

    if enable_distribution:
        st.subheader("Distribution Configuration")
        
        # Variable Selection
        selected_distribution_vars = st.multiselect(
            "Select Variables for Distribution Analysis:",
            options=numerical_columns,
            default=numerical_columns,  # Default to all numerical columns
            key="selected_distribution_vars"
        )

        # Results in Center Column
        with col2:
            st.subheader("Distribution Analysis Results")

            if selected_distribution_vars:
                for var in selected_distribution_vars:
                    st.markdown(f"### Distribution for {var}")

                    # Create two columns for horizontal display
                    hist_col, qq_col = st.columns(2)

                    # Histogram
                    with hist_col:
                        st.markdown(f"**Histogram of {var}:**")
                        data_to_analyze = filtered_data if slicer_active else dataframe
                        fig_hist = px.histogram(
                            data_to_analyze,
                            x=var,
                            nbins=30,
                            title=f"Histogram of {var}",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Q-Q Plot
                    with qq_col:
                        st.markdown(f"**Q-Q Plot of {var}:**")
                        data_sorted = np.sort(data_to_analyze[var].dropna())
                        qq_fig = px.scatter(
                            x=data_sorted,
                            y=np.random.normal(
                                loc=data_to_analyze[var].mean(),
                                scale=data_to_analyze[var].std(),
                                size=data_sorted.shape[0]
                            ),
                            title=f"Q-Q Plot of {var}",
                            labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                            template="plotly_white"
                        )
                        qq_fig.update_traces(marker=dict(size=5, opacity=0.7))
                        st.plotly_chart(qq_fig, use_container_width=True)

                
            else:
                st.warning("Please select at least one variable for distribution analysis.")
    else:
        st.info("Enable the distribution analysis to proceed.")

        

        # Statistical Tests in Sidebar
with st.sidebar.expander("🔬 Statistical Tests", expanded=True):
    st.header("🔬 Statistical Tests")

    # Enable/Disable Statistical Tests
    enable_stat_tests = st.checkbox("Enable Statistical Tests", key="enable_stat_tests")

    if enable_stat_tests:
        st.subheader("Statistical Test Configuration")
        
        # Select Test Type
        test_type = st.radio(
            "Select Statistical Test",
            ["Normality Test (Shapiro-Wilk)", "Chi-Square Test"],
            key="stat_test_type"
        )

        # Configuration for Normality Test
        if test_type == "Normality Test (Shapiro-Wilk)":
            selected_column = st.selectbox(
                "Select a Numerical Column for Testing:",
                numerical_columns,
                key="normality_col"
            )

        # Configuration for Chi-Square Test
        elif test_type == "Chi-Square Test":
            column_x = st.selectbox(
                "Select First Categorical Column (X):",
                categorical_columns,
                key="chi_square_x"
            )
            column_y = st.selectbox(
                "Select Second Categorical Column (Y):",
                categorical_columns,
                key="chi_square_y"
            )

        # Results in Center Column
        with col2:
            st.subheader("Statistical Test Results")
            if test_type == "Normality Test (Shapiro-Wilk)":
                if selected_column:
                    st.markdown(f"### Normality Test for `{selected_column}`")

                    # Perform Shapiro-Wilk Test
                    data_to_test = filtered_data[selected_column].dropna() if slicer_active else dataframe[selected_column].dropna()
                    stat, p_value = shapiro(data_to_test)
                    st.write(f"- **Statistic**: {stat:.4f}")
                    st.write(f"- **p-value**: {p_value:.4f}")

                    if p_value > 0.05:
                        st.success("The data appears to be normally distributed (fail to reject H0).")
                    else:
                        st.error("The data does not appear to be normally distributed (reject H0).")
                else:
                    st.warning("Please select a numerical column for the normality test.")

            elif test_type == "Chi-Square Test":
                if column_x and column_y:
                    st.markdown(f"### Chi-Square Test for `{column_x}` vs `{column_y}`")

                    # Create a contingency table
                    data_to_test = filtered_data if slicer_active else dataframe
                    contingency_table = pd.crosstab(data_to_test[column_x], data_to_test[column_y])

                    # Perform Chi-Square Test
                    from scipy.stats import chi2_contingency
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                    st.write(f"- **Chi-Square Statistic**: {chi2:.4f}")
                    st.write(f"- **Degrees of Freedom**: {dof}")
                    st.write(f"- **p-value**: {p_value:.4f}")
                    st.write("**Expected Frequencies:**")
                    st.dataframe(expected)

                    if p_value > 0.05:
                        st.success("The variables are likely independent (fail to reject H0).")
                    else:
                        st.error("The variables are not independent (reject H0).")
                else:
                    st.warning("Please select two categorical columns for the Chi-Square test.")
    else:
        st.info("Enable the statistical tests to proceed.")



        
        # Bivariate and Trivariate Analysis in Sidebar
with st.sidebar.expander("📈 Bivariate & Trivariate Analysis", expanded=True):
    st.header("📈 Bivariate & Trivariate Analysis")

    # Enable/Disable Analysis
    enable_bi_tri_analysis = st.checkbox("Enable Bivariate & Trivariate Analysis", key="enable_bi_tri_analysis")

    if enable_bi_tri_analysis:
        st.subheader("Analysis Configuration")

        # Bivariate Analysis Configuration
        st.markdown("### Bivariate Analysis")
        x_var = st.selectbox("Select X-axis Variable:", options=dataframe.columns, key="bivariate_x")
        y_var = st.selectbox("Select Y-axis Variable (Numerical):", options=numerical_columns, key="bivariate_y")
        bivariate_chart_type = st.selectbox(
            "Select Chart Type for Bivariate Analysis:",
            ["Scatter", "Line", "Bar"],
            key="bivariate_chart_type"
        )

        # Trivariate Analysis Configuration
        st.markdown("### Trivariate Analysis")
        enable_trivariate = st.checkbox("Enable Trivariate Analysis", key="enable_trivariate")
        z_var = st.selectbox(
            "Select Color/Category Variable for Trivariate Analysis (Optional):",
            options=categorical_columns,
            key="trivariate_z"
        ) if enable_trivariate else None

        # Results in Center Column
        with col2:
            st.subheader("Results")
            
            if x_var and y_var:
                # Title for Bivariate Analysis
                st.markdown(f"#### Bivariate Analysis: {x_var} vs {y_var}")

                # Generate Bivariate Chart
                if bivariate_chart_type == "Scatter":
                    fig_bi = px.scatter(
                        filtered_data if slicer_active else dataframe,
                        x=x_var, y=y_var,
                        title=f"{x_var} vs {y_var} (Scatter Plot)"
                    )
                elif bivariate_chart_type == "Line":
                    fig_bi = px.line(
                        filtered_data if slicer_active else dataframe,
                        x=x_var, y=y_var,
                        title=f"{x_var} vs {y_var} (Line Chart)"
                    )
                elif bivariate_chart_type == "Bar":
                    fig_bi = px.bar(
                        filtered_data if slicer_active else dataframe,
                        x=x_var, y=y_var,
                        title=f"{x_var} vs {y_var} (Bar Chart)"
                    )

                # Display Bivariate Chart
                st.plotly_chart(fig_bi, use_container_width=True)

                # Generate Trivariate Chart if enabled
                if enable_trivariate and z_var:
                    st.markdown(f"#### Trivariate Analysis: {x_var} vs {y_var} with {z_var}")
                    fig_tri = px.scatter(
                        filtered_data if slicer_active else dataframe,
                        x=x_var, y=y_var, color=z_var,
                        title=f"{x_var} vs {y_var} with {z_var}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_tri, use_container_width=True)
            else:
                st.warning("Please select both X and Y variables for Bivariate Analysis.")
    else:
        st.info("Enable the Bivariate & Trivariate Analysis to proceed.")


        # Interactive Data Filtering in Sidebar
with st.sidebar.expander("🔍 Interactive Data Filtering", expanded=True):
    st.header("🔍 Interactive Data Filtering")

    # Enable/Disable Filtering
    enable_filtering = st.checkbox("Enable Data Filtering", key="enable_filtering")

    if enable_filtering:
        st.subheader("Configure Filters")

        # Create filter sliders for numerical columns
        filters = {}
        for column in numerical_columns:
            min_val, max_val = float(dataframe[column].min()), float(dataframe[column].max())
            if min_val < max_val:  # Ensure valid range for slider
                filters[column] = st.slider(
                    f"Filter Range for {column}:",
                    min_val,
                    max_val,
                    (min_val, max_val),
                    key=f"filter_{column}"
                )

        # Results in Center Column
        with col2:
            st.subheader("Filtered Data Results")
            if filters:
                # Apply filters to dataframe
                filtered_df = dataframe.copy()
                for column, (min_val, max_val) in filters.items():
                    filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]

                # Display filtered data
                st.markdown("### Filtered Data Preview")
                st.dataframe(filtered_df, use_container_width=True)

                # Save Filtered Data Option
                if st.checkbox("Save Filtered Data", key="save_filtered_data"):
                    st.session_state.saved_charts.append(("Filtered Data", filtered_df))
                    st.success("Filtered data saved successfully!")
            else:
                st.info("Adjust the sliders to apply filters and view results.")
    else:
        st.info("Enable the Interactive Data Filtering to configure and apply filters.")


        

        # Outlier Detection in Sidebar
with st.sidebar.expander("🚨 Outlier Detection", expanded=True):
    st.header("🚨 Outlier Detection")

    # Enable/Disable Outlier Detection
    enable_outlier_detection = st.checkbox("Enable Outlier Detection", key="enable_outlier_detection")

    if enable_outlier_detection:
        st.subheader("Configure Outlier Detection")

        # Select column for outlier detection
        selected_outlier_col = st.selectbox(
            "Select Numerical Column for Outlier Detection:",
            numerical_columns,
            key="outlier_column"
        )

        # Choose outlier detection method
        outlier_method = st.selectbox(
            "Select Outlier Detection Method:",
            ["Z-Score", "IQR (Interquartile Range)", "Modified Z-Score"],
            key="outlier_method"
        )

        # Input thresholds for outlier methods
        if outlier_method == "Z-Score":
            z_threshold = st.number_input(
                "Z-Score Threshold:",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.1,
                key="z_threshold"
            )
        elif outlier_method == "IQR (Interquartile Range)":
            iqr_multiplier = st.number_input(
                "IQR Multiplier:",
                min_value=1.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
                key="iqr_multiplier"
            )
        elif outlier_method == "Modified Z-Score":
            mz_threshold = st.number_input(
                "Modified Z-Score Threshold:",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.1,
                key="mz_threshold"
            )

        # Results in Center Column
        with col2:
            st.header("Outlier Detection Results")

            if selected_outlier_col:
                st.markdown(f"### Outliers in `{selected_outlier_col}` using {outlier_method} Method")

                # Perform outlier detection
                outliers = pd.DataFrame()
                data_to_check = filtered_data if slicer_active else dataframe

                if outlier_method == "Z-Score":
                    z_scores = zscore(data_to_check[selected_outlier_col].dropna())
                    outliers = data_to_check[(z_scores < -z_threshold) | (z_scores > z_threshold)]

                elif outlier_method == "IQR (Interquartile Range)":
                    Q1 = data_to_check[selected_outlier_col].quantile(0.25)
                    Q3 = data_to_check[selected_outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    outliers = data_to_check[
                        (data_to_check[selected_outlier_col] < lower_bound) | 
                        (data_to_check[selected_outlier_col] > upper_bound)
                    ]

                elif outlier_method == "Modified Z-Score":
                    median = data_to_check[selected_outlier_col].median()
                    MAD = np.median(np.abs(data_to_check[selected_outlier_col] - median))
                    modified_z_scores = 0.6745 * (data_to_check[selected_outlier_col] - median) / MAD
                    outliers = data_to_check[
                        (modified_z_scores < -mz_threshold) | 
                        (modified_z_scores > mz_threshold)
                    ]

                # Display outliers
                if not outliers.empty:
                    st.subheader(f"Detected {len(outliers)} Outliers")
                    st.dataframe(outliers, use_container_width=True)
                else:
                    st.info("No outliers detected for the selected column and method.")
            else:
                st.warning("Please select a column for outlier detection.")


        # Pivot Table in Sidebar
with st.sidebar.expander("📊 Pivot Table", expanded=True):
    st.header("📊 Pivot Table Analysis")

    # Enable/Disable Pivot Table
    enable_pivot_table = st.checkbox("Enable Pivot Table", key="enable_pivot_table")

    if enable_pivot_table:
        st.subheader("Configure Pivot Table")

        # Pivot table index, values, and aggregation selection
        pivot_index = st.selectbox(
            "Select Index Column:",
            categorical_columns,
            key="pivot_index"
        )
        pivot_values = st.selectbox(
            "Select Values Column:",
            numerical_columns,
            key="pivot_values"
        )
        agg_func = st.selectbox(
            "Select Aggregation Function:",
            ["Sum", "Mean", "Count"],
            key="pivot_agg"
        )

        # Results in Center Column
        with col2:
            st.header("Pivot Table Results")

            if pivot_index and pivot_values and agg_func:
                # Use filtered data if slicer is active
                data_to_pivot = filtered_data if slicer_active else dataframe

                # Generate pivot table
                pivot_table = data_to_pivot.pivot_table(
                    index=pivot_index,
                    values=pivot_values,
                    aggfunc=agg_func.lower()
                )

                # Display pivot table
                st.subheader(f"Pivot Table ({agg_func} of {pivot_values} by {pivot_index})")
                st.dataframe(pivot_table, use_container_width=True)
            else:
                st.warning("Please select an index, values column, and aggregation function for the pivot table.")


        # Funneling Tool in Sidebar
with st.sidebar.expander("🔀 Funneling Tool", expanded=True):
    st.header("🔀 Funneling Analysis")

    # Enable/Disable Funneling
    enable_funneling = st.checkbox("Enable Funneling Tool", key="enable_funneling")

    if enable_funneling:
        st.subheader("Configure Funneling")

        # Select variables for funneling
        funnel_stage1 = st.selectbox(
            "Select Start Variable (Stage 1):",
            categorical_columns + numerical_columns,
            key="funnel_stage1"
        )
        funnel_stage2 = st.selectbox(
            "Select End Variable (Stage 2):",
            categorical_columns + numerical_columns,
            key="funnel_stage2"
        )

        # Optional filter for numerical columns
        if funnel_stage1 in numerical_columns:
            filter_stage1 = st.slider(
                f"Filter {funnel_stage1} Range:",
                min_value=float(dataframe[funnel_stage1].min()),
                max_value=float(dataframe[funnel_stage1].max()),
                value=(
                    float(dataframe[funnel_stage1].min()),
                    float(dataframe[funnel_stage1].max())
                ),
                key="filter_stage1"
            )

        # Results in Center Column
        with col2:
            st.header("Funneling Results")

            if funnel_stage1 and funnel_stage2:
                st.subheader(f"Funnel Analysis: {funnel_stage1} → {funnel_stage2}")

                # Use filtered data if slicer is active
                data_to_use = filtered_data if slicer_active else dataframe

                # Categorical funnel analysis
                if funnel_stage1 in categorical_columns and funnel_stage2 in categorical_columns:
                    funnel_data = data_to_use.groupby([funnel_stage1, funnel_stage2]).size().reset_index(name='Count')
                    fig_funnel = px.funnel(
                        funnel_data,
                        x="Count",
                        y=funnel_stage1,
                        color=funnel_stage2,
                        title=f"Funnel Analysis: {funnel_stage1} → {funnel_stage2}"
                    )
                    st.plotly_chart(fig_funnel, use_container_width=True)

                # Numerical funnel analysis with filters
                elif funnel_stage1 in numerical_columns and funnel_stage2 in numerical_columns:
                    filtered_data = data_to_use[
                        (data_to_use[funnel_stage1] >= filter_stage1[0]) &
                        (data_to_use[funnel_stage1] <= filter_stage1[1])
                    ]
                    st.subheader(f"Filtered Data for {funnel_stage1} Range: {filter_stage1[0]} to {filter_stage1[1]}")
                    st.dataframe(filtered_data)

                    # Aggregate and visualize progression
                    agg_result = filtered_data.groupby(funnel_stage2).size().reset_index(name="Count")
                    st.subheader(f"Aggregated Counts for {funnel_stage2} After Filtering {funnel_stage1}:")
                    st.dataframe(agg_result)

                    fig_progress = px.bar(
                        agg_result,
                        x=funnel_stage2,
                        y="Count",
                        title=f"Progression from {funnel_stage1} to {funnel_stage2}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_progress, use_container_width=True)

                else:
                    st.warning("Combination of selected variables is not supported for funneling.")
            else:
                st.warning("Please select both Stage 1 and Stage 2 variables for funneling.")

        
        # Clustering Analysis in Sidebar
with st.sidebar.expander("🌀 Clustering Analysis", expanded=True):
    st.header("🌀 Clustering Analysis")

    # Enable/Disable Analysis
    enable_clustering = st.checkbox("Enable Clustering Analysis", key="enable_clustering")

    if enable_clustering:
        st.subheader("Clustering Configuration")

        # Select features for clustering
        selected_features = st.multiselect(
            "Select Numerical Features for Clustering:",
            numerical_columns,
            key="clustering_features"
        )
        n_clusters = st.slider(
            "Select Number of Clusters (k):",
            min_value=2,
            max_value=10,
            value=3,
            key="num_clusters"
        )

        # Results in Center Column
        with col2:
            st.header("Clustering Results")

            if selected_features:
                try:
                    # Perform clustering
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(dataframe[selected_features])
                    dataframe["Cluster"] = clusters

                    # 2D Cluster Visualization
                    if len(selected_features) >= 2:
                        st.subheader("2D Cluster Visualization")
                        fig_cluster_2d = px.scatter(
                            dataframe,
                            x=selected_features[0],
                            y=selected_features[1],
                            color="Cluster",
                            title=f"Clusters Based on {selected_features[0]} and {selected_features[1]}",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_cluster_2d, use_container_width=True)

                    # Optional 3D Cluster Visualization
                    if len(selected_features) >= 3:
                        st.subheader("3D Cluster Visualization")
                        fig_cluster_3d = px.scatter_3d(
                            dataframe,
                            x=selected_features[0],
                            y=selected_features[1],
                            z=selected_features[2],
                            color="Cluster",
                            title=f"Clusters in 3D: {selected_features[:3]}",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_cluster_3d, use_container_width=True)

                    # Display Cluster Centroids
                    st.subheader("Cluster Centroids")
                    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
                    st.dataframe(centroids)

                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")

            else:
                st.warning("Please select at least two features for clustering.")

        # Time Series Analysis in Sidebar
with st.sidebar.expander("⏳ Time Series Analysis", expanded=True):
    st.header("⏳ Time Series Analysis")

    # Enable/Disable Analysis
    enable_time_series = st.checkbox("Enable Time Series Analysis", key="enable_time_series")

    if enable_time_series:
        st.subheader("Time Series Configuration")

        # Select time and value columns
        time_column = st.selectbox(
            "Select Time Column (e.g., Date):",
            options=dataframe.columns,
            key="time_series_time_column"
        )
        value_column = st.selectbox(
            "Select Value Column (e.g., Sales):",
            options=numerical_columns,
            key="time_series_value_column"
        )

        # Aggregation type selection
        aggregation_type = st.selectbox(
            "Select Aggregation Type:",
            ["Sum", "Mean"],
            key="time_series_aggregation_type"
        )

        # Date conversion option
        convert_to_datetime = st.checkbox("Convert Time Column to Datetime", key="convert_to_datetime")

        # Frequency selection
        frequency = st.selectbox(
            "Select Resampling Frequency:",
            ["Daily", "Weekly", "Monthly", "Yearly"],
            key="time_series_frequency"
        )

        # Results in Center Column
        with col2:
            st.header("Time Series Results")

            if time_column and value_column:
                try:
                    # Convert time column to datetime if required
                    if convert_to_datetime:
                        dataframe[time_column] = pd.to_datetime(dataframe[time_column], errors="coerce")
                        dataframe.dropna(subset=[time_column], inplace=True)

                    # Resample data based on frequency
                    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
                    resampled_data = dataframe.set_index(time_column).resample(freq_map[frequency])[value_column].agg(aggregation_type.lower())

                    # Display resampled data
                    st.subheader("Resampled Data Preview")
                    st.dataframe(resampled_data.reset_index())

                    # Time series plot
                    st.subheader(f"Time Series: {value_column} Over {frequency}")
                    fig_time_series = px.line(
                        resampled_data,
                        x=resampled_data.index,
                        y=resampled_data.values,
                        title=f"{value_column} ({aggregation_type}) Over {frequency}",
                        labels={"x": time_column, "y": value_column},
                        template="plotly_white"
                    )
                    fig_time_series.update_layout(xaxis_title=time_column, yaxis_title=value_column)
                    st.plotly_chart(fig_time_series, use_container_width=True)

                    

                except Exception as e:
                    st.error(f"Error in time series analysis: {e}")
            else:
                st.warning("Please select both a time column and a value column.")

