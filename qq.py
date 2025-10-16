
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimizer", layout="wide", page_icon="💰")

st.markdown("""
<style>
    .main {padding: 1.5rem;}
    .stButton>button {
        width: 100%;
        background: #667eea;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {background: #764ba2;}
    h1 {color: #667eea;}
    .budget-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .budget-card h2 {color: white; margin: 0; font-size: 2rem;}
    .budget-card p {color: #f0f0f0; margin: 0.5rem 0 0 0;}
    .highlight-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("💰 Budget Optimizer with CPM")
st.markdown("*Optimize budget allocation based on impressions and CPM*")
st.markdown("---")

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'custom_constraints' not in st.session_state:
    st.session_state.custom_constraints = {}

# Sidebar
with st.sidebar:
    st.header("📁 Upload Files")
    budget_file = st.file_uploader("Budget Allocation", type=['xlsx', 'csv'], key='budget')
    betas_file = st.file_uploader("Betas (Impressions)", type=['xlsx', 'csv'], key='betas', 
                                   help="Betas represent coefficient per 1000 impressions")
    cpm_file = st.file_uploader("CPM (Cost per 1000)", type=['xlsx', 'csv'], key='cpm',
                                help="Cost per 1000 impressions for each product")

    files_loaded = budget_file and betas_file and cpm_file
    if files_loaded:
        st.success("✅ All files loaded")
    else:
        missing = []
        if not budget_file: missing.append("Budget")
        if not betas_file: missing.append("Betas")
        if not cpm_file: missing.append("CPM")
        st.warning(f"⚠️ Missing: {', '.join(missing)}")

    st.markdown("---")
    with st.expander("📋 File Format Help"):
        st.code("Budget:\nProduct | Period1\nProduct A | 1500")
        st.code("Betas:\nProduct | Coefficients\nProduct A | 0.40\nIntercept | 100")
        st.code("CPM:\nProduct | CPM\nProduct A | 5.50")

if not files_loaded:
    st.info("👈 **Upload all 3 files in the sidebar to get started**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Budget File")
        st.markdown("Budget allocated per product/period")
    with col2:
        st.markdown("### 📈 Betas File")
        st.markdown("Coefficient per 1000 impressions")
    with col3:
        st.markdown("### 💵 CPM File")
        st.markdown("Cost per 1000 impressions")
    st.stop()

try:
    # Read files
    df_budget = pd.read_excel(budget_file) if budget_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(budget_file)
    df_betas = pd.read_excel(betas_file) if betas_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(betas_file)
    df_cpm = pd.read_excel(cpm_file) if cpm_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(cpm_file)

    budget_prod_col = df_budget.columns[0]
    betas_prod_col = df_betas.columns[0]
    betas_coef_col = df_betas.columns[1]
    cpm_prod_col = df_cpm.columns[0]
    cpm_value_col = df_cpm.columns[1]

    periods = [c for c in df_budget.columns if c != budget_prod_col]

    # Extract intercept
    df_betas_no_nan = df_betas[df_betas[betas_prod_col].notna()].copy()
    intercept_row = df_betas_no_nan[df_betas_no_nan[betas_prod_col].astype(str).str.lower().str.contains('intercept', na=False)]
    intercept = 0
    if len(intercept_row) > 0:
        try:
            intercept = float(intercept_row[betas_coef_col].iloc[0])
        except:
            intercept = 0

    # Two tabs
    tab1, tab2 = st.tabs(["⚙️ Configure & Optimize", "📊 Results"])

    # TAB 1: CONFIG + OPTIMIZE
    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### 📅 Select Time Period")
            period = st.selectbox("Choose period", periods, label_visibility="collapsed")

            original_total = pd.to_numeric(df_budget[period], errors='coerce').sum()

            st.markdown("### 💵 Budget Settings")

            # Budget adjustment slider
            budget_change_pct = st.slider(
                "Adjust Total Budget (%)",
                -50, 50, 0, 5,
                help="Increase or decrease total budget. 0% = keep original budget"
            )

            adjusted_total = original_total * (1 + budget_change_pct / 100.0)
            budget_diff = adjusted_total - original_total

            # Show budget comparison
            budget_col1, budget_col2, budget_col3 = st.columns(3)
            budget_col1.metric("Original Budget", f"${original_total:,.0f}")
            budget_col2.metric(
                "Adjusted Budget", 
                f"${adjusted_total:,.0f}",
                delta=f"{budget_change_pct:+.0f}%" if budget_change_pct != 0 else "No change"
            )
            budget_col3.metric(
                "Difference",
                f"${abs(budget_diff):,.0f}",
                delta="Increase" if budget_diff > 0 else ("Decrease" if budget_diff < 0 else "Same")
            )

            if budget_change_pct != 0:
                st.markdown(f"""
                <div class="highlight-box">
                    <strong>📌 Budget Change:</strong> You are {'increasing' if budget_change_pct > 0 else 'decreasing'} 
                    the total budget by {abs(budget_change_pct)}% (${abs(budget_diff):,.0f})
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
                <strong>ℹ️ How it works:</strong> Budget is converted to impressions using CPM, 
                then optimized based on betas (coefficient per 1000 impressions), 
                then converted back to budget.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Additional Info")
            info_col1, info_col2 = st.columns(2)
            info_col1.metric("Products", len(df_budget))
            info_col2.metric("Intercept", f"{intercept:,.0f}")

        with col_right:
            st.markdown("### 🎯 Constraints")
            constraint_pct = st.slider(
                "Per-Product Change Limit (%)",
                0, 100, 25, 5,
                help="How much each product can change from its current allocation"
            )

            st.info(f"**Default:** ±{constraint_pct}% per product")

            st.markdown("---")
            st.markdown("**Summary:**")
            st.markdown(f"• Total Budget: ${adjusted_total:,.0f}")
            st.markdown(f"• Product Limit: ±{constraint_pct}%")

        st.markdown("---")

        # Advanced constraints
        with st.expander("🔧 Custom Per-Product Constraints (Optional)"):
            st.caption("Override default constraint for specific products")

            products_list = df_budget[budget_prod_col].tolist()
            search = st.text_input("🔍 Search product", "", key="search_prod")
            filtered = [p for p in products_list if search.lower() in str(p).lower()] if search else products_list[:10]

            if search:
                st.caption(f"Found {len([p for p in products_list if search.lower() in str(p).lower()])} products")
            else:
                st.caption("Showing first 10 products. Use search to find more.")

            col1, col2 = st.columns(2)
            mid = len(filtered) // 2

            with col1:
                for prod in filtered[:mid]:
                    val = st.number_input(
                        prod[:45] + "..." if len(prod) > 45 else prod,
                        0, 100,
                        st.session_state.custom_constraints.get(prod, constraint_pct),
                        5,
                        key=f"c1_{prod}"
                    )
                    st.session_state.custom_constraints[prod] = val

            with col2:
                for prod in filtered[mid:]:
                    val = st.number_input(
                        prod[:45] + "..." if len(prod) > 45 else prod,
                        0, 100,
                        st.session_state.custom_constraints.get(prod, constraint_pct),
                        5,
                        key=f"c2_{prod}"
                    )
                    st.session_state.custom_constraints[prod] = val

        # Data preview
        with st.expander("📋 View Uploaded Data"):
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            preview_col1.markdown("**Budget**")
            preview_col1.dataframe(df_budget, use_container_width=True, height=200)
            preview_col2.markdown("**Betas**")
            preview_col2.dataframe(df_betas, use_container_width=True, height=200)
            preview_col3.markdown("**CPM**")
            preview_col3.dataframe(df_cpm, use_container_width=True, height=200)

        st.markdown("---")

        # Optimize button
        st.markdown("### 🚀 Run Optimization")

        opt_col1, opt_col2, opt_col3 = st.columns([1, 2, 1])
        with opt_col2:
            if st.button("OPTIMIZE BUDGET", use_container_width=True, type="primary"):
                with st.spinner("⚡ Optimizing with CPM conversion..."):
                    try:
                        # Clean betas
                        df_betas_clean = df_betas[df_betas[betas_prod_col].notna()].copy()
                        df_betas_clean = df_betas_clean[~df_betas_clean[betas_prod_col].astype(str).str.lower().str.contains('intercept', na=False)]

                        def fix_coef(x):
                            try:
                                s = str(x).strip()
                                if not s or s == 'nan':
                                    return 0.0
                                if s.count('.') > 1:
                                    s = s.replace('.', '', s.count('.')-1)
                                return float(s)
                            except:
                                return 0.0

                        df_betas_clean[betas_coef_col] = df_betas_clean[betas_coef_col].apply(fix_coef)

                        # Clean CPM
                        df_cpm_clean = df_cpm[df_cpm[cpm_prod_col].notna()].copy()
                        df_cpm_clean[cpm_value_col] = df_cpm_clean[cpm_value_col].apply(fix_coef)

                        # Merge all three
                        merged = df_budget[[budget_prod_col, period]].merge(
                            df_betas_clean[[betas_prod_col, betas_coef_col]],
                            left_on=budget_prod_col,
                            right_on=betas_prod_col,
                            how='inner'
                        ).merge(
                            df_cpm_clean[[cpm_prod_col, cpm_value_col]],
                            left_on=budget_prod_col,
                            right_on=cpm_prod_col,
                            how='inner'
                        )

                        if merged.empty:
                            st.error("❌ No matching products across all files!")
                            st.stop()

                        products = merged[budget_prod_col].values
                        baseline_budget = pd.to_numeric(merged[period], errors='coerce').fillna(0).values.astype(float)
                        betas = merged[betas_coef_col].values.astype(float)
                        cpm = merged[cpm_value_col].values.astype(float)

                        # Avoid division by zero
                        cpm = np.where(cpm == 0, 0.01, cpm)

                        # Convert budget to impressions (in thousands)
                        baseline_impressions = baseline_budget / cpm

                        # Use adjusted total budget
                        total_budget = float(adjusted_total)

                        # Apply constraints on impressions
                        lb_array = []
                        ub_array = []
                        constraint_used = []

                        for prod, base_imp in zip(products, baseline_impressions):
                            pct = st.session_state.custom_constraints.get(prod, constraint_pct) / 100.0
                            lb_array.append(max(base_imp * (1 - pct), 0))
                            ub_array.append(base_imp * (1 + pct))
                            constraint_used.append(pct * 100)

                        lb = np.array(lb_array)
                        ub = np.array(ub_array)

                        # Check feasibility
                        min_possible_budget = (lb * cpm).sum()
                        max_possible_budget = (ub * cpm).sum()

                        if total_budget < min_possible_budget - 1e-6:
                            st.error(f"❌ Target budget ${total_budget:,.0f} is too low. Minimum possible: ${min_possible_budget:,.0f}")
                            st.info("Try increasing the budget or increasing the constraint percentages.")
                            st.stop()
                        elif total_budget > max_possible_budget + 1e-6:
                            st.error(f"❌ Target budget ${total_budget:,.0f} is too high. Maximum possible: ${max_possible_budget:,.0f}")
                            st.info("Try decreasing the budget or increasing the constraint percentages.")
                            st.stop()

                        # Optimize on impressions
                        n = len(betas)
                        A_eq = cpm.reshape(1, -1)
                        b_eq = np.array([total_budget])

                        result = linprog(
                            c=-betas,
                            A_eq=A_eq,
                            b_eq=b_eq,
                            bounds=list(zip(lb, ub)),
                            method='highs',
                            options={'presolve': True}
                        )

                        if result.success:
                            optimized_impressions = result.x
                            optimized_budget = optimized_impressions * cpm

                            vol_baseline = intercept + (betas * baseline_impressions).sum()
                            vol_opt = intercept + (betas * optimized_impressions).sum()
                            lift = vol_opt - vol_baseline
                            lift_pct = (lift / vol_baseline * 100) if vol_baseline > 0 else 0

                            results = pd.DataFrame({
                                'Product': products,
                                'Original Budget': baseline_budget.round(0).astype(int),
                                'Optimized Budget': optimized_budget.round(0).astype(int),
                                'Budget Change': (optimized_budget - baseline_budget).round(0).astype(int),
                                'Budget Change %': ((optimized_budget - baseline_budget) / np.where(baseline_budget == 0, 1, baseline_budget) * 100).round(2),
                                'CPM': cpm.round(2),
                                'Original Impr (000s)': baseline_impressions.round(1),
                                'Optimized Impr (000s)': optimized_impressions.round(1),
                                'Constraint': [f"±{c:.0f}%" for c in constraint_used],
                                'Beta': betas.round(3),
                                'Original Volume': (betas * baseline_impressions).round(0).astype(int),
                                'Optimized Volume': (betas * optimized_impressions).round(0).astype(int)
                            })

                            st.session_state.results = {
                                'df': results,
                                'vol_baseline': vol_baseline,
                                'vol_opt': vol_opt,
                                'lift': lift,
                                'lift_pct': lift_pct,
                                'period': period,
                                'budget_original': original_total,
                                'budget_adjusted': adjusted_total,
                                'budget_optimized': optimized_budget.sum(),
                                'budget_change_pct': budget_change_pct,
                                'intercept': intercept
                            }

                            st.success("✅ Optimization completed! Check the **Results** tab →")
                            st.balloons()
                        else:
                            st.error(f"❌ Optimization failed: {result.message}")
                            st.info("This may be due to infeasible constraints. Try increasing the constraint percentage or adjusting the budget.")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("Details"):
                            import traceback
                            st.code(traceback.format_exc())

    # TAB 2: RESULTS
    with tab2:
        if st.session_state.results is None:
            st.warning("⚠️ No results yet. Please run optimization first.")
            st.info("👈 Go to the **Configure & Optimize** tab and click the **OPTIMIZE BUDGET** button")
            st.stop()

        r = st.session_state.results
        df = r['df']

        # Budget cards
        st.markdown("### 💰 Budget & Volume Summary")
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)

        with card_col1:
            st.markdown(f"""
            <div class="budget-card">
                <p>Original Budget</p>
                <h2>${r['budget_original']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with card_col2:
            budget_change_color = "#ffc107" if r['budget_change_pct'] != 0 else "#667eea"
            st.markdown(f"""
            <div class="budget-card" style="background: linear-gradient(135deg, {budget_change_color} 0%, {budget_change_color}cc 100%);">
                <p>Target Budget</p>
                <h2>${r['budget_adjusted']:,.0f}</h2>
                <p>{r['budget_change_pct']:+.0f}% from original</p>
            </div>
            """, unsafe_allow_html=True)

        with card_col3:
            st.markdown(f"""
            <div class="budget-card">
                <p>Original Volume</p>
                <h2>{r['vol_baseline']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with card_col4:
            st.markdown(f"""
            <div class="budget-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                <p>Optimized Volume</p>
                <h2>{r['vol_opt']:,.0f}</h2>
                <p style="color: white; font-weight: bold;">+{r['lift']:,.0f} ({r['lift_pct']:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Table
        st.markdown("### 📊 Detailed Allocation with CPM & Impressions")

        total_row = pd.DataFrame({
            'Product': ['TOTAL'],
            'Original Budget': [df['Original Budget'].sum()],
            'Optimized Budget': [df['Optimized Budget'].sum()],
            'Budget Change': [df['Budget Change'].sum()],
            'Budget Change %': ['—'],
            'CPM': ['—'],
            'Original Impr (000s)': [df['Original Impr (000s)'].sum()],
            'Optimized Impr (000s)': [df['Optimized Impr (000s)'].sum()],
            'Constraint': ['—'],
            'Beta': ['—'],
            'Original Volume': [df['Original Volume'].sum()],
            'Optimized Volume': [df['Optimized Volume'].sum()]
        })

        df_display = pd.concat([df, total_row], ignore_index=True)

        st.dataframe(
            df_display.style.apply(
                lambda x: ['background-color: #e6f3ff; font-weight: bold' if x.name == len(df_display)-1 else '' for i in x],
                axis=1
            ),
            use_container_width=True,
            height=400
        )

        # Downloads
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv = df.to_csv(index=False)
            st.download_button("📥 CSV", csv, f"optimized_{r['period']}.csv", "text/csv", use_container_width=True)

        with dl_col2:
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine='xlsxwriter') as w:
                df.to_excel(w, 'Results', index=False)
                pd.DataFrame({
                    'Metric': ['Original Budget', 'Target Budget', 'Budget Change %', 'Optimized Budget',
                              'Original Volume', 'Optimized Volume', 'Volume Lift', 'Lift %', 'Intercept'],
                    'Value': [r['budget_original'], r['budget_adjusted'], r['budget_change_pct'], 
                             r['budget_optimized'], r['vol_baseline'], r['vol_opt'], 
                             r['lift'], r['lift_pct'], r['intercept']]
                }).to_excel(w, 'Summary', index=False)
            st.download_button("📥 Excel", bio.getvalue(), f"optimized_{r['period']}.xlsx", 
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              use_container_width=True)

        st.markdown("---")

        # Charts
        with st.expander("📊 Budget Charts", expanded=True):
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig = go.Figure([
                    go.Bar(name='Original', x=df['Product'], y=df['Original Budget'], marker_color='#667eea'),
                    go.Bar(name='Optimized', x=df['Product'], y=df['Optimized Budget'], marker_color='#764ba2')
                ])
                fig.update_layout(title="Budget Allocation", barmode='group', xaxis_tickangle=-45, height=380)
                st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                colors = ['#28a745' if x > 0 else '#dc3545' for x in df['Budget Change']]
                fig2 = go.Figure(go.Bar(x=df['Product'], y=df['Budget Change'], marker_color=colors))
                fig2.update_layout(title="Budget Changes", xaxis_tickangle=-45, height=380)
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📈 Impressions Analysis", expanded=False):
            fig3 = go.Figure([
                go.Bar(name='Original', x=df['Product'], y=df['Original Impr (000s)'], marker_color='lightblue'),
                go.Bar(name='Optimized', x=df['Product'], y=df['Optimized Impr (000s)'], marker_color='darkblue')
            ])
            fig3.update_layout(title="Impressions (000s)", barmode='group', xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📊 Volume Analysis", expanded=False):
            fig4 = go.Figure([
                go.Bar(name='Original', x=df['Product'], y=df['Original Volume'], marker_color='lightcoral'),
                go.Bar(name='Optimized', x=df['Product'], y=df['Optimized Volume'], marker_color='darkgreen')
            ])
            fig4.update_layout(title="Volume Contribution", barmode='group', xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("🏆 Top Movers", expanded=False):
            mover_col1, mover_col2 = st.columns(2)
            with mover_col1:
                st.markdown("**💚 Top 5 Budget Increases**")
                st.dataframe(df.nlargest(5, 'Budget Change')[['Product', 'Budget Change', 'Budget Change %']].reset_index(drop=True), use_container_width=True)
            with mover_col2:
                st.markdown("**💔 Top 5 Budget Decreases**")
                st.dataframe(df.nsmallest(5, 'Budget Change')[['Product', 'Budget Change', 'Budget Change %']].reset_index(drop=True), use_container_width=True)

except Exception as e:
    st.error(f"❌ Error loading files: {str(e)}")
    with st.expander("Error Details"):
        import traceback
        st.code(traceback.format_exc())
