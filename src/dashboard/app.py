import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set up the dashboard page configuration
st.set_page_config(
    page_title="Customer Churn & Retention Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional SaaS UI supporting both Light and Dark themes
st.markdown("""
<style>
    /* Typography and generic setup */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600;
        letter-spacing: -0.025em;
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(127, 127, 127, 0.05) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(127, 127, 127, 0.2);
    }
    
    /* Hide top padding for a cleaner edge-to-edge feel */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom divider */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 0;
        border-top: 1px solid rgba(127, 127, 127, 0.2);
    }
    
    /* Fix metrics overlap styling if native metrics were used */
    div[data-testid="metric-container"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the processed PowerBI export data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_dir, 'data', 'processed', 'powerbi_export.csv')
    df = pd.read_csv(file_path)
    if 'last_purchase_date' in df.columns:
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'data/processed/powerbi_export.csv' exists.")
    st.stop()


# ==========================================
# SIDEBAR - FILTERS
# ==========================================
st.sidebar.markdown("""
<div style="margin-bottom: 16px;">
    <h2 style="margin-bottom: 4px; font-size: 20px;">Filters</h2>
    <p style="color: var(--text-color); opacity: 0.7; font-size: 13px; margin-top: 0;">Adjust criteria to filter dashboard</p>
</div>
""", unsafe_allow_html=True)

gender_options = sorted(df['gender'].dropna().unique().tolist())
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

membership_options = sorted(df['membership_type'].dropna().unique().tolist())
selected_membership = st.sidebar.multiselect("Membership Type", membership_options, default=membership_options)

segment_options = sorted(df['rfm_segment'].dropna().unique().tolist())
selected_segment = st.sidebar.multiselect("Customer Segment (RFM)", segment_options, default=segment_options)

filtered_df = df[
    (df['gender'].isin(selected_gender)) &
    (df['membership_type'].isin(selected_membership)) &
    (df['rfm_segment'].isin(selected_segment))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your criteria.")
    st.stop()


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown("""
<div style="margin-bottom: 32px;">
    <h1 style="margin-bottom: 8px; font-size: 32px;">Customer Churn & Retention Analytics</h1>
    <p style="color: var(--text-color); opacity: 0.7; font-size: 16px; margin-top: 0;">Interactive performance overview based on final machine learning pipeline results.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# CUSTOM KPI CARDS (HTML/CSS)
# ==========================================
def render_kpi(title, value, icon_svg, color):
    return f"""
    <div style="
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(127, 127, 127, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        cursor: default;
        "
        onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05)';"
        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.02)';"
    >
        <div style="
            background-color: {color}25;
            border-radius: 12px;
            width: 56px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 20px;
            color: {color};
            flex-shrink: 0;
        ">
            {icon_svg}
        </div>
        <div>
            <div style="color: var(--text-color); opacity: 0.7; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{title}</div>
            <div style="color: var(--text-color); font-size: 28px; font-weight: 700; line-height: 1;">{value}</div>
        </div>
    </div>
    """

# Professional SVGs for icons
icon_users = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
icon_trend = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline><polyline points="17 18 23 18 23 12"></polyline></svg>'
icon_dollar = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
icon_target = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'

# Calculate KPI values
total_customers = len(filtered_df)
churn_rate = (filtered_df['churn'].sum() / total_customers) * 100 if total_customers > 0 else 0
avg_order_value = filtered_df['average_order_value'].mean()
predicted_churn = filtered_df['predicted_churn'].sum()

# Render KPI Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(render_kpi("Total Customers", f"{total_customers:,}", icon_users, "#3B82F6"), unsafe_allow_html=True)
with k2:
    st.markdown(render_kpi("Churn Rate", f"{churn_rate:.1f}%", icon_trend, "#EF4444"), unsafe_allow_html=True)
with k3:
    st.markdown(render_kpi("Avg Order Value", f"${avg_order_value:.2f}", icon_dollar, "#10B981"), unsafe_allow_html=True)
with k4:
    st.markdown(render_kpi("Predicted Churn", f"{predicted_churn:,}", icon_target, "#F59E0B"), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# ==========================================
# CHART CONFIG & COLORS
# ==========================================
# Premium pastel palette (keeping colors for bars/lines as they are brand colors)
COLOR_RETAINED = '#A1C6EA'  # Pastel blue
COLOR_CHURNED = '#F6A8A8'   # Soft coral
COLOR_REVENUE = '#98D8A8'   # Pastel green

# Minimal layout config - Streamlit will automatically handle background and text colors via `theme="streamlit"`
chart_layout_config = {
    'margin': dict(t=50, b=30, l=10, r=10),
}
chart_display_config = {'displayModeBar': False}


# ==========================================
# ROW 1 CHARTS
# ==========================================
r1c1, r1c2, r1c3 = st.columns(3)

# Chart 1: Churn Distribution (Donut)
churn_counts = filtered_df['churn'].value_counts().reset_index()
churn_counts.columns = ['Churn Status', 'Count']
churn_counts['Churn Status'] = churn_counts['Churn Status'].map({0: 'Retained', 1: 'Churned'})

fig_churn = px.pie(
    churn_counts, 
    values='Count', 
    names='Churn Status', 
    hole=0.65,
    color='Churn Status',
    color_discrete_map={'Retained': COLOR_RETAINED, 'Churned': COLOR_CHURNED},
    title='Overall Churn Distribution'
)
fig_churn.update_traces(
    hoverinfo='label+percent', 
    textinfo='none', 
    marker=dict(line=dict(color='rgba(0,0,0,0)', width=0))
)
fig_churn.update_layout(
    **chart_layout_config, 
    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    title_font_size=16, title_x=0.0
)
r1c1.plotly_chart(fig_churn, use_container_width=True, config=chart_display_config, theme="streamlit")


# Chart 2: Customer Segments
segment_counts = filtered_df['rfm_segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

fig_segment = px.bar(
    segment_counts,
    x='Count',
    y='Segment',
    orientation='h',
    title='Customers by RFM Segment',
    color_discrete_sequence=[COLOR_RETAINED]
)
fig_segment.update_layout(
    **chart_layout_config,
    yaxis={'categoryorder': 'total ascending', 'title': ''},
    xaxis={'title': ''},
    title_font_size=16, title_x=0.0
)
r1c2.plotly_chart(fig_segment, use_container_width=True, config=chart_display_config, theme="streamlit")


# Chart 3: Monthly Revenue Trend
filtered_df['Month'] = filtered_df['last_purchase_date'].dt.to_period('M').astype(str)
monthly_rev = filtered_df.groupby('Month')['total_spend'].sum().reset_index()

fig_rev = px.line(
    monthly_rev,
    x='Month',
    y='total_spend',
    title='Monthly Revenue (Total Spend)',
    markers=True,
    color_discrete_sequence=[COLOR_REVENUE]
)
fig_rev.update_traces(
    line=dict(width=3), 
    marker=dict(size=8)
)
fig_rev.update_layout(
    **chart_layout_config,
    xaxis={'title': '', 'showgrid': False},
    yaxis={'title': ''},
    title_font_size=16, title_x=0.0
)
r1c3.plotly_chart(fig_rev, use_container_width=True, config=chart_display_config, theme="streamlit")

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# ==========================================
# ROW 2 CHARTS
# ==========================================
r2c1, r2c2 = st.columns(2)

# Chart 4: Churn by Membership Type
churn_mem = filtered_df.groupby(['membership_type', 'churn']).size().reset_index(name='Count')
churn_mem['churn'] = churn_mem['churn'].map({0: 'Retained', 1: 'Churned'})

fig_mem = px.bar(
    churn_mem,
    x='membership_type',
    y='Count',
    color='churn',
    barmode='group',
    color_discrete_map={'Retained': COLOR_RETAINED, 'Churned': COLOR_CHURNED},
    title='Churn by Membership Tier'
)
fig_mem.update_layout(
    **chart_layout_config,
    xaxis={'title': '', 'showgrid': False},
    yaxis={'title': ''},
    legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_font_size=16, title_x=0.0
)
r2c1.plotly_chart(fig_mem, use_container_width=True, config=chart_display_config, theme="streamlit")


# Chart 5: Prediction Summary
eng_churn = filtered_df.groupby('predicted_churn')['engagement_score'].mean().reset_index()
eng_churn['predicted_churn'] = eng_churn['predicted_churn'].map({0: 'Predicted Retained', 1: 'Predicted Churned'})

fig_eng = px.bar(
    eng_churn,
    x='predicted_churn',
    y='engagement_score',
    title='Average Engagement Score (Prediction Analysis)',
    color='predicted_churn',
    color_discrete_map={'Predicted Retained': COLOR_RETAINED, 'Predicted Churned': COLOR_CHURNED}
)
fig_eng.update_layout(
    **chart_layout_config,
    xaxis={'title': '', 'showgrid': False},
    yaxis={'title': ''},
    showlegend=False,
    title_font_size=16, title_x=0.0
)
r2c2.plotly_chart(fig_eng, use_container_width=True, config=chart_display_config, theme="streamlit")


# ==========================================
# DATA TABLE: HIGH RISK CUSTOMERS
# ==========================================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 16px;">
    <h2 style="margin-bottom: 4px; font-size: 24px; color: var(--text-color);">Top High-Risk Customers</h2>
    <p style="color: var(--text-color); opacity: 0.7; font-size: 14px; margin-top: 0;">Customers predicted to churn, ranked by total spend to prioritize retention marketing.</p>
</div>
""", unsafe_allow_html=True)

high_risk = filtered_df[filtered_df['predicted_churn'] == 1].copy()

if not high_risk.empty:
    display_cols = ['customer_id', 'rfm_segment', 'total_spend', 'loyalty_score', 'engagement_score', 'churn']
    # Sort by total_spend descending
    high_risk_sorted = high_risk.sort_values(by='total_spend', ascending=False)[display_cols]
    
    # Rename columns for cleaner display
    high_risk_sorted.columns = ['Customer ID', 'RFM Segment', 'Total Spend ($)', 'Loyalty Score', 'Engagement Score', 'Actual Churn']
    
    st.dataframe(
        high_risk_sorted.head(100),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No high-risk customers found in the current selection.")
