import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

from src.pipeline import run_pipeline

st.set_page_config(page_title="AI Insight & Decision Engine", layout="wide", page_icon="⚡")


@st.cache_data
def generate_sample_data(n_rows: int = 1500) -> pd.DataFrame:
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=45)

    dates = [start_date + timedelta(minutes=int(x)) for x in np.random.randint(0, 45 * 24 * 60, n_rows)]
    customers = [f"CUST_{i:04d}" for i in np.random.randint(1, 100, n_rows)]
    locations = np.random.choice(["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune"], n_rows)

    amounts = np.random.exponential(scale=150.0, size=n_rows)
    # Inject outliers
    outlier_indices = np.random.choice(n_rows, size=20, replace=False)
    amounts[outlier_indices] = amounts[outlier_indices] * 20.0

    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i:05d}" for i in range(n_rows)],
        "timestamp": dates,
        "customer_id": customers,
        "location": locations,
        "amount": np.round(amounts, 2),
    }).sort_values("timestamp").reset_index(drop=True)

    # Inject deliberate surge on recent date
    recent_date = df["timestamp"].max().date()
    surge_mask = df["timestamp"].dt.date == recent_date
    df.loc[surge_mask, "amount"] = df.loc[surge_mask, "amount"] * 3.5

    return df


st.title("⚡ AI-Driven Decision & Insight Engine")
st.caption("Automated pattern detection $\\rightarrow$ Agent prioritization $\\rightarrow$ GenAI executive narrative")

# Sidebar Configuration
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Select Data Source:", ["Synthetic Demo Data", "Upload CSV"])
enable_genai = st.sidebar.checkbox("Enable GenAI Summarization", value=False)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.info("Awaiting file upload. Using default demo dataset.")
        raw_df = generate_sample_data()
else:
    raw_df = generate_sample_data()

# Run Pipeline
with st.spinner("Analyzing signals, filtering patterns, and prompting agent..."):
    result = run_pipeline(raw_df, generate_llm=enable_genai)

# Section 1: Overview KPIs
st.subheader("1. Key Operational Metrics")
col1, col2, col3, col4 = st.columns(4)

total_txns = len(result.enriched_df)
anomaly_cnt = int(result.enriched_df["is_anomaly"].sum())
cluster_cnt = int(result.enriched_df["cluster_id"].nunique())
critical_alerts = sum(1 for i in result.insights if i.priority.value == "CRITICAL")

col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Detected Anomalies", f"{anomaly_cnt} ({(anomaly_cnt/total_txns)*100:.1f}%)")
col3.metric("Customer Segments", f"{cluster_cnt}")
col4.metric("Critical Alerts", f"{critical_alerts}")

st.divider()

# Section 4: AI Strategic Explanation (Displayed prominently at the top)
st.subheader("2. GenAI Executive Briefing")
if result.ai_summary:
    st.info(result.ai_summary)
else:
    st.write("GenAI summarization is toggled off or pending an API key.")

st.divider()

# Section 3: Agentic Priorities
st.subheader("3. Agent Prioritized Action Feed")
if result.insights:
    for item in result.insights:
        color = "🔴" if item.priority.value == "CRITICAL" else ("🟡" if item.priority.value == "MEDIUM" else "🟢")
        with st.expander(f"{color} [{item.priority.value}] {item.candidate.title} (Importance Score: {item.score}/100)"):
            st.write(f"**Description:** {item.candidate.description}")
            st.write(f"**Recommended Action:** `{item.recommended_action}`")
            st.json(item.candidate.raw_metrics)
else:
    st.success("No anomalies or major variance detected exceeding agent action thresholds.")

st.divider()

# Section 2: Visualizations
st.subheader("4. Analytical Visualizations")
tab1, tab2, tab3 = st.tabs(["Daily Volume Trends", "Anomaly Scatter", "Customer Segments"])

with tab1:
    fig_trend = px.line(
        result.trend_df,
        x="date",
        y="daily_amount",
        title="Daily Transaction Value with Spikes Highlighted",
        labels={"daily_amount": "Total Amount ($)", "date": "Date"}
    )
    spikes = result.trend_df[result.trend_df["is_spike"] == 1]
    if not spikes.empty:
        fig_trend.add_scatter(
            x=spikes["date"],
            y=spikes["daily_amount"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Spike (Z > 2.0)"
        )
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    fig_anom = px.scatter(
        result.enriched_df,
        x="timestamp",
        y="amount",
        color=result.enriched_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
        color_discrete_map={"Normal": "#1f77b4", "Anomaly": "#d62728"},
        title="Transaction Value Outliers (Isolation Forest)",
        hover_data=["customer_id", "location"]
    )
    st.plotly_chart(fig_anom, use_container_width=True)

with tab3:
    fig_clust = px.box(
        result.enriched_df,
        x="cluster_id",
        y="amount",
        color="cluster_id",
        title="Spend Distribution across Discovered Segments"
    )
    st.plotly_chart(fig_clust, use_container_width=True)