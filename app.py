import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "sales_pipeline.db"

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product",
    "category",
    "region",
    "quantity",
    "unit_price",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time Sales Data Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------
       GENERAL PAGE
    --------------------------------*/

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    body {
        background-color: #f7f9fc;
    }


    /* -------------------------------
       HERO
    --------------------------------*/

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #202638;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }


    /* -------------------------------
       SMALL LABEL
    --------------------------------*/

    .eyebrow {
        font-size: 13px;
        font-weight: 700;
        color: #2563eb;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }


    /* -------------------------------
       HERO DESCRIPTION
    --------------------------------*/

    .hero-description {
        font-size: 17px;
        line-height: 1.7;
        color: #555d6e;
        max-width: 850px;
    }


    /* -------------------------------
       PIPELINE CARDS
    --------------------------------*/

    .stage-number {
        font-size: 13px;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 8px;
    }

    .stage-icon {
        font-size: 30px;
        margin-bottom: 8px;
    }

    .stage-title {
        font-size: 17px;
        font-weight: 750;
        color: #202638;
        margin-bottom: 6px;
    }

    .stage-description {
        font-size: 13px;
        line-height: 1.5;
        color: #6b7280;
    }


    /* -------------------------------
       SECTION TITLES
    --------------------------------*/

    .section-title {
        font-size: 27px;
        font-weight: 800;
        color: #202638;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }


    /* -------------------------------
       FOOTER
    --------------------------------*/

    .footer {
        text-align: center;
        color: #8a91a0;
        font-size: 13px;
        padding-top: 40px;
        padding-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TEXT,
            rows_in INTEGER,
            rows_out INTEGER,
            revenue REAL,
            anomalies INTEGER
        )
        """
    )

    connection.commit()
    connection.close()


def save_pipeline_run(
    rows_in,
    rows_out,
    revenue,
    anomalies
):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO pipeline_runs
        (
            run_time,
            rows_in,
            rows_out,
            revenue,
            anomalies
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            rows_in,
            rows_out,
            revenue,
            anomalies,
        ),
    )

    connection.commit()
    connection.close()


def get_pipeline_history():

    connection = sqlite3.connect(DB_PATH)

    history = pd.read_sql_query(
        """
        SELECT
            id,
            run_time,
            rows_in,
            rows_out,
            revenue,
            anomalies
        FROM pipeline_runs
        ORDER BY id DESC
        """,
        connection,
    )

    connection.close()

    return history


# ============================================================
# COLUMN STANDARDIZATION
# ============================================================

def standardize_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return df


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(df):

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return True


# ============================================================
# ETL TRANSFORMATION
# ============================================================

def transform_data(df):

    df = standardize_columns(df)

    validate_columns(df)

    rows_before = len(df)

    # -----------------------------------------
    # DATE
    # -----------------------------------------

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    # -----------------------------------------
    # NUMERIC DATA
    # -----------------------------------------

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["unit_price"] = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    )

    # -----------------------------------------
    # REMOVE INVALID ESSENTIAL RECORDS
    # -----------------------------------------

    df = df.dropna(
        subset=[
            "order_id",
            "order_date",
            "customer_id",
            "product",
        ]
    )

    # -----------------------------------------
    # FILL MISSING QUANTITY
    # -----------------------------------------

    if df["quantity"].isna().any():

        median_quantity = df["quantity"].median()

        if pd.isna(median_quantity):
            median_quantity = 1

        df["quantity"] = df[
            "quantity"
        ].fillna(median_quantity)

    # -----------------------------------------
    # FILL MISSING PRICE
    # -----------------------------------------

    if df["unit_price"].isna().any():

        median_price = df["unit_price"].median()

        if pd.isna(median_price):
            median_price = 0

        df["unit_price"] = df[
            "unit_price"
        ].fillna(median_price)

    # -----------------------------------------
    # REMOVE NEGATIVE VALUES
    # -----------------------------------------

    df["quantity"] = df[
        "quantity"
    ].clip(lower=0)

    df["unit_price"] = df[
        "unit_price"
    ].clip(lower=0)

    # -----------------------------------------
    # CATEGORICAL CLEANING
    # -----------------------------------------

    categorical_columns = [
        "product",
        "category",
        "region",
    ]

    for col in categorical_columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # -----------------------------------------
    # REMOVE DUPLICATE ORDERS
    # -----------------------------------------

    df = df.drop_duplicates(
        subset=["order_id"],
        keep="first"
    )

    # -----------------------------------------
    # CALCULATE REVENUE
    # -----------------------------------------

    df["revenue"] = (
        df["quantity"]
        * df["unit_price"]
    )

    # -----------------------------------------
    # SORT DATA
    # -----------------------------------------

    df = df.sort_values(
        "order_date"
    ).reset_index(drop=True)

    rows_after = len(df)

    return (
        df,
        rows_before,
        rows_after
    )


# ============================================================
# MACHINE LEARNING ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):

    df = df.copy()

    features = [
        "quantity",
        "unit_price",
        "revenue",
    ]

    model_data = df[
        features
    ].fillna(0)

    # Not enough records
    if len(df) < 5:

        df["anomaly"] = False
        df["anomaly_score"] = 0.0

        return df

    model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42,
    )

    predictions = model.fit_predict(
        model_data
    )

    scores = model.decision_function(
        model_data
    )

    df["anomaly"] = (
        predictions == -1
    )

    df["anomaly_score"] = scores

    return df


# ============================================================
# FILE READER
# ============================================================

def read_uploaded_file(
    uploaded_file
):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        return pd.read_csv(
            uploaded_file
        )

    if file_name.endswith(".xlsx"):

        return pd.read_excel(
            uploaded_file
        )

    if file_name.endswith(".xls"):

        return pd.read_excel(
            uploaded_file
        )

    raise ValueError(
        "Unsupported file format. "
        "Please upload CSV or Excel."
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Pipeline Controls"
    )

    st.caption(
        "Upload a sales dataset to begin."
    )

    uploaded_file = st.file_uploader(
        "Upload sales dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )

    st.divider()

    st.markdown(
        "### 📋 Required Columns"
    )

    for column in REQUIRED_COLUMNS:

        st.code(
            column,
            language="text"
        )

    st.divider()

    st.info(
        "💡 You can use the included "
        "`sample_sales.csv` file to test "
        "the complete pipeline."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="eyebrow">DATA ENGINEERING • ETL • MACHINE LEARNING</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">📊 Real-Time Sales Data Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    'Transform raw sales data into reliable, actionable business intelligence.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-description">'
    'An end-to-end data engineering platform that performs data ingestion, '
    'validation, ETL transformation, revenue calculation, machine-learning '
    'anomaly detection, database storage and interactive sales analytics.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# FRONT PAGE / PIPELINE OVERVIEW
# ============================================================

if uploaded_file is None:

    st.markdown(
        '<div class="section-title">'
        '🚀 End-to-End Data Engineering Pipeline'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'From raw transactions to business insights — all in one pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------
    # PIPELINE STAGES
    # -----------------------------------------

    stages = [
        (
            "01",
            "📥",
            "Data Ingestion",
            "Load CSV or Excel sales datasets."
        ),
        (
            "02",
            "✅",
            "Data Validation",
            "Verify required columns and data structure."
        ),
        (
            "03",
            "🔄",
            "ETL Transformation",
            "Clean, standardize and transform raw data."
        ),
        (
            "04",
            "💰",
            "Revenue Engine",
            "Calculate transaction-level revenue."
        ),
        (
            "05",
            "🤖",
            "ML Detection",
            "Identify unusual sales transactions."
        ),
        (
            "06",
            "📈",
            "Analytics",
            "Generate interactive business insights."
        ),
    ]

    # First row
    row1 = st.columns(3)

    for index in range(3):

        number, icon, title, description = stages[index]

        with row1[index]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f'<div class="stage-number">'
                    f'STAGE {number}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="stage-icon">'
                    f'{icon}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="stage-title">'
                    f'{title}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="stage-description">'
                    f'{description}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.write("")

    # Second row
    row2 = st.columns(3)

    for index in range(3, 6):

        number, icon, title, description = stages[index]

        with row2[index - 3]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f'<div class="stage-number">'
                    f'STAGE {number}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="stage-icon">'
                    f'{icon}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="stage-title">'
                    f'{title}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="stage-description">'
                    f'{description}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.write("")


    # ========================================================
    # TECHNOLOGY SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🛠️ Technology Stack'
        '</div>',
        unsafe_allow_html=True,
    )

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        with st.container(border=True):
            st.markdown("### 🐍 Python")
            st.caption(
                "Core programming and pipeline logic"
            )

    with tech2:
        with st.container(border=True):
            st.markdown("### 🐼 Pandas")
            st.caption(
                "Data processing and ETL"
            )

    with tech3:
        with st.container(border=True):
            st.markdown("### 🤖 Scikit-learn")
            st.caption(
                "Machine-learning anomaly detection"
            )

    with tech4:
        with st.container(border=True):
            st.markdown("### 🗄️ SQLite")
            st.caption(
                "Pipeline execution history"
            )


    # ========================================================
    # CAPABILITIES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '💡 Platform Capabilities'
        '</div>',
        unsafe_allow_html=True,
    )

    capability1, capability2 = st.columns(2)

    with capability1:

        with st.container(border=True):

            st.markdown(
                "### 📊 Data Engineering"
            )

            st.markdown(
                """
                - 📥 CSV / Excel ingestion
                - 🔍 Automatic schema validation
                - 🧹 Missing-value handling
                - 🔄 Duplicate removal
                - 💰 Revenue calculation
                """
            )

    with capability2:

        with st.container(border=True):

            st.markdown(
                "### 📈 Analytics & ML"
            )

            st.markdown(
                """
                - 🤖 Isolation Forest anomaly detection
                - 📈 Revenue trend analysis
                - 🌎 Regional performance
                - 🏆 Product performance
                - 📥 Downloadable reports
                """
            )


    # ========================================================
    # START MESSAGE
    # ========================================================

    st.write("")

    st.info(
        "👈 Upload `sample_sales.csv` from the sidebar "
        "to run the complete data pipeline."
    )

    st.stop()


# ============================================================
# RUN PIPELINE
# ============================================================

try:

    raw_df = read_uploaded_file(
        uploaded_file
    )

    rows_in = len(raw_df)

    processed_df, rows_before, rows_after = (
        transform_data(raw_df)
    )

    processed_df = detect_anomalies(
        processed_df
    )

    total_revenue = float(
        processed_df["revenue"].sum()
    )

    anomaly_count = int(
        processed_df["anomaly"].sum()
    )

    save_pipeline_run(
        rows_in=rows_in,
        rows_out=rows_after,
        revenue=total_revenue,
        anomalies=anomaly_count,
    )

except Exception as error:

    st.error(
        f"❌ Pipeline error: {error}"
    )

    st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    "✅ Pipeline executed successfully!"
)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Pipeline Results'
    '</div>',
    unsafe_allow_html=True,
)

total_orders = len(
    processed_df
)

total_customers = (
    processed_df[
        "customer_id"
    ].nunique()
)

total_revenue = (
    processed_df[
        "revenue"
    ].sum()
)

anomaly_count = int(
    processed_df[
        "anomaly"
    ].sum()
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "📦 Total Orders",
        f"{total_orders:,}"
    )


with metric2:

    st.metric(
        "💰 Total Revenue",
        f"₹{total_revenue:,.0f}"
    )


with metric3:

    st.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )


with metric4:

    st.metric(
        "🚨 Anomalies",
        f"{anomaly_count:,}"
    )


# ============================================================
# TABS
# ============================================================

tab_data, tab_analytics, tab_anomalies, tab_pipeline, tab_export = (
    st.tabs(
        [
            "📋 Data",
            "📈 Analytics",
            "🚨 Anomalies",
            "⚙️ Pipeline",
            "📥 Export",
        ]
    )
)


# ============================================================
# DATA TAB
# ============================================================

with tab_data:

    st.markdown(
        "## 📋 Processed Sales Data"
    )

    st.caption(
        f"{len(processed_df):,} records after ETL processing."
    )

    display_df = processed_df.copy()

    display_df["order_date"] = (
        display_df[
            "order_date"
        ].dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ANALYTICS TAB
# ============================================================

with tab_analytics:

    st.markdown(
        "## 📈 Sales Analytics"
    )

    st.caption(
        "Interactive business intelligence generated from processed sales data."
    )

    # -----------------------------------------
    # DAILY REVENUE
    # -----------------------------------------

    daily_revenue = (
        processed_df
        .groupby(
            "order_date",
            as_index=False
        )["revenue"]
        .sum()
    )

    fig_daily = px.line(
        daily_revenue,
        x="order_date",
        y="revenue",
        markers=True,
        title="Daily Revenue Trend",
    )

    fig_daily.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True,
    )


    # -----------------------------------------
    # REGION
    # -----------------------------------------

    left_chart, right_chart = st.columns(2)

    with left_chart:

        region_revenue = (
            processed_df
            .groupby(
                "region",
                as_index=False
            )["revenue"]
            .sum()
            .sort_values(
                "revenue",
                ascending=False
            )
        )

        fig_region = px.bar(
            region_revenue,
            x="region",
            y="revenue",
            title="Revenue by Region",
            text_auto=".2s",
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True,
        )


    # -----------------------------------------
    # PRODUCTS
    # -----------------------------------------

    with right_chart:

        product_revenue = (
            processed_df
            .groupby(
                "product",
                as_index=False
            )["revenue"]
            .sum()
            .sort_values(
                "revenue",
                ascending=False
            )
            .head(10)
        )

        fig_product = px.bar(
            product_revenue,
            x="product",
            y="revenue",
            title="Top Products by Revenue",
            text_auto=".2s",
        )

        fig_product.update_layout(
            xaxis_tickangle=-35
        )

        st.plotly_chart(
            fig_product,
            use_container_width=True,
        )


    # -----------------------------------------
    # CATEGORY
    # -----------------------------------------

    category_revenue = (
        processed_df
        .groupby(
            "category",
            as_index=False
        )["revenue"]
        .sum()
    )

    fig_category = px.pie(
        category_revenue,
        names="category",
        values="revenue",
        title="Revenue by Category",
        hole=0.4,
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True,
    )


# ============================================================
# ANOMALIES TAB
# ============================================================

with tab_anomalies:

    st.markdown(
        "## 🚨 Anomaly Detection"
    )

    st.caption(
        "Isolation Forest analyzes quantity, unit price and revenue "
        "to identify unusual transactions."
    )

    anomalies_df = processed_df[
        processed_df["anomaly"] == True
    ].copy()

    if len(anomalies_df) == 0:

        st.success(
            "✅ No unusual transactions were detected."
        )

    else:

        st.warning(
            f"⚠️ {len(anomalies_df)} unusual "
            "transaction(s) detected."
        )

        anomaly_display = (
            anomalies_df.copy()
        )

        anomaly_display["order_date"] = (
            anomaly_display[
                "order_date"
            ].dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            anomaly_display,
            use_container_width=True,
            hide_index=True,
        )


    # -----------------------------------------
    # ANOMALY DISTRIBUTION
    # -----------------------------------------

    score_chart = px.histogram(
        processed_df,
        x="anomaly_score",
        color="anomaly",
        title="Anomaly Score Distribution",
        nbins=20,
    )

    st.plotly_chart(
        score_chart,
        use_container_width=True,
    )


# ============================================================
# PIPELINE TAB
# ============================================================

with tab_pipeline:

    st.markdown(
        "## ⚙️ Pipeline Execution"
    )

    st.caption(
        "Execution history is stored locally using SQLite."
    )

    history_df = (
        get_pipeline_history()
    )

    if history_df.empty:

        st.info(
            "No pipeline executions recorded yet."
        )

    else:

        history_display = (
            history_df.copy()
        )

        history_display[
            "revenue"
        ] = (
            history_display[
                "revenue"
            ].round(2)
        )

        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True,
        )


    st.markdown(
        "### 🔄 Current Pipeline Status"
    )

    pipeline_status = [
        (
            "1",
            "📥 Data Ingestion",
            "Completed"
        ),
        (
            "2",
            "✅ Data Validation",
            "Completed"
        ),
        (
            "3",
            "🔄 ETL Transformation",
            "Completed"
        ),
        (
            "4",
            "💰 Revenue Calculation",
            "Completed"
        ),
        (
            "5",
            "🤖 Anomaly Detection",
            "Completed"
        ),
        (
            "6",
            "📈 Analytics",
            "Completed"
        ),
        (
            "7",
            "🗄️ SQLite Storage",
            "Completed"
        ),
    ]

    for number, stage, status in pipeline_status:

        col1, col2, col3 = st.columns(
            [1, 5, 2]
        )

        with col1:

            st.write(
                f"**{number}**"
            )

        with col2:

            st.write(stage)

        with col3:

            st.success(
                status
            )


# ============================================================
# EXPORT TAB
# ============================================================

with tab_export:

    st.markdown(
        "## 📥 Export Results"
    )

    st.caption(
        "Download processed datasets and pipeline reports."
    )

    # -----------------------------------------
    # PROCESSED DATASET
    # -----------------------------------------

    export_df = (
        processed_df.copy()
    )

    export_df["order_date"] = (
        export_df[
            "order_date"
        ].dt.strftime("%Y-%m-%d")
    )

    processed_csv = (
        export_df.to_csv(
            index=False
        )
    )

    st.download_button(
        label="⬇️ Download Processed Dataset",
        data=processed_csv,
        file_name="processed_sales_data.csv",
        mime="text/csv",
        key="download_processed_dataset",
    )


    # -----------------------------------------
    # ANOMALY REPORT
    # -----------------------------------------

    anomaly_export = (
        anomalies_df.copy()
    )

    if len(anomaly_export) > 0:

        anomaly_export["order_date"] = (
            anomaly_export[
                "order_date"
            ].dt.strftime("%Y-%m-%d")
        )

        anomaly_csv = (
            anomaly_export.to_csv(
                index=False
            )
        )

    else:

        anomaly_csv = (
            "No anomalies detected."
        )


    st.download_button(
        label="⬇️ Download Anomaly Report",
        data=anomaly_csv,
        file_name="sales_anomaly_report.csv",
        mime="text/csv",
        key="download_anomalies",
    )


    # -----------------------------------------
    # PIPELINE REPORT
    # -----------------------------------------

    report_text = f"""
REAL-TIME SALES DATA PIPELINE
==============================

Pipeline Run:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

INPUT
-----
Rows received: {rows_in}

OUTPUT
------
Rows after processing: {rows_after}

SALES METRICS
-------------
Total orders: {total_orders}
Unique customers: {total_customers}
Total revenue: ₹{total_revenue:,.2f}

ANOMALY DETECTION
-----------------
Anomalies detected: {anomaly_count}

TECHNOLOGIES
------------
Python
Pandas
Streamlit
Plotly
Scikit-learn
SQLite

PIPELINE
--------
Data Ingestion
Data Validation
ETL Transformation
Revenue Calculation
Anomaly Detection
Analytics
SQLite Storage
"""

    st.download_button(
        label="⬇️ Download Pipeline Report",
        data=report_text,
        file_name="sales_pipeline_report.txt",
        mime="text/plain",
        key="download_pipeline_report",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Real-Time Sales Data Pipeline
        <br>
        Python • Pandas • Streamlit • Plotly •
        Scikit-learn • SQLite
    </div>
    """,
    unsafe_allow_html=True,
)