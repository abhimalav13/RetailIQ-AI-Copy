# ============================================================
# RETAILIQ AI — ADAPTIVE DATA INTELLIGENCE PLATFORM
# General-purpose, schema-aware analytics for CSV / Excel / JSON
# ============================================================
# Design principles:
# 1) Never assume every dataset is retail or has revenue/country.
# 2) Profile columns using names + observed values + data types.
# 3) Show only visuals supported by the uploaded data.
# 4) Keep multiple tables separate unless a safe relationship is known
#    (Olist) or the user explicitly configures a join.
# 5) Label inferred metrics and disclose assumptions.
#
# Run: streamlit run app.py
# Core dependencies: streamlit, pandas, numpy, plotly, openpyxl
# ============================================================

import io
import json
import re
import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="RetailIQ AI | Adaptive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- THEME ------------------------------
st.markdown("""
<style>
.stApp { background: #F5F7FA; color: #14213D; }
.block-container { padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: #14213D; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown { color: #FFFFFF; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #14213D !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #14213D !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  background: #FFFFFF; border-radius: 12px;
}
h1, h2, h3 { color: #14213D !important; font-weight: 750 !important; }
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E5EAF1; padding: 15px;
  border-radius: 13px; box-shadow: 0 3px 12px rgba(20,33,61,.045);
}
[data-testid="stMetricValue"] { color: #00A6A6 !important; font-weight: 750; }
.stButton > button {
  background: #00A6A6; color: white; border: 0; border-radius: 9px;
  font-weight: 650;
}
.stButton > button:hover { background: #008B8B; color: white; }
div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- CONSTANTS -----------------------------
MAX_UPLOAD_MB = 200
MAX_CHART_CATEGORIES = 25
MAX_CORRELATION_COLUMNS = 12
DATE_PARSE_THRESHOLD = 0.80

DATE_NAME_HINTS = ("date", "time", "timestamp", "created", "updated", "month", "year")
ID_NAME_HINTS = ("_id", " id", "id_", "code", "sku", "zip", "postal", "phone")
MONEY_NAME_HINTS = ("revenue", "sales", "amount", "price", "cost", "spend",
                    "payment", "value", "fare", "salary", "profit", "income",
                    "budget", "fee", "discount", "freight", "tax")
QUANTITY_NAME_HINTS = ("quantity", "qty", "units", "count", "volume", "stock")
CATEGORY_NAME_HINTS = ("category", "segment", "region", "country", "city",
                       "state", "status", "type", "department", "brand", "channel")
TEXT_NAME_HINTS = ("description", "comment", "review", "message", "notes", "text")

# ----------------------- UTILITIES ----------------------------
def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def nice_name(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").strip().title()


def safe_read_csv(uploaded):
    raw = uploaded.getvalue()
    # Try common encodings and separators; preserve identifiers as text initially.
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False,
                                sep=None, engine="python")
            if frame.shape[1] == 1 and b";" in raw[:5000]:
                frame = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=";", low_memory=False)
            return frame
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV: {last_error}")


def read_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = safe_read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded, engine="openpyxl" if name.endswith("xlsx") else None)
    elif name.endswith(".json"):
        raw = uploaded.getvalue()
        try:
            obj = json.loads(raw.decode("utf-8"))
            df = pd.json_normalize(obj)
        except Exception:
            df = pd.read_json(io.BytesIO(raw), lines=True)
    else:
        raise ValueError("Supported formats: CSV, XLSX, XLS, JSON.")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The file has no tabular rows.")
    df.columns = make_unique_columns([str(c).strip() for c in df.columns])
    return df


def make_unique_columns(columns):
    seen, result = {}, []
    for col in columns:
        base = col if col else "unnamed_column"
        key = base.lower()
        seen[key] = seen.get(key, 0) + 1
        result.append(base if seen[key] == 1 else f"{base}_{seen[key]}")
    return result


def clean_frame(source: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = source.copy()
    notes = []
    # Remove fully empty rows/columns only. Do not silently drop legitimate records.
    before = df.shape
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.shape != before:
        notes.append(f"Removed fully empty rows/columns: {before} → {df.shape}.")
    # Trim whitespace in text columns, preserving missing values.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        s = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA,
                       "NULL": pd.NA, "None": pd.NA})
        df[col] = s
    # Convert numeric-looking text only when conversion is highly successful
    # and the field does not look like an identifier.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        n = norm_name(col)
        if any(h in n for h in ID_NAME_HINTS):
            continue
        raw = df[col].dropna()
        if len(raw) < 5:
            continue
        cleaned = raw.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= 0.95:
            converted = pd.to_numeric(
                df[col].astype("string").str.replace(",", "", regex=False).str.replace("$", "", regex=False),
                errors="coerce"
            )
            df[col] = converted
            notes.append(f"Detected numeric values in '{col}'.")
    return df, notes


def infer_semantic_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer a conservative semantic type for every column."""
    result = {}
    for col in df.columns:
        s = df[col]
        name = norm_name(col)
        nonnull = s.dropna()
        if len(nonnull) == 0:
            result[col] = "Empty"
            continue
        if pd.api.types.is_bool_dtype(s):
            result[col] = "Boolean"
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            result[col] = "Date / Time"
            continue
        # Parse dates from text only when the column name or values support it.
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            date_hint = any(h in name for h in DATE_NAME_HINTS)
            if date_hint or len(nonnull) >= 10:
                sample = nonnull.astype(str).head(500)
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
                ratio = parsed.notna().mean() if len(parsed) else 0
                if ratio >= DATE_PARSE_THRESHOLD:
                    result[col] = "Date / Time"
                    continue
        if pd.api.types.is_numeric_dtype(s):
            unique = int(nonnull.nunique())
            if unique <= 2 and set(pd.unique(nonnull)).issubset({0, 1}):
                result[col] = "Boolean-like"
            elif any(h in name for h in ID_NAME_HINTS):
                result[col] = "Identifier"
            elif unique <= 12 and not any(h in name for h in MONEY_NAME_HINTS + QUANTITY_NAME_HINTS):
                result[col] = "Numeric Category"
            else:
                result[col] = "Numeric Measure"
            continue
        unique = int(nonnull.nunique())
        ratio = unique / max(len(nonnull), 1)
        if any(h in name for h in ID_NAME_HINTS) or (ratio > 0.85 and unique > 30):
            result[col] = "Identifier"
        elif any(h in name for h in TEXT_NAME_HINTS) or (nonnull.astype(str).str.len().mean() > 70):
            result[col] = "Text"
        elif unique <= min(50, max(10, int(len(nonnull) * 0.20))):
            result[col] = "Category"
        else:
            result[col] = "Text"
    return result


def profile_table(df: pd.DataFrame, types: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()
        rows.append({
            "Column": col,
            "Detected type": types.get(col, "Unknown"),
            "Data type": str(s.dtype),
            "Non-null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(float(s.isna().mean() * 100), 2),
            "Unique": int(s.nunique(dropna=True)),
            "Example values": ", ".join(map(str, nonnull.head(3).tolist()))[:120],
        })
    return pd.DataFrame(rows)


def find_column_by_alias(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    normalized = {norm_name(c): c for c in df.columns}
    for alias in aliases:
        if norm_name(alias) in normalized:
            return normalized[norm_name(alias)]
    return None


def prepare_olist(tables: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Safely create Olist order-item grain; joins are validated many-to-one."""
    notes = []
    orders = tables.get("olist_orders_dataset")
    items = tables.get("olist_order_items_dataset")
    products = tables.get("olist_products_dataset")
    customers = tables.get("olist_customers_dataset")
    if orders is None or items is None:
        return None, notes
    required = {"order_id"}
    if not required.issubset(orders.columns) or not required.issubset(items.columns):
        return None, ["Olist files detected, but order_id is missing; no automatic join was performed."]
    # Item table is the transaction grain: one row per order item.
    merged = items.copy()
    merged = merged.merge(orders, on="order_id", how="left", validate="many_to_one",
                          suffixes=("_item", "_order"))
    notes.append("Joined order items to orders on order_id (many-to-one validation passed).")
    if products is not None and "product_id" in merged and "product_id" in products:
        p = products.copy()
        merged = merged.merge(p, on="product_id", how="left", validate="many_to_one",
                              suffixes=("", "_product"))
        notes.append("Joined products on product_id (many-to-one validation passed).")
    if customers is not None and "customer_id" in merged and "customer_id" in customers:
        c = customers.copy()
        merged = merged.merge(c, on="customer_id", how="left", validate="many_to_one",
                              suffixes=("", "_customer"))
        notes.append("Joined customers on customer_id (many-to-one validation passed).")
    sellers = tables.get("olist_sellers_dataset")
    if sellers is not None and "seller_id" in merged and "seller_id" in sellers:
        merged = merged.merge(sellers, on="seller_id", how="left", validate="many_to_one",
                              suffixes=("", "_seller"))
        notes.append("Joined sellers on seller_id (many-to-one validation passed).")
    # Reviews are aggregated to one row per order before joining, preventing
    # multiple review rows from multiplying order-item records.
    reviews = tables.get("olist_order_reviews_dataset")
    if reviews is not None and {"order_id", "review_score"}.issubset(reviews.columns):
        review_agg = reviews.groupby("order_id", as_index=False).agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "nunique") if "review_id" in reviews.columns else ("review_score", "count")
        )
        merged = merged.merge(review_agg, on="order_id", how="left", validate="many_to_one")
        notes.append("Aggregated reviews to order level and joined on order_id; order-item rows were not multiplied.")
    translations = tables.get("product_category_name_translation")
    if translations is not None and {"product_category_name", "product_category_name_english"}.issubset(translations.columns) and "product_category_name" in merged:
        merged = merged.merge(translations, on="product_category_name", how="left",
                              validate="many_to_one")
        notes.append("Added English product category labels.")
    # Standardized semantic names are additive; original fields are retained.
    rename_map = {
        "order_purchase_timestamp": "Order Date",
        "order_status": "Order Status",
        "customer_unique_id": "Customer",
        "customer_city": "Customer City",
        "customer_state": "Customer State",
        "product_category_name_english": "Product Category (English)",
        "product_category_name": "Product Category",
        "price": "Item Price",
        "freight_value": "Freight Value",
        "review_score": "Review Score",
    }
    for old, new in rename_map.items():
        if old in merged.columns and new not in merged.columns:
            merged[new] = merged[old]
    # Each order_items row is one order line. It is not necessarily a physical
    # unit count in all datasets, so label it explicitly as item lines.
    merged["Order Item Lines"] = 1
    notes.append("Order Item Lines counts rows in the Olist order-items table; it is not a supplied quantity field.")
    return merged, notes


def format_number(value):
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def choose_measure_columns(df, types):
    return [c for c, t in types.items() if t == "Numeric Measure" and df[c].nunique(dropna=True) > 1]


def choose_category_columns(df, types):
    return [c for c, t in types.items()
            if t in ("Category", "Numeric Category", "Boolean", "Boolean-like")
            and 1 < df[c].nunique(dropna=True) <= 200]


def choose_date_columns(types):
    return [c for c, t in types.items() if t == "Date / Time"]


def guess_role(df, types, role):
    aliases = {
        "date": ["date", "order_date", "invoice_date", "transaction_date", "timestamp",
                 "created_at", "order_purchase_timestamp"],
        "amount": ["revenue", "sales", "amount", "total", "price", "unit_price",
                   "payment_value", "item_price", "net_sales"],
        "category": ["category", "product_category", "product_category_name_english",
                     "segment", "department", "product_type"],
        "entity": ["customer_id", "customer_unique_id", "client_id", "user_id",
                   "account_id", "member_id"],
    }
    match = find_column_by_alias(df, aliases[role])
    if match and (role != "date" or types.get(match) == "Date / Time"):
        return match
    if role == "date":
        cols = choose_date_columns(types)
        return cols[0] if cols else None
    if role == "amount":
        candidates = [c for c in choose_measure_columns(df, types)
                      if any(h in norm_name(c) for h in MONEY_NAME_HINTS)]
        return candidates[0] if candidates else None
    if role == "category":
        cols = choose_category_columns(df, types)
        return cols[0] if cols else None
    if role == "entity":
        cols = [c for c, t in types.items() if t == "Identifier"]
        return cols[0] if cols else None
    return None


# --------------------- DATA UPLOAD -----------------------------
st.sidebar.title("RetailIQ AI")
st.sidebar.caption("Adaptive Analytics & Decision Support")
st.sidebar.markdown("---")
uploads = st.sidebar.file_uploader(
    "Upload one or more datasets",
    type=["csv", "xlsx", "xls", "json"],
    accept_multiple_files=True,
    help="Upload one table or multiple related tables. Unrelated files remain separate until you configure a join.",
)
st.sidebar.caption("No universal schema is assumed. Retail, finance, HR, marketing, operations, survey and other tabular datasets are profiled before analysis.")

if not uploads:
    st.title("RetailIQ AI")
    st.subheader("Smart Analytics That Adapts to Your Data")
    st.write("Upload CSV, Excel or JSON files to begin. RetailIQ will inspect each table, infer data types, flag data-quality issues and recommend only the analyses supported by the data.")
    a, b, c = st.columns(3)
    a.metric("Schema-aware", "Yes", "Names + observed values")
    b.metric("Visual selection", "Dynamic", "Based on data evidence")
    c.metric("Unsupported metrics", "Hidden", "No invented KPIs")
    st.info("For multiple files, upload related tables together. The app automatically recognizes the supported Olist relationship pattern. For other datasets, use the Join Tables page and select matching keys yourself.")
    st.stop()

if sum(f.size for f in uploads) > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"Combined upload exceeds {MAX_UPLOAD_MB} MB. Upload fewer or smaller files.")
    st.stop()

tables: Dict[str, pd.DataFrame] = {}
load_notes = []
for file in uploads:
    table_name = norm_name(file.name.rsplit(".", 1)[0])
    try:
        raw_df = read_upload(file)
        df, notes = clean_frame(raw_df)
        tables[table_name] = df
        load_notes.extend([f"{file.name}: {n}" for n in notes])
    except Exception as exc:
        st.sidebar.error(f"{file.name}: {exc}")

if not tables:
    st.error("No file could be loaded. Check the file format and try again.")
    st.stop()

# Attempt known Olist model only when its core tables exist.
olist_df, olist_notes = prepare_olist(tables)
if olist_df is not None:
    tables["olist_combined_order_items"] = olist_df
    load_notes.extend(olist_notes)
    st.sidebar.success("Recognized Olist relational dataset and built an order-item-level table.")

# ---------------------- DATASET SELECTOR ----------------------
st.title("RetailIQ AI")
st.caption("Adaptive Data Intelligence | Data-driven analysis without fixed dataset assumptions")
table_names = list(tables.keys())
default_table = "olist_combined_order_items" if "olist_combined_order_items" in tables else table_names[0]
selected_table = st.sidebar.selectbox("Select table to analyse", table_names,
                                      index=table_names.index(default_table),
                                      format_func=nice_name)
df = tables[selected_table].copy()
if df.empty:
    st.error("Selected table is empty.")
    st.stop()

# Parse inferred dates into real datetime columns (conservative).
types = infer_semantic_types(df)
for col, typ in types.items():
    if typ == "Date / Time" and not pd.api.types.is_datetime64_any_dtype(df[col]):
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().mean() >= DATE_PARSE_THRESHOLD:
            df[col] = parsed

types = infer_semantic_types(df)
profile = profile_table(df, types)
date_cols = choose_date_columns(types)
measure_cols = choose_measure_columns(df, types)
category_cols = choose_category_columns(df, types)
id_cols = [c for c, t in types.items() if t == "Identifier"]

# ------------------------- FILTERS ----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Explore filters")
filtered = df.copy()
filter_candidates = [c for c in category_cols if df[c].nunique(dropna=True) <= 60][:4]
for col in filter_candidates:
    values = sorted(df[col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect(nice_name(col), values, default=[])
    if selected:
        filtered = filtered[filtered[col].astype(str).isin(selected)]
for col in date_cols[:1]:
    valid_dates = pd.to_datetime(df[col], errors="coerce").dropna()
    if len(valid_dates) > 1:
        min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
        date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                           min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            filtered = filtered[(pd.to_datetime(filtered[col], errors="coerce") >= start) &
                                (pd.to_datetime(filtered[col], errors="coerce") < end)]

# ---------------------- OVERVIEW PAGE -------------------------
pages = ["Overview", "Explore & Visualize", "Data Quality", "Relationships", "Insights"]
page = st.sidebar.radio("Navigation", pages)

if page == "Overview":
    st.header("Dataset overview")
    row1 = st.columns(4)
    row1[0].metric("Rows", format_number(len(filtered)))
    row1[1].metric("Columns", format_number(filtered.shape[1]))
    row1[2].metric("Duplicate rows", format_number(int(filtered.duplicated().sum())))
    row1[3].metric("Missing cells", format_number(int(filtered.isna().sum().sum())))
    st.subheader("Dataset snapshot")
    st.dataframe(filtered.head(12), use_container_width=True)
    st.subheader("Detected column roles")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.subheader("Available analysis")
    capability_rows = [
        ("Time analysis", bool(date_cols), "A reliable date/time column was detected."),
        ("Numeric distributions", bool(measure_cols), "One or more numeric measures were detected."),
        ("Category comparisons", bool(category_cols), "Categorical fields with useful variation were detected."),
        ("Correlation analysis", len(measure_cols) >= 2, "At least two numeric measures are available."),
        ("Entity-level analysis", bool(id_cols), "Potential identifiers were detected; uniqueness is checked before use."),
    ]
    for label, available, reason in capability_rows:
        (st.success if available else st.info)(f"{'Available' if available else 'Not shown'} — **{label}:** {reason}")
    if load_notes:
        with st.expander("Import and integration notes"):
            for note in load_notes:
                st.write("• " + note)
    st.caption("Column roles are inferred heuristically. Review the detected roles before using them for high-impact decisions.")

elif page == "Explore & Visualize":
    st.header("Explore & visualize")
    st.caption("Charts are offered only when the selected columns contain enough valid observations and variation.")
    if not filtered.shape[0]:
        st.warning("No rows remain after filters.")
        st.stop()
    chart_options = []
    if measure_cols:
        chart_options.append("Numeric distribution")
    if category_cols and measure_cols:
        chart_options.append("Measure by category")
    if len(measure_cols) >= 2:
        chart_options.append("Relationship between numeric measures")
    if date_cols and measure_cols:
        chart_options.append("Measure over time")
    if category_cols:
        chart_options.append("Category frequency")
    if not chart_options:
        st.info("No suitable chart can be generated from the detected fields. Review the data types and upload a more structured table.")
    else:
        chart_type = st.selectbox("Choose a recommended visual", chart_options)
        if chart_type == "Numeric distribution":
            x = st.selectbox("Numeric field", measure_cols)
            plot_df = filtered[[x]].dropna()
            if len(plot_df):
                fig = px.histogram(plot_df, x=x, nbins=35, marginal="box",
                                   title=f"Distribution of {nice_name(x)}")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Histogram shows the observed distribution; the box plot helps reveal spread and potential outliers.")
        elif chart_type == "Measure by category":
            x = st.selectbox("Category field", category_cols)
            y = st.selectbox("Numeric measure", measure_cols)
            agg = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Count"])
            work = filtered[[x, y]].dropna()
            if work.empty:
                st.warning("No valid values for this combination.")
            else:
                func = {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count"}[agg]
                grouped = work.groupby(x, dropna=False)[y].agg(func).sort_values(ascending=False).head(MAX_CHART_CATEGORIES).reset_index()
                fig = px.bar(grouped, x=x, y=y, title=f"{agg} of {nice_name(y)} by {nice_name(x)}",
                             labels={x: nice_name(x), y: f"{agg} {nice_name(y)}"})
                st.plotly_chart(fig, use_container_width=True)
                if filtered[x].nunique(dropna=True) == 1:
                    st.info("Only one category exists in the filtered data. A comparison chart is not meaningful; this bar shows the single observed group.")
        elif chart_type == "Relationship between numeric measures":
            x = st.selectbox("X-axis measure", measure_cols, index=0)
            y_default = 1 if len(measure_cols) > 1 else 0
            y = st.selectbox("Y-axis measure", measure_cols, index=y_default)
            sample = filtered[[x, y]].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            if x == y:
                st.warning("Choose two different measures.")
            elif len(sample) > 1:
                fig = px.scatter(sample, x=x, y=y, trendline=None,
                                 title=f"{nice_name(x)} vs {nice_name(y)}",
                                 opacity=0.65)
                st.plotly_chart(fig, use_container_width=True)
                corr = sample[x].corr(sample[y])
                st.caption(f"Pearson correlation: {corr:.3f}. Correlation does not establish causation.")
        elif chart_type == "Measure over time":
            d = st.selectbox("Date field", date_cols)
            y = st.selectbox("Numeric measure", measure_cols)
            freq_label = st.selectbox("Time grouping", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
            freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}[freq_label]
            work = filtered[[d, y]].copy()
            work[d] = pd.to_datetime(work[d], errors="coerce")
            work = work.dropna()
            if not work.empty:
                series = work.set_index(d)[y].resample(freq).sum().reset_index()
                fig = px.line(series, x=d, y=y, markers=True,
                              title=f"{nice_name(y)} over time ({freq_label.lower()})")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Values are summed within each time period. Select a different aggregation in a future metric-definition layer if the measure is not additive.")
        elif chart_type == "Category frequency":
            x = st.selectbox("Category field", category_cols)
            counts = filtered[x].astype("string").fillna("(Missing)").value_counts().head(MAX_CHART_CATEGORIES).rename_axis(x).reset_index(name="Records")
            fig = px.bar(counts, x=x, y="Records", title=f"Record count by {nice_name(x)}")
            st.plotly_chart(fig, use_container_width=True)
            if filtered[x].nunique(dropna=True) == 1:
                st.info("Only one category is present. This is a frequency summary, not a comparison.")

    with st.expander("Create a custom chart"):
        all_cols = list(filtered.columns)
        chart_kind = st.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram", "Box"])
        custom_x = st.selectbox("X field", all_cols, key="custom_x")
        numeric_for_y = [c for c in all_cols if pd.api.types.is_numeric_dtype(filtered[c])]
        custom_y = st.selectbox("Y field (if needed)", ["(none)"] + numeric_for_y, key="custom_y")
        if st.button("Build custom chart"):
            try:
                if chart_kind == "Histogram":
                    fig = px.histogram(filtered, x=custom_x)
                elif chart_kind == "Box":
                    fig = px.box(filtered, x=custom_x, y=None if custom_y == "(none)" else custom_y)
                elif chart_kind == "Bar":
                    if custom_y == "(none)":
                        tmp = filtered[custom_x].astype("string").value_counts().head(50).reset_index()
                        tmp.columns = [custom_x, "Records"]
                        fig = px.bar(tmp, x=custom_x, y="Records")
                    else:
                        fig = px.bar(filtered, x=custom_x, y=custom_y)
                elif chart_kind == "Line":
                    if custom_y == "(none)":
                        st.error("Line chart needs a Y field.")
                        st.stop()
                    fig = px.line(filtered, x=custom_x, y=custom_y)
                else:
                    if custom_y == "(none)":
                        st.error("Scatter chart needs a Y field.")
                        st.stop()
                    fig = px.scatter(filtered, x=custom_x, y=custom_y)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not build this chart: {exc}")

elif page == "Data Quality":
    st.header("Data quality assessment")
    n_rows = len(df)
    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "Potential constant": [df[c].nunique(dropna=True) <= 1 for c in df.columns],
    })
    q1, q2, q3 = st.columns(3)
    q1.metric("Complete rows", f"{(df.notna().all(axis=1).mean()*100):.1f}%" if n_rows else "—")
    q2.metric("Duplicate rows", format_number(df.duplicated().sum()))
    q3.metric("Columns with missing data", format_number((df.isna().any()).sum()))
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.subheader("Potential data issues")
    issues = []
    dup = int(df.duplicated().sum())
    if dup:
        issues.append(f"{dup:,} fully duplicated rows. Check whether these are accidental duplicates or valid repeated observations.")
    for col in df.columns:
        missing_pct = float(df[col].isna().mean() * 100)
        if missing_pct >= 30:
            issues.append(f"'{col}' has {missing_pct:.1f}% missing values.")
        if df[col].nunique(dropna=True) <= 1:
            issues.append(f"'{col}' is constant (or empty); it usually adds little analytical value.")
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 10:
            s = df[col].dropna()
            q1v, q3v = s.quantile([0.25, 0.75])
            iqr = q3v - q1v
            if iqr > 0:
                outliers = int(((s < q1v - 1.5 * iqr) | (s > q3v + 1.5 * iqr)).sum())
                if outliers:
                    issues.append(f"'{col}' has {outliers:,} possible IQR outliers. Outliers are not automatically errors.")
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("No major structural quality warnings were detected by the basic checks.")
    st.subheader("Data cleaning policy")
    st.write("RetailIQ trims surrounding text whitespace and removes fully empty rows/columns. It does not automatically delete duplicates, impute missing values, remove outliers or change business values. Review and approve those transformations before modelling.")

elif page == "Relationships":
    st.header("Table relationships")
    if len(tables) < 2:
        st.info("Upload at least two tables to inspect or join relationships.")
    else:
        st.write("Only the known Olist relationships are joined automatically. For other data, configure a join explicitly; RetailIQ will not guess a relationship and risk multiplying rows.")
        st.subheader("Available tables")
        table_summary = pd.DataFrame([
            {"Table": name, "Rows": len(frame), "Columns": len(frame.columns),
             "Potential key columns": ", ".join([c for c in frame.columns if "id" in norm_name(c) or "key" in norm_name(c)][:8])}
            for name, frame in tables.items()
        ])
        st.dataframe(table_summary, use_container_width=True, hide_index=True)
        if "olist_combined_order_items" in tables:
            st.success("Olist combined table is available in the table selector. Joins were validated as many-to-one.")
        st.subheader("Build a manual join")
        left_name = st.selectbox("Left table", list(tables.keys()), key="left_table")
        right_choices = [x for x in tables if x != left_name]
        right_name = st.selectbox("Right table", right_choices, key="right_table")
        left_df, right_df = tables[left_name], tables[right_name]
        left_key = st.selectbox("Left join key", list(left_df.columns), key="left_key")
        right_key = st.selectbox("Right join key", list(right_df.columns), key="right_key")
        how = st.selectbox("Join type", ["left", "inner", "outer"])
        validate = st.selectbox("Expected relationship", ["many_to_one", "one_to_one", "one_to_many", "many_to_many"])
        if st.button("Preview join", type="primary"):
            try:
                joined = left_df.merge(right_df, left_on=left_key, right_on=right_key,
                                       how=how, validate=validate, suffixes=("_left", "_right"))
                st.success(f"Join succeeded: {len(joined):,} rows × {joined.shape[1]} columns.")
                st.dataframe(joined.head(20), use_container_width=True)
                st.download_button("Download joined CSV", joined.to_csv(index=False).encode("utf-8"),
                                   file_name="retailiq_joined_data.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Join was not accepted: {exc}. Check key values and expected relationship. Many-to-many joins can multiply rows.")

elif page == "Insights":
    st.header("Evidence-based insights")
    st.caption("These are descriptive signals, not causal conclusions or a substitute for domain-specific metric definitions.")
    if len(filtered) == 0:
        st.info("No records remain after filters.")
    else:
        insight_count = 0
        for col in measure_cols[:8]:
            s = filtered[col].dropna()
            if len(s) >= 2:
                st.write(f"**{nice_name(col)}:** median {format_number(s.median())}, mean {format_number(s.mean())}, range {format_number(s.min())} to {format_number(s.max())}.")
                insight_count += 1
        for col in category_cols[:5]:
            counts = filtered[col].astype("string").fillna("(Missing)").value_counts()
            if len(counts):
                share = counts.iloc[0] / max(counts.sum(), 1) * 100
                st.write(f"**{nice_name(col)}:** most common value is **{counts.index[0]}** ({share:.1f}% of non-missing records).")
                insight_count += 1
        for col in date_cols[:3]:
            dates = pd.to_datetime(filtered[col], errors="coerce").dropna()
            if len(dates):
                st.write(f"**{nice_name(col)}:** observed date range {dates.min().date()} to {dates.max().date()}.")
                insight_count += 1
        if insight_count == 0:
            st.info("There are not enough valid values to generate descriptive insights.")

# ---------------------- EXPORT / FOOTER -----------------------
st.sidebar.markdown("---")
st.sidebar.caption("RetailIQ AI • Adaptive analytics")
with st.expander("Export current analysed table"):
    st.download_button(
        "Download filtered data as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_table}_filtered.csv",
        mime="text/csv",
    )
st.caption("Important: automatic semantic detection is heuristic. Validate column meaning, units, currency, row grain and join keys before making business or financial decisions.")
