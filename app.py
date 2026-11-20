
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# RETAILIQ AI - BUSINESS INTELLIGENCE APP
# ==========================================

st.set_page_config(
    page_title="RetailIQ AI",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# RETAILIQ AI - CUSTOM UI THEME
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F5F7FA;
    color: #14213D;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14213D;
}


/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #14213D;
}

/* Keep sidebar headings and labels white */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {
    color: #FFFFFF;
}

/* Currency selector - dark readable text */
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #14213D !important;
}

/* File uploader - readable text */
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #14213D !important;
}

/* File uploader label */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #FFFFFF;
    border-radius: 12px;
}

/* Sidebar information box */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #E8F1FF;
    border-radius: 12px;
}

[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #14213D !important;
}

/* Navigation labels */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #FFFFFF !important;
}


/* Main headings */
h1, h2, h3 {
    color: #14213D !important;
    font-weight: 700 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5EAF1;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(20, 33, 61, 0.06);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #00A6A6 !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #00A6A6;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #008B8B;
    color: white;
    border: none;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #E5EAF1;
    border-radius: 10px;
}

/* Input fields */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Info and success messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}
.block-container {
    padding-top: 1.5rem;
}
h1, h2, h3 {
    color: #172554;
}
</style>
""", unsafe_allow_html=True)


st.image(
    "ff744770-ea67-497d-916e-55c45b633585.png",
    width=520
)

st.caption("Smart Retail Analytics & Business Decision Support")

st.sidebar.title("RetailIQ AI")
st.sidebar.caption("Business Intelligence Platform")

# CURRENCY CONFIGURATION

currency_options = {
    "INR": {"name": "Indian Rupee (₹)", "symbol": "₹"},
    "USD": {"name": "US Dollar ($)", "symbol": "$"},
    "GBP": {"name": "British Pound (£)", "symbol": "£"},
    "EUR": {"name": "Euro (€)", "symbol": "€"},
    "AED": {"name": "UAE Dirham (د.إ)", "symbol": "د.إ"},
    "CAD": {"name": "Canadian Dollar (C$)", "symbol": "C$"},
    "AUD": {"name": "Australian Dollar (A$)", "symbol": "A$"}
}

selected_currency = st.sidebar.selectbox(
    "Display Currency (symbol only)",
    options=list(currency_options.keys()),
    format_func=lambda x: currency_options[x]["name"]
)

currency_symbol = currency_options[selected_currency]["symbol"]

uploaded_files = st.sidebar.file_uploader(
    "Upload retail dataset(s)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    help="Upload one transaction file, or multiple related Olist CSV files."
)

st.sidebar.info(
    "Upload one retail transaction file, or upload related Olist CSV files "
    "together. RetailIQ AI will detect and combine supported Olist tables."
)


# ==========================================
# DATA LOADING & OLIST MULTI-FILE INTEGRATION
# ==========================================

import io
import os

@st.cache_data
def load_data(file_bytes, file_name):
    """Read CSVs with common encodings; leave Excel handling unchanged."""
    if file_name.lower().endswith(".csv"):
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, low_memory=False)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(
            "Could not decode this CSV using UTF-8, UTF-8-BOM, Windows-1252 or Latin-1. "
            f"Please check that the file is a text CSV. Details: {last_error}"
        )
    return pd.read_excel(io.BytesIO(file_bytes))


def normalize_column(col):
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())


def find_uploaded_file(file_map, required_name):
    """Find an uploaded CSV by its base filename, ignoring extension/case."""
    target = normalize_column(required_name)
    for normalized_name, dataframe in file_map.items():
        if normalized_name == target:
            return dataframe
    return None


if not uploaded_files:
    st.warning(
        "Please upload a CSV/Excel transaction file or multiple related Olist CSV files "
        "using the sidebar to begin."
    )
    st.markdown("""
    ### Welcome to RetailIQ AI

    Analyse your retail business with:
    - Sales and revenue intelligence
    - Product performance analytics
    - Market analysis
    - Sales forecasting
    - Business recommendations

    **Getting started:** Upload your retail transaction data. For Olist, select the
    orders, order-items, products and customers CSV files together.
    """)
    st.stop()


try:
    uploaded_data = {}
    for file in uploaded_files:
        base_name = os.path.splitext(file.name)[0]
        uploaded_data[normalize_column(base_name)] = load_data(
            file.getvalue(), file.name
        )
except Exception as e:
    st.error(f"Unable to read one of the uploaded files: {e}")
    st.stop()



# ==========================================
# ADAPTIVE DATA PROFILING & GENERIC ANALYTICS
# ==========================================

import re

def _clean_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_identifier_name(column):
    """Identify record keys and code-like columns without treating all numbers as measures."""
    name = _clean_name(column)
    exact_ids = {
        "id", "rownumber", "rownumberid", "index", "recordnumber", "serialnumber",
        "customerid", "customeruniqueid", "employeeid", "employeenumber", "productid", "orderid",
        "orderitemid", "invoiceid", "invoiceno", "transactionid", "sellerid",
        "departmentid", "categoryid", "stockcode", "productcode", "sku", "uuid",
        "zipcode", "postalcode", "phonenumber", "accesskey"
    }
    return (
        name in exact_ids
        or name.endswith("id")
        or name.endswith("uuid")
        or name.endswith("code")
        or name.endswith("key")
        or name in {"latitude", "longitude", "lat", "lon", "lng"}
        or name in {"customeremail", "customerpassword", "customerstreet"}
    )


def _is_flag_name(column):
    name = _clean_name(column)
    flag_tokens = (
        "attrition", "exited", "churn", "isactive", "hascrcard", "late_delivery_risk",
        "late delivery risk", "flag", "risk", "isdelivered", "isreturned", "cancelled"
    )
    return any(_clean_name(token) in name for token in flag_tokens)


def _profile_table(frame):
    profile = {
        "rows": len(frame), "columns": len(frame.columns), "numeric": [],
        "dates": [], "categorical": [], "identifiers": [], "geo": [], "metadata": [],
        "missing": {}, "unique": {}, "duplicate_rows": int(frame.duplicated().sum())
    }
    for col in frame.columns:
        series = frame[col]
        profile["missing"][col] = int(series.isna().sum())
        profile["unique"][col] = int(series.nunique(dropna=True))
        name = _clean_name(col)

        if name in {"latitude", "longitude", "lat", "lon", "lng"}:
            profile["geo"].append(col)
            continue
        if _is_identifier_name(col):
            profile["identifiers"].append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            profile["dates"].append(col)
            continue

        date_name = any(token in name for token in (
            "date", "time", "timestamp", "created", "updated", "month", "year"
        ))
        if date_name and not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            nonmissing = int(series.notna().sum())
            if nonmissing and parsed.notna().sum() / nonmissing >= 0.65:
                profile["dates"].append(col)
                continue

        converted = pd.to_numeric(series, errors="coerce")
        nonmissing = int(series.notna().sum())
        numeric_ratio = converted.notna().sum() / nonmissing if nonmissing else 0
        unique_count = profile["unique"][col]
        if unique_count <= 1 and pd.api.types.is_numeric_dtype(series):
            profile["metadata"].append(col)
            continue
        if _is_flag_name(col) or (unique_count == 2 and numeric_ratio >= 0.85):
            profile["categorical"].append(col)
        elif pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.85:
            profile["numeric"].append(col)
        elif unique_count <= max(50, int(len(series) * 0.20)):
            profile["categorical"].append(col)

    return profile


def _retail_schema_is_compatible(frame):
    aliases = {
        "order": {"invoiceno", "invoice", "invoiceid", "orderid", "ordernumber", "transactionid"},
        "product": {"stockcode", "productid", "productcode", "sku", "itemcode"},
        "date": {"invoicedate", "orderdate", "transactiondate", "date", "orderpurchasetimestamp"},
        "quantity": {"quantity", "qty", "unitssold", "numberofitems"},
        "price": {"unitprice", "sellingprice", "priceperunit", "price"},
    }
    normalized = {_clean_name(c) for c in frame.columns}
    return all(bool(normalized & values) for values in aliases.values())


def _find_column(frame, aliases):
    lookup = {_clean_name(c): c for c in frame.columns}
    for alias in aliases:
        if _clean_name(alias) in lookup:
            return lookup[_clean_name(alias)]
    return None


def _truthy_mask(series):
    """Interpret common binary/text outcomes without assuming a particular encoding."""
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "yes", "y", "true", "left", "exited", "churned", "late", "delayed"})


def _domain_summary(frame):
    cols = {_clean_name(c): c for c in frame.columns}
    has = lambda *names: any(_clean_name(n) in cols for n in names)
    if has("Attrition") and (has("Department") or has("JobRole")):
        return "HR / Employee Analytics"
    if has("Exited", "Churn") and (has("CreditScore") or has("Balance") or has("Geography")):
        return "Banking / Customer Churn"
    if has("Late_delivery_risk", "Delivery Status", "Days for shipping (real)"):
        return "Supply Chain / Logistics"
    if has("InvoiceNo", "StockCode", "Quantity", "UnitPrice"):
        return "Retail Transactions"
    return "General Dataset Analytics"


def _render_domain_kpis(frame, domain):
    """Render only KPIs supported by recognizable columns in the uploaded data."""
    cards = []
    if domain == "HR / Employee Analytics":
        target = _find_column(frame, ["Attrition"])
        if target:
            left = _truthy_mask(frame[target])
            cards.extend([
                ("Employees", f"{len(frame):,}"),
                ("Employees left", f"{int(left.sum()):,}"),
                ("Attrition rate", f"{left.mean() * 100:.1f}%" if len(left) else "—")
            ])
        age = _find_column(frame, ["Age"])
        if age:
            vals = pd.to_numeric(frame[age], errors="coerce").dropna()
            if len(vals): cards.append(("Average age", f"{vals.mean():.1f}"))
    elif domain == "Banking / Customer Churn":
        target = _find_column(frame, ["Exited", "Churn", "Churned"])
        active = _find_column(frame, ["IsActiveMember", "ActiveMember", "IsActive"])
        balance = _find_column(frame, ["Balance", "AccountBalance"])
        cards.append(("Customers", f"{len(frame):,}"))
        if target:
            churn = _truthy_mask(frame[target])
            cards.extend([("Customers churned", f"{int(churn.sum()):,}"),
                          ("Churn rate", f"{churn.mean() * 100:.1f}%" if len(churn) else "—")])
        if active:
            active_mask = _truthy_mask(frame[active])
            cards.append(("Active customers", f"{int(active_mask.sum()):,}"))
        if balance:
            vals = pd.to_numeric(frame[balance], errors="coerce").dropna()
            if len(vals): cards.append(("Average balance", f"{vals.mean():,.2f}"))
    elif domain == "Supply Chain / Logistics":
        shipping = _find_column(frame, ["Days for shipping (real)", "ActualShippingDays", "ShippingDays"])
        late = _find_column(frame, ["Late_delivery_risk", "LateDeliveryRisk"])
        sales = _find_column(frame, ["Sales", "Revenue", "OrderSales"])
        cards.append(("Order / line records", f"{len(frame):,}"))
        if shipping:
            vals = pd.to_numeric(frame[shipping], errors="coerce").dropna()
            if len(vals): cards.append(("Average shipping days", f"{vals.mean():.2f}"))
        if late:
            risk = _truthy_mask(frame[late])
            cards.append(("Late-delivery risk", f"{risk.mean() * 100:.1f}%" if len(risk) else "—"))
        if sales:
            vals = pd.to_numeric(frame[sales], errors="coerce").dropna()
            if len(vals): cards.append(("Sales total", f"{vals.sum():,.2f}"))
    else:
        cards.append(("Records", f"{len(frame):,}"))
        for label, aliases, agg in [
            ("Total sales", ["Sales", "Revenue", "Amount"], "sum"),
            ("Average value", ["Amount", "Price", "Balance"], "mean")
        ]:
            col = _find_column(frame, aliases)
            if col:
                vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                if len(vals): cards.append((label, f"{getattr(vals, agg)():,.2f}"))
    if cards:
        cols = st.columns(min(4, len(cards)))
        for i, (label, value) in enumerate(cards):
            cols[i % len(cols)].metric(label, value)


def _render_adaptive_workspace(tables):
    """Dataset-aware analytics for non-legacy and multi-table uploads."""
    st.title("RetailIQ AI")
    st.subheader("Adaptive Analytics & Data Explorer")
    st.caption(
        "RetailIQ profiles the uploaded data, detects likely business context and only displays analyses supported by available fields."
    )
    st.success("Adaptive mode enabled — the original Online Retail dashboard logic has not been applied.")

    table_names = list(tables.keys())
    if not table_names:
        st.error("No readable tables were found.")
        return

    inventory = pd.DataFrame([
        {"Table": name, "Rows": len(tables[name]), "Columns": len(tables[name].columns)}
        for name in table_names
    ])
    with st.expander("Uploaded data inventory", expanded=len(table_names) > 1):
        st.dataframe(inventory, use_container_width=True, hide_index=True)

    working, base_name = None, None
    if len(table_names) == 1:
        base_name = table_names[0]
        working = tables[base_name].copy()
    else:
        st.subheader("Choose a primary table")
        default_base = max(table_names, key=lambda n: len(tables[n]))
        base_name = st.selectbox("Primary table for analysis", table_names,
                                 index=table_names.index(default_base), key="adaptive_primary_table")
        working = tables[base_name].copy()
        st.subheader("Optional table relationships")
        st.caption("Join matching keys only when they represent the same entity. Many-to-many joins can multiply records.")
        for idx, other_name in enumerate([n for n in table_names if n != base_name]):
            other = tables[other_name]
            with st.expander(f"Join {other_name}", expanded=False):
                common = [c for c in working.columns if c in other.columns]
                include = st.checkbox(f"Include {other_name}", value=False, key=f"adaptive_join_include_{idx}")
                if not include:
                    st.caption(f"Potential common columns: {', '.join(common[:8]) if common else 'None detected'}")
                    continue
                left_options, right_options = list(working.columns), list(other.columns)
                suggested_left = common[0] if common else left_options[0]
                suggested_right = suggested_left if suggested_left in right_options else right_options[0]
                c1, c2 = st.columns(2)
                left_key = c1.selectbox("Primary-table key", left_options, index=left_options.index(suggested_left), key=f"adaptive_left_key_{idx}")
                right_key = c2.selectbox("Related-table key", right_options, index=right_options.index(suggested_right), key=f"adaptive_right_key_{idx}")
                join_how = st.selectbox("Join type", ["left", "inner", "outer"], key=f"adaptive_join_how_{idx}")
                if working[left_key].duplicated().any() and other[right_key].duplicated().any():
                    st.warning("Both selected keys contain duplicates. This many-to-many join may multiply rows.")
                if st.button(f"Apply join: {other_name}", key=f"adaptive_apply_join_{idx}"):
                    try:
                        joined = working.merge(other, left_on=left_key, right_on=right_key, how=join_how,
                                               suffixes=("", f"_{_clean_name(other_name)[:12]}"))
                        st.session_state["adaptive_joined_frame"] = joined
                        st.session_state["adaptive_joined_base"] = base_name
                        st.success(f"Join applied: {len(joined):,} rows and {len(joined.columns):,} columns.")
                    except Exception as exc:
                        st.error(f"Join could not be completed: {exc}")
                    st.rerun()
        if st.session_state.get("adaptive_joined_base") == base_name and "adaptive_joined_frame" in st.session_state:
            working = st.session_state["adaptive_joined_frame"].copy()

    if working is None or working.empty:
        st.warning("The selected table is empty.")
        return

    profile = _profile_table(working)
    date_options, numeric_options, category_options = profile["dates"], profile["numeric"], profile["categorical"]
    domain = _domain_summary(working)
    st.subheader("Data quality and structure")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Records", f"{len(working):,}")
    q2.metric("Columns", f"{len(working.columns):,}")
    q3.metric("Business measures", f"{len(numeric_options):,}")
    q4.metric("Date fields", f"{len(date_options):,}")

    profile_rows = []
    for col in working.columns:
        detected = ("Date" if col in profile["dates"] else
                    "Identifier / key" if col in profile["identifiers"] else
                    "Geographic coordinate" if col in profile["geo"] else
                    "Constant / metadata" if col in profile["metadata"] else
                    "Numeric measure" if col in profile["numeric"] else
                    "Category / flag" if col in profile["categorical"] else "Text / high-cardinality")
        profile_rows.append({"Column": str(col), "Detected type": detected,
                             "Missing": profile["missing"][col], "Unique values": profile["unique"][col]})
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    total_cells = max(1, len(working) * len(working.columns))
    missing_cells = int(working.isna().sum().sum())
    missing_pct = missing_cells / total_cells * 100
    duplicate_rows = profile["duplicate_rows"]
    with st.expander("Data quality checks", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Missing cells", f"{missing_cells:,}", f"{missing_pct:.2f}% of all cells")
        d2.metric("Duplicate rows", f"{duplicate_rows:,}")
        d3.metric("Identifier fields", f"{len(profile['identifiers']):,}")
        high_missing = [(c, n) for c, n in profile["missing"].items() if n / max(1, len(working)) >= 0.20]
        if high_missing:
            st.warning("Columns with at least 20% missing values: " + ", ".join(f"{c} ({n:,})" for c, n in high_missing[:12]))
        else:
            st.success("No columns have 20% or more missing values.")

    st.subheader(f"Adaptive overview — {domain}")
    _render_domain_kpis(working, domain)
    overview = st.columns(3)
    overview[0].metric("Distinct categories", f"{sum(working[c].nunique(dropna=True) for c in category_options):,}")
    overview[1].metric("Numeric fields", f"{len(numeric_options):,}")
    overview[2].metric("Missing cells", f"{missing_cells:,}")

    if not numeric_options and not category_options and not date_options:
        st.info("No suitable numeric, date or categorical fields were detected.")
        st.dataframe(working.head(100), use_container_width=True)
        return

    # Choose a useful measure by semantic priority, excluding identifiers, flags and coordinates.
    preferred_measure = None
    measure_aliases = {
        "HR / Employee Analytics": ["MonthlyIncome", "Age", "YearsAtCompany"],
        "Banking / Customer Churn": ["Balance", "EstimatedSalary", "CreditScore", "Age"],
        "Supply Chain / Logistics": ["Sales", "Order Item Total", "Order Profit Per Order", "Days for shipping (real)"],
        "General Dataset Analytics": ["Sales", "Revenue", "Amount", "Profit", "Price"]
    }
    for alias in measure_aliases.get(domain, []):
        found = _find_column(working, [alias])
        if found in numeric_options:
            preferred_measure = found
            break
    if preferred_measure is None and numeric_options:
        preferred_measure = numeric_options[0]

    st.subheader("Explore your data")
    measure = st.selectbox("Numeric measure", numeric_options,
                           index=numeric_options.index(preferred_measure) if preferred_measure in numeric_options else 0,
                           key=f"adaptive_measure_{_clean_name(base_name)}") if numeric_options else None
    if measure:
        values = pd.to_numeric(working[measure], errors="coerce").dropna()
        a1, a2, a3 = st.columns(3)
        a1.metric(f"Total {measure}", f"{values.sum():,.2f}" if len(values) else "—")
        a2.metric(f"Average {measure}", f"{values.mean():,.2f}" if len(values) else "—")
        a3.metric(f"Median {measure}", f"{values.median():,.2f}" if len(values) else "—")

    # Prefer meaningful domain category fields rather than stale widget selections.
    preferred_category = None
    category_aliases = {
        "HR / Employee Analytics": ["Attrition", "Department", "JobRole", "Gender"],
        "Banking / Customer Churn": ["Exited", "Geography", "Gender", "IsActiveMember"],
        "Supply Chain / Logistics": ["Delivery Status", "Order Region", "Type", "Shipping Mode", "Market"],
        "General Dataset Analytics": []
    }
    for alias in category_aliases.get(domain, []):
        found = _find_column(working, [alias])
        if found in category_options:
            preferred_category = found
            break
    if preferred_category is None and category_options:
        preferred_category = category_options[0]

    tabs = st.tabs(["Overview & visuals", "Relationships", "Data preview"])
    with tabs[0]:
        if date_options:
            date_default = date_options[0]
            date_col = st.selectbox("Date field", date_options, index=0,
                                    key=f"adaptive_date_{_clean_name(base_name)}")
            if measure:
                trend_agg = st.selectbox("Time trend aggregation", ["Sum", "Average", "Record count"],
                                         index=1 if any(t in _clean_name(measure) for t in ["days", "duration", "age", "score", "rate"]) else 0,
                                         key=f"adaptive_time_agg_{_clean_name(base_name)}")
                temp = working[[date_col, measure]].copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
                temp = temp.dropna()
                if not temp.empty:
                    grouped_series = temp.set_index(date_col)[measure].resample("MS")
                    trend = (grouped_series.sum() if trend_agg == "Sum" else
                             grouped_series.mean() if trend_agg == "Average" else grouped_series.count()).reset_index(name=measure)
                    st.plotly_chart(px.line(trend, x=date_col, y=measure, markers=True,
                                            title=f"{trend_agg} {measure} by month", template="plotly_white"),
                                    use_container_width=True, key=f"adaptive_trend_{_clean_name(base_name)}")
                else:
                    st.caption("No valid date-and-measure pairs were found for this trend.")

        if category_options:
            category_col = st.selectbox("Category field", category_options,
                                        index=category_options.index(preferred_category) if preferred_category in category_options else 0,
                                        key=f"adaptive_category_{_clean_name(base_name)}")
            limit = st.slider("Number of categories to display", 5, 25, 10,
                              key=f"adaptive_category_limit_{_clean_name(base_name)}")
            chart_metric_options = ["Record count"]
            if measure:
                chart_metric_options += [f"Sum of {measure}", f"Average {measure}"]
            chart_metric = st.selectbox("Category chart value", chart_metric_options,
                                        key=f"adaptive_chart_metric_{_clean_name(base_name)}")
            if chart_metric == "Record count":
                grouped = working[category_col].fillna("Missing").value_counts().head(limit).rename("Records").reset_index()
                grouped.columns = [category_col, "Records"]
                y_col = "Records"
            else:
                temp = working[[category_col, measure]].copy()
                temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
                grouped_obj = temp.dropna(subset=[measure]).groupby(category_col, dropna=False)[measure]
                grouped_obj = grouped_obj.sum() if chart_metric.startswith("Sum") else grouped_obj.mean()
                grouped = grouped_obj.sort_values(ascending=False).head(limit).reset_index()
                y_col = measure
            if not grouped.empty:
                title_metric = "Number of records" if chart_metric == "Record count" else chart_metric
                st.plotly_chart(px.bar(grouped, x=category_col, y=y_col,
                                       title=f"{title_metric} by {category_col}", template="plotly_white"),
                                use_container_width=True, key=f"adaptive_category_bar_{_clean_name(base_name)}")
                if 2 <= len(grouped) <= 8:
                    st.plotly_chart(px.pie(grouped, names=category_col, values=y_col, hole=0.5,
                                           title=f"{title_metric} share by {category_col}"),
                                    use_container_width=True, key=f"adaptive_category_pie_{_clean_name(base_name)}")

        if measure:
            st.plotly_chart(px.histogram(working, x=measure, nbins=30,
                                         title=f"Distribution of {measure}", template="plotly_white"),
                            use_container_width=True, key=f"adaptive_hist_{_clean_name(base_name)}")

        corr_fields = numeric_options[:15]
        if len(corr_fields) >= 2:
            corr = working[corr_fields].apply(pd.to_numeric, errors="coerce").corr()
            if corr.shape[0] >= 2:
                st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto",
                                          title="Correlation between numeric measures",
                                          color_continuous_scale="RdBu", zmin=-1, zmax=1),
                                use_container_width=True, key=f"adaptive_corr_{_clean_name(base_name)}")

        if not date_options and not category_options and not numeric_options:
            st.info("No chart is displayed because the data does not support a meaningful visual.")

    with tabs[1]:
        st.caption("Detected potential join keys, identifiers and geographic coordinates.")
        if profile["identifiers"]:
            st.write("Identifier fields:", ", ".join(map(str, profile["identifiers"])))
        if profile["geo"]:
            st.write("Geographic fields:", ", ".join(map(str, profile["geo"])))
        if len(table_names) > 1:
            st.write("Uploaded tables:", ", ".join(table_names))
            st.info("Review key meaning and row counts before relying on joined metrics.")
        else:
            st.info("Upload related tables together to explore possible relationships.")

    with tabs[2]:
        st.dataframe(working.head(500), use_container_width=True)
        st.download_button("Download analysed table", data=working.to_csv(index=False).encode("utf-8"),
                           file_name="retailiq_analysed_data.csv", mime="text/csv")


# Detect Olist multi-table upload by its characteristic file names.
olist_orders = find_uploaded_file(uploaded_data, "olist_orders_dataset")
olist_items = find_uploaded_file(uploaded_data, "olist_order_items_dataset")
olist_products = find_uploaded_file(uploaded_data, "olist_products_dataset")
olist_customers = find_uploaded_file(uploaded_data, "olist_customers_dataset")
olist_sellers = find_uploaded_file(uploaded_data, "olist_sellers_dataset")
olist_reviews = find_uploaded_file(uploaded_data, "olist_order_reviews_dataset")
olist_translation = find_uploaded_file(uploaded_data, "product_category_name_translation")

is_olist_upload = any([
    olist_orders is not None,
    olist_items is not None,
    olist_products is not None,
    olist_customers is not None
])

# Keep the original retail transaction dashboard unchanged for compatible
# single-file transaction datasets, and keep the existing Olist path.
# Other schemas use the adaptive analytics workspace instead of being forced
# into Online Retail's InvoiceNo/StockCode/Quantity/UnitPrice assumptions.
if not is_olist_upload and (
    len(uploaded_files) > 1
    or not _retail_schema_is_compatible(next(iter(uploaded_data.values())))
):
    _render_adaptive_workspace(uploaded_data)
    st.stop()

if is_olist_upload:
    # Orders + order items are the minimum reliable grain for Olist sales.
    # Product, customer, seller, review and translation files enrich analysis
    # but are not required to load the dataset.
    missing_olist = []
    if olist_orders is None:
        missing_olist.append("olist_orders_dataset.csv")
    if olist_items is None:
        missing_olist.append("olist_order_items_dataset.csv")

    if missing_olist:
        st.error(
            "RetailIQ needs the Olist orders and order-items files to calculate "
            "reliable sales and date-based metrics. Missing: " + ", ".join(missing_olist)
        )
        st.info(
            "Product, customer, seller, review and category-translation files are "
            "optional. Upload them when available to unlock additional analysis."
        )
        st.stop()

    required_schemas = {
        "orders": {"order_id", "customer_id", "order_purchase_timestamp", "order_status"},
        "items": {"order_id", "order_item_id", "product_id", "seller_id", "price"},
    }
    supplied_tables = {"orders": olist_orders, "items": olist_items}
    schema_errors = []
    for table_name, required in required_schemas.items():
        missing = required - set(supplied_tables[table_name].columns)
        if missing:
            schema_errors.append(f"{table_name}: {sorted(missing)}")
    if schema_errors:
        st.error("The uploaded Olist files have unexpected columns: " + "; ".join(schema_errors))
        st.stop()

    optional_status = {
        "Products": olist_products is not None,
        "Customers": olist_customers is not None,
        "Sellers": olist_sellers is not None,
        "Reviews": olist_reviews is not None,
        "Category translation": olist_translation is not None,
    }
    unavailable = [name for name, available in optional_status.items() if not available]
    if unavailable:
        st.info(
            "Partial Olist upload detected. Available data will be analysed; "
            "these optional features are unavailable until you upload: "
            + ", ".join(unavailable) + "."
        )

    orders_for_join = olist_orders.copy()
    items_for_join = olist_items.copy()
    orders_for_join["order_purchase_timestamp"] = pd.to_datetime(
        orders_for_join["order_purchase_timestamp"], errors="coerce"
    )
    orders_for_join = orders_for_join.dropna(subset=["order_id"])
    items_for_join = items_for_join.dropna(subset=["order_id", "product_id"])
    items_for_join["price"] = pd.to_numeric(items_for_join["price"], errors="coerce")
    items_for_join = items_for_join.dropna(subset=["price"])
    items_for_join = items_for_join[items_for_join["price"] > 0]

    delivered_orders = orders_for_join[
        orders_for_join["order_status"].astype(str).str.lower().eq("delivered")
    ].copy()
    if delivered_orders.empty:
        st.error("No delivered Olist orders were found. Completed-sales analysis cannot continue.")
        st.stop()

    combined = items_for_join.merge(
        delivered_orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]],
        on="order_id", how="inner", validate="many_to_one"
    )

    # Optional product dimension: retain item rows even when product metadata is absent.
    if olist_products is not None and {"product_id", "product_category_name"}.issubset(olist_products.columns):
        products_for_join = olist_products.drop_duplicates(subset=["product_id"])
        combined = combined.merge(
            products_for_join, on="product_id", how="left", validate="many_to_one"
        )
        category_name = combined["product_category_name"].fillna("Uncategorised")
    else:
        # Without the optional products table, use product_id as a reliable
        # item-level label instead of grouping every item as "Uncategorised".
        combined["product_category_name"] = "Product ID"
        category_name = combined["product_id"].astype(str)

    # Optional customer dimension.
    if olist_customers is not None and {
        "customer_id", "customer_unique_id", "customer_city", "customer_state"
    }.issubset(olist_customers.columns):
        customers_for_join = olist_customers.drop_duplicates(subset=["customer_id"])
        combined = combined.merge(
            customers_for_join[[
                "customer_id", "customer_unique_id", "customer_city", "customer_state"
            ]],
            on="customer_id", how="left", validate="many_to_one"
        )
    else:
        combined["customer_unique_id"] = combined["customer_id"]
        combined["customer_city"] = np.nan
        combined["customer_state"] = np.nan

    if olist_translation is not None and {
        "product_category_name", "product_category_name_english"
    }.issubset(olist_translation.columns):
        translation = olist_translation.drop_duplicates(subset=["product_category_name"])
        combined = combined.merge(
            translation, on="product_category_name", how="left", validate="many_to_one"
        )
        category_name = combined["product_category_name_english"].fillna(
            combined["product_category_name"]
        )

    # Optional seller and review dimensions are joined without multiplying item rows.
    if olist_sellers is not None and {"seller_id", "seller_city", "seller_state"}.issubset(olist_sellers.columns):
        seller_dim = olist_sellers.drop_duplicates(subset=["seller_id"])
        combined = combined.merge(
            seller_dim[["seller_id", "seller_city", "seller_state"]],
            on="seller_id", how="left", validate="many_to_one"
        )

    if olist_reviews is not None and {"order_id", "review_score"}.issubset(olist_reviews.columns):
        review_summary = (
            olist_reviews.assign(review_score=pd.to_numeric(
                olist_reviews["review_score"], errors="coerce"
            ))
            .groupby("order_id", as_index=False)
            .agg(ReviewScore=("review_score", "mean"))
        )
        combined = combined.merge(
            review_summary, on="order_id", how="left", validate="many_to_one"
        )

    raw_df = pd.DataFrame({
        "InvoiceNo": combined["order_id"],
        "StockCode": combined["product_id"],
        "Description": category_name.fillna("Uncategorised"),
        "Quantity": 1,
        "InvoiceDate": combined["order_purchase_timestamp"],
        "UnitPrice": combined["price"],
        "Country": "Brazil",
        "CustomerID": combined["customer_unique_id"],
    })
    for optional_col in [
        "customer_city", "customer_state", "seller_id", "seller_city",
        "seller_state", "freight_value", "order_item_id", "ReviewScore",
        "product_category_name"
    ]:
        if optional_col in combined.columns:
            raw_df[optional_col] = combined[optional_col].values

    st.success(
        f"Olist data integrated: {len(raw_df):,} order-item rows "
        f"from {raw_df['InvoiceNo'].nunique():,} delivered orders."
    )
    st.caption(
        "Olist preparation: one row per order item; Quantity is represented as 1 per item row; "
        "revenue uses item price and excludes freight. Only delivered orders are included. "
        "When the products file is missing, product_id is used as the product label; "
        "category-level insights require the optional products file. "
        "Customer/seller/review insights depend on the optional files uploaded."
    )

else:
    if len(uploaded_files) != 1:
        st.error(
            "These files were not recognised as an Olist multi-file dataset. "
            "Please upload one standard transaction file, or the four required Olist files."
        )
        st.stop()

    raw_df = next(iter(uploaded_data.values()))

    required_columns = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice"
    ]
    column_aliases = {
        "InvoiceNo": ["invoiceno", "invoice", "invoiceid", "orderid", "ordernumber", "transactionid"],
        "StockCode": ["stockcode", "productid", "productcode", "sku", "itemcode"],
        "Description": ["description", "productname", "itemname", "product", "productcategoryname"],
        "Quantity": ["quantity", "qty", "unitssold", "numberofitems"],
        "InvoiceDate": ["invoicedate", "orderdate", "transactiondate", "date", "orderpurchasetimestamp"],
        "UnitPrice": ["unitprice", "sellingprice", "priceperunit", "price"],
        "Country": ["country", "customercountry", "market", "salescountry"],
        "CustomerID": ["customerid", "customeruniqueid", "clientid", "buyerid"],
    }
    column_lookup = {normalize_column(col): col for col in raw_df.columns}
    rename_map = {}
    used_columns = set()
    unmapped_columns = []

    for target in required_columns + ["Country", "CustomerID"]:
        matched = None
        for alias in column_aliases[target]:
            candidate = column_lookup.get(normalize_column(alias))
            if candidate is not None and candidate not in used_columns:
                matched = candidate
                break
        if matched is not None:
            rename_map[matched] = target
            used_columns.add(matched)
        elif target in required_columns:
            unmapped_columns.append(target)

    if unmapped_columns:
        st.subheader("A few columns need your attention")
        st.info(
            "RetailIQ AI could not confidently identify every required field. "
            "Select the matching columns below."
        )
        for target in unmapped_columns:
            available = [c for c in raw_df.columns if c not in used_columns]
            selected = st.selectbox(
                f"Select column for {target}",
                ["-- Select a column --"] + available,
                key=f"mapping_{target}"
            )
            if selected != "-- Select a column --":
                rename_map[selected] = target
                used_columns.add(selected)

    mapped_targets = list(rename_map.values())
    missing = [c for c in required_columns if c not in mapped_targets]
    if missing:
        st.warning(f"Please map these required fields: {missing}")
        st.stop()

    raw_df = raw_df.rename(columns=rename_map)
    if "Country" not in raw_df.columns:
        raw_df["Country"] = "Unknown"
        st.info("No country column was detected. Market analysis will show 'Unknown'.")
    if "CustomerID" not in raw_df.columns:
        raw_df["CustomerID"] = np.nan

    st.success("Dataset columns successfully mapped!")
    with st.expander("Review automatically detected columns"):
        mapping_df = pd.DataFrame(
            list(rename_map.items()),
            columns=["Original Column", "RetailIQ AI Column"]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)


# ==========================================
# DATA CLEANING
# ==========================================

df = raw_df.copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["Country"] = df["Country"].fillna("Unknown").astype(str)
df["Description"] = df["Description"].fillna("Uncategorised").astype(str)

cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
returns = df["Quantity"] < 0
return_count = int(returns.sum())
cancelled_count = int(cancelled.sum())

df = df[
    (~cancelled) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["InvoiceDate"].notna()) &
    (df["StockCode"].notna())
].copy()

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

if df.empty:
    st.error("No valid sales records found after data cleaning.")
    st.stop()


# ==========================================
# BUSINESS METRICS
# ==========================================

total_revenue = df["Revenue"].sum()

total_orders = df["InvoiceNo"].nunique()

total_units = df["Quantity"].sum()

total_products = df["StockCode"].nunique()

total_customers = (
    df["CustomerID"].nunique()
    if "CustomerID" in df.columns
    else 0
)

aov = (
    total_revenue / total_orders
    if total_orders else 0
)

# ==========================================
# ANALYSIS TABLES
# ==========================================

df["Month"] = (
    df["InvoiceDate"].dt.to_period("M")
    .dt.to_timestamp()
)

df["Date"] = (
    df["InvoiceDate"].dt.normalize()
)

monthly = (
    df.groupby("Month")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units=("Quantity", "sum")
    )
    .reset_index()
)

products = (
    df.groupby("Description")
    .agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique")
    )
    .sort_values("Revenue", ascending=False)
    .reset_index()
)

countries = (
    df.groupby("Country")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique")
    )
    .sort_values("Revenue", ascending=False)
    .reset_index()
)

daily = (
    df.groupby("Date")["Revenue"]
    .sum()
    .sort_index()
)

date_range = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D"
)

daily = daily.reindex(
    date_range, fill_value=0
)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Business Overview",
        "Sales Analytics",
        "Product Intelligence",
        "Market Analysis",
        "Sales Forecast",
        "AI Business Advisor"
    ]
)

# ==========================================
# PAGE 1: BUSINESS OVERVIEW
# ==========================================

if page == "Business Overview":

    st.header("Business Performance Overview")

    # ==========================================
    # KPI CARDS
    # ==========================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Revenue",
        f"{currency_symbol}{total_revenue:,.2f}"
    )

    c2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    c3.metric(
        "Average Order Value",
        f"{currency_symbol}{aov:,.2f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric("Units Sold", f"{total_units:,.0f}")
    c5.metric("Unique Products", f"{total_products:,}")
    c6.metric("Unique Customers", f"{total_customers:,}")

    # ==========================================
    # CHART THEME
    # ==========================================

    navy = "#14213D"
    teal = "#14B8A6"
    light_blue = "#DCE6F1"

    chart_colors = [
        navy, teal, "#38BDF8", "#64748B",
        "#A5B4FC", "#2DD4BF", "#94A3B8",
        "#0F766E"
    ]

    # ==========================================
    # MONTHLY REVENUE TREND
    # ==========================================

    st.subheader("Monthly Revenue Trend")

    fig_monthly = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        template="plotly_white"
    )

    fig_monthly.update_traces(
        line=dict(color=teal, width=3),
        marker=dict(size=7),
        hovertemplate=(
            "Month: %{x|%B %Y}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    fig_monthly.update_layout(
        title="Monthly Revenue Performance",
        title_font_color=navy,
        yaxis_title=f"Revenue ({selected_currency})",
        xaxis_title="Month",
        yaxis_tickprefix=currency_symbol,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    # ==========================================
    # COUNTRY REVENUE DONUT — DATA-DRIVEN VISIBILITY
    # ==========================================

    if countries["Country"].nunique() > 1:
        st.subheader("Revenue Distribution by Country")

        country_chart = countries.head(8).copy()

        if len(countries) > 8:
            other_revenue = countries.iloc[8:]["Revenue"].sum()
            country_chart = pd.concat(
                [
                    country_chart,
                    pd.DataFrame({
                        "Country": ["Other Countries"],
                        "Revenue": [other_revenue]
                    })
                ],
                ignore_index=True
            )

        fig_donut = px.pie(
            country_chart,
            names="Country",
            values="Revenue",
            hole=0.58,
            color_discrete_sequence=chart_colors
        )

        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "Country: %{label}<br>"
                f"Revenue: {currency_symbol}%{{value:,.2f}}"
                "<br>Share: %{percent}"
                "<extra></extra>"
            )
        )

        fig_donut.update_layout(
            title="Country-wise Revenue Share",
            title_font_color=navy,
            font=dict(color=navy),
            legend_title="Markets",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.caption(
            "Country-share chart hidden: only one country is present, "
            "so a country comparison would not add useful information."
        )

    # ==========================================
    # DAILY SALES TREND
    # ==========================================

    st.subheader("Daily Sales Performance")

    daily_chart = daily.rename_axis(
        "Date"
    ).reset_index(name="Revenue")

    fig_daily = px.line(
        daily_chart,
        x="Date",
        y="Revenue",
        template="plotly_white"
    )

    fig_daily.update_traces(
        line=dict(color=teal, width=2),
        hovertemplate=(
            "Date: %{x|%d %b %Y}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    fig_daily.update_layout(
        title="Daily Revenue Trend",
        title_font_color=navy,
        xaxis_title="Date",
        yaxis_title=f"Revenue ({selected_currency})",
        yaxis_tickprefix=currency_symbol,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )


    # ==========================================
    # ORDER VALUE DISTRIBUTION
    # ==========================================

    st.subheader("Order Value Distribution")

    order_values = (
        df.groupby("InvoiceNo")["Revenue"]
        .sum()
        .reset_index(name="OrderValue")
    )

    # Keep only positive order values
    positive_orders = order_values[
        order_values["OrderValue"] > 0
    ]["OrderValue"]

    if not positive_orders.empty:

        # Summary metrics
        avg_order = positive_orders.mean()
        median_order = positive_orders.median()
        max_order = positive_orders.max()

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Average Order Value",
            f"{currency_symbol}{avg_order:,.2f}"
        )

        m2.metric(
            "Median Order Value",
            f"{currency_symbol}{median_order:,.2f}"
        )

        m3.metric(
            "Highest Order Value",
            f"{currency_symbol}{max_order:,.2f}"
        )

        # Create logarithmic bins manually
        min_value = positive_orders.min()
        max_value = positive_orders.max()

        if min_value == max_value:
            bin_edges = np.linspace(
                min_value * 0.5,
                max_value * 1.5,
                11
            )
        else:
            bin_edges = np.geomspace(
                min_value,
                max_value * 1.001,
                21
            )

        counts, edges = np.histogram(
            positive_orders,
            bins=bin_edges
        )

        histogram_df = pd.DataFrame({
            "Order Range": [
                f"{currency_symbol}{edges[i]:,.0f} - "
                f"{currency_symbol}{edges[i+1]:,.0f}"
                for i in range(len(counts))
            ],
            "Number of Orders": counts
        })

        histogram_df = histogram_df[
            histogram_df["Number of Orders"] > 0
        ]

        fig_hist = px.bar(
            histogram_df,
            x="Order Range",
            y="Number of Orders",
            title="Order Value Distribution",
            color_discrete_sequence=["#14B8A6"],
            template="plotly_white"
        )

        fig_hist.update_traces(
            marker_line_color="#14213D",
            marker_line_width=0.5,
            hovertemplate=(
                "Order Range: %{x}<br>"
                "Number of Orders: %{y:,}"
                "<extra></extra>"
            )
        )

        fig_hist.update_layout(
            title_font_color="#14213D",
            xaxis_title="Order Value Ranges",
            yaxis_title="Number of Orders",
            xaxis_tickangle=-35,
            font=dict(color="#14213D"),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    else:
        st.warning("No positive order values available.")



    # ==========================================
    # GLOBAL SALES MAP
    # ==========================================
    if countries["Country"].nunique() > 1:
        st.subheader("Global Sales Map")

        # Convert country names into standard ISO-3 codes
        country_iso3 = {
            "United Kingdom": "GBR",
            "Netherlands": "NLD",
            "EIRE": "IRL",
            "Germany": "DEU",
            "France": "FRA",
            "Australia": "AUS",
            "Spain": "ESP",
            "Switzerland": "CHE",
            "Belgium": "BEL",
            "Sweden": "SWE",
            "Italy": "ITA",
            "Portugal": "PRT",
            "Norway": "NOR",
            "Finland": "FIN",
            "Denmark": "DNK",
            "Austria": "AUT",
            "Cyprus": "CYP",
            "Greece": "GRC",
            "Poland": "POL",
            "Japan": "JPN",
            "Singapore": "SGP",
            "Hong Kong": "HKG",
            "USA": "USA",
            "Canada": "CAN",
            "Brazil": "BRA",
            "Israel": "ISR",
            "Iceland": "ISL",
            "Bahrain": "BHR",
            "Lebanon": "LBN",
            "Lithuania": "LTU",
            "Malta": "MLT",
            "Saudi Arabia": "SAU",
            "United Arab Emirates": "ARE",
            "Czech Republic": "CZE",
            "RSA": "ZAF"
        }

        map_countries = countries.copy()

        map_countries["ISO3"] = (
            map_countries["Country"].map(country_iso3)
        )

        map_countries = map_countries.dropna(
            subset=["ISO3"]
        )

        if not map_countries.empty:

            fig_map = px.choropleth(
                map_countries,
                locations="ISO3",
                locationmode="ISO-3",
                color="Revenue",
                hover_name="Country",
                color_continuous_scale=[
                    "#DCE6F1",
                    "#14B8A6",
                    "#14213D"
                ],
                template="plotly_white"
            )

            fig_map.update_geos(
                fitbounds="locations",
                visible=True,
                showcountries=True,
                showcoastlines=True,
                showland=True,
                landcolor="#E2E8F0",
                bgcolor="white"
            )

            fig_map.update_layout(
                title="Worldwide Revenue Distribution",
                coloraxis_colorbar=dict(
                    title=f"Revenue ({selected_currency})"
                ),
                paper_bgcolor="white",
                font=dict(color="#14213D")
            )

            fig_map.update_traces(
                hovertemplate=(
                    "Country: %{hovertext}<br>"
                    f"Revenue: {currency_symbol}%{{z:,.2f}}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                fig_map,
                use_container_width=True
            )

        else:
            st.warning("No recognized country locations found.")


    else:
        st.caption(
            "Global country map hidden: the dataset contains only one country. "
            "A geographic comparison will appear when multiple countries are available."
        )


    # ==========================================
    # TOP PERFORMING PRODUCTS
    # ==========================================

    st.subheader("Top Performing Products")

    display_products = products.head(10).copy()

    display_products["Revenue"] = (
        display_products["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_products,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 2: SALES ANALYTICS
# ==========================================


elif page == "Sales Analytics":

    st.header("Sales Analytics")
    navy = "#14213D"
    teal = "#14B8A6"

    # ==========================================
    # MONTHLY REVENUE
    # ==========================================

    st.subheader("Monthly Revenue Performance")

    fig = px.bar(
        monthly,
        x="Month",
        y="Revenue",
        title="Monthly Revenue",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig.update_layout(
        yaxis_title=f"Revenue ({selected_currency})",
        xaxis_title="Month",
        yaxis_tickprefix=currency_symbol,
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_traces(
        hovertemplate=(
            "Month: %{x|%B %Y}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # MONTHLY ORDERS
    # ==========================================

    st.subheader("Monthly Order Trends")

    fig2 = px.line(
        monthly,
        x="Month",
        y="Orders",
        markers=True,
        title="Monthly Orders",
        color_discrete_sequence=[navy],
        template="plotly_white"
    )

    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Orders",
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # DAILY SALES WITH DATE FILTER
    # ==========================================

    st.subheader("Daily Sales Analysis")

    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_df = df[
            df["InvoiceDate"].dt.date.between(
                start_date, end_date
            )
        ].copy()

        if not filtered_df.empty:

            filtered_daily = (
                filtered_df.groupby(
                    filtered_df["InvoiceDate"].dt.date
                )["Revenue"]
                .sum()
                .reset_index()
            )

            filtered_daily.columns = ["Date", "Revenue"]

            fig3 = px.area(
                filtered_daily,
                x="Date",
                y="Revenue",
                title="Daily Revenue Trend",
                color_discrete_sequence=[teal],
                template="plotly_white"
            )

            fig3.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Revenue ({selected_currency})",
                yaxis_tickprefix=currency_symbol,
                title_font_color=navy,
                font=dict(color=navy),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        else:
            st.warning("No sales found in this date range.")

    # ==========================================
    # WEEKDAY REVENUE ANALYSIS
    # ==========================================

    st.subheader("Revenue by Day of the Week")

    weekday_df = df.copy()

    weekday_df["Weekday"] = (
        weekday_df["InvoiceDate"].dt.day_name()
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_sales = (
        weekday_df.groupby("Weekday")["Revenue"]
        .sum()
        .reindex(weekday_order, fill_value=0)
        .reset_index()
    )

    fig4 = px.bar(
        weekday_sales,
        x="Weekday",
        y="Revenue",
        title="Which Days Generate the Most Revenue?",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig4.update_layout(
        xaxis_title="Day of the Week",
        yaxis_title=f"Revenue ({selected_currency})",
        yaxis_tickprefix=currency_symbol,
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ==========================================
    # MONTHLY PERFORMANCE TABLE
    # ==========================================

    st.subheader("Monthly Performance Data")

    display_monthly = monthly.copy()

    display_monthly["Revenue"] = (
        display_monthly["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_monthly,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 3: PRODUCT INTELLIGENCE
# ==========================================


elif page == "Product Intelligence":

    st.header("Product Intelligence")

    navy = "#14213D"
    teal = "#14B8A6"

    # ==========================================
    # TOP PRODUCTS BY REVENUE
    # ==========================================

    st.subheader("Top 10 Products by Revenue")

    top_products = products.head(10).copy()

    fig = px.bar(
        top_products.sort_values("Revenue"),
        x="Revenue",
        y="Description",
        orientation="h",
        color="Revenue",
        color_continuous_scale=["#14B8A6", "#14213D"],
        template="plotly_white"
    )

    fig.update_layout(
        title="Highest Revenue Generating Products",
        xaxis_title=f"Revenue ({selected_currency})",
        xaxis_tickprefix=currency_symbol,
        yaxis_title="Product",
        title_font_color=navy,
        font=dict(color=navy),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_traces(
        hovertemplate=(
            "Product: %{y}<br>"
            f"Revenue: {currency_symbol}%{{x:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PRODUCT REVENUE DONUT
    # ==========================================

    st.subheader("Product Revenue Contribution")

    product_donut = products.head(8).copy()

    if len(products) > 8:

        other_revenue = products.iloc[8:]["Revenue"].sum()

        product_donut = pd.concat(
            [
                product_donut,
                pd.DataFrame({
                    "Description": ["Other Products"],
                    "Revenue": [other_revenue]
                })
            ],
            ignore_index=True
        )

    fig2 = px.pie(
        product_donut,
        names="Description",
        values="Revenue",
        hole=0.58,
        color_discrete_sequence=[
            navy, teal, "#38BDF8", "#64748B",
            "#A5B4FC", "#2DD4BF", "#94A3B8",
            "#0F766E", "#CBD5E1"
        ]
    )

    fig2.update_traces(
        textinfo="percent",
        hovertemplate=(
            "Product: %{label}<br>"
            f"Revenue: {currency_symbol}%{{value:,.2f}}"
            "<br>Share: %{percent}"
            "<extra></extra>"
        )
    )

    fig2.update_layout(
        title="Revenue Share by Product",
        title_font_color=navy,
        font=dict(color=navy),
        paper_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # TOP PRODUCTS BY QUANTITY
    # ==========================================

    st.subheader("Top 10 Products by Quantity Sold")

    quantity_top = products.nlargest(
        10, "Units"
    ).sort_values("Units")

    fig3 = px.bar(
        quantity_top,
        x="Units",
        y="Description",
        orientation="h",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig3.update_layout(
        title="Best Selling Products by Quantity",
        xaxis_title="Units Sold",
        yaxis_title="Product",
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # PRODUCT PERFORMANCE TABLE
    # ==========================================

    st.subheader("Complete Product Performance")

    display_products = products.copy()

    display_products["Revenue"] = (
        display_products["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_products,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 4: MARKET ANALYSIS
# ==========================================



elif page == "Market Analysis":

    st.header("Market Analysis")

    country_count = countries["Country"].nunique()

    if country_count > 1:
        st.subheader("Country-wise Revenue Comparison")
        fig = px.bar(
            countries.head(10),
            x="Country",
            y="Revenue",
            title="Top Countries by Revenue",
            template="plotly_white"
        )
        fig.update_layout(
            yaxis_title=f"Revenue ({selected_currency})",
            yaxis_tickprefix=currency_symbol
        )
        fig.update_traces(
            hovertemplate=(
                "Country: %{x}<br>"
                f"Revenue: {currency_symbol}%{{y:,.2f}}"
                "<extra></extra>"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        display_countries = countries.copy()
        display_countries["Revenue"] = display_countries["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
        st.dataframe(display_countries, use_container_width=True, hide_index=True)

        # Only show this legacy KPI when the selected data actually contains UK.
        uk_rows = countries[
            countries["Country"].astype(str).str.casefold() == "united kingdom"
        ]
        if not uk_rows.empty and total_revenue:
            uk_revenue = uk_rows["Revenue"].sum()
            st.metric(
                "United Kingdom Revenue Contribution",
                f"{uk_revenue / total_revenue * 100:.1f}%"
            )
    elif country_count == 1:
        only_market = countries.iloc[0]
        st.subheader("Available Market")
        m1, m2 = st.columns(2)
        m1.metric("Detected Country", str(only_market["Country"]))
        m2.metric("Revenue", f"{currency_symbol}{only_market['Revenue']:,.2f}")
        st.metric("Orders", f"{int(only_market['Orders']):,}")
        st.caption(
            "Country comparison charts and country-share metrics are hidden "
            "because only one country is present in this dataset."
        )
        display_countries = countries.copy()
        display_countries["Revenue"] = display_countries["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
        st.dataframe(display_countries, use_container_width=True, hide_index=True)
    else:
        st.info("No country values are available for market analysis.")


# ==========================================
# PAGE 5: SALES FORECAST
# ==========================================

elif page == "Sales Forecast":

    st.header("Sales Forecasting")

    if len(daily) < 28:

        st.warning(
            "At least 28 days of sales data is required "
            "for this forecasting model."
        )

    else:

        try:

            seasonal = len(daily) >= 60

            model = ExponentialSmoothing(
                daily,
                trend="add",
                seasonal="add" if seasonal else None,
                seasonal_periods=7 if seasonal else None,
                initialization_method="estimated"
            )

            fitted = model.fit()

            forecast = fitted.forecast(30)

            forecast_df = pd.DataFrame({
                "Date": forecast.index,
                "Predicted Revenue": forecast.values
            })

            
            st.metric(
                "Next 30-Day Forecast",
                f"{currency_symbol}{forecast.sum():,.2f}"
            )
            
            fig = px.line(
                forecast_df,
                x="Date",
                y="Predicted Revenue",
                markers=True,
                title="Future Revenue Forecast",
                template="plotly_white"
            )
            
            fig.update_layout(
                yaxis_title=f"Predicted Revenue ({selected_currency})",
                yaxis_tickprefix=currency_symbol
            )
            
            fig.update_traces(
                hovertemplate=
                f"Date: %{{x}}<br>"
                f"Predicted Revenue: {currency_symbol}%{{y:,.2f}}"
                "<extra></extra>"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

           
            st.dataframe(
                forecast_df.style.format({
                    "Predicted Revenue": f"{currency_symbol}{{:,.2f}}"
                }),
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:
            st.error(f"Forecasting error: {e}")

# ==========================================
# PAGE 6: AI BUSINESS ADVISOR
# ==========================================

elif page == "AI Business Advisor":

    st.header("✨ AI Business Advisor")

    best_month = monthly.loc[
        monthly["Revenue"].idxmax()
    ]

    best_country = countries.iloc[0]
    best_product = products.iloc[0]

    st.subheader("Automated Business Insights")

    st.info(
        f"Your highest-revenue month was "
        f"{best_month['Month'].strftime('%B %Y')} "
        f"with {currency_symbol}{best_month['Revenue']:,.2f}."
    )

    st.info(
        f"Your leading market is {best_country['Country']} "
        f"with {currency_symbol}{best_country['Revenue']:,.2f} revenue."
    )

    st.info(
        f"Your highest-revenue product is "
        f"{best_product['Description']}."
    )

    if len(daily) >= 28:

        st.success(
            "Your sales forecasting feature is available. "
            "Use the Sales Forecast page to inspect predictions."
        )

    st.subheader("Ask RetailIQ")

    question = st.text_input(
        "Ask a question about your business:"
    )

    if st.button("Generate Answer") and question:

        q = question.lower()

        if "country" in q or "market" in q:

            answer = (
                f"{best_country['Country']} leads with "
                f"{currency_symbol}{best_country['Revenue']:,.2f} revenue."
            )

        elif "product" in q:

            answer = (
                f"{best_product['Description']} is the "
                f"highest-revenue product with "
                f"{currency_symbol}{best_product['Revenue']:,.2f} revenue."
            )

        elif "month" in q:

            answer = (
                f"{best_month['Month'].strftime('%B %Y')} "
                f"was the best month, generating "
                f"{currency_symbol}{best_month['Revenue']:,.2f}."
            )

        elif "customer" in q:

            answer = (
                f"The dataset contains {total_customers:,} "
                "identified customers."
            )

        elif "order" in q:

            answer = f"Total valid orders: {total_orders:,}."

        elif "sales" in q or "revenue" in q:

            answer = (
                f"Total revenue is {currency_symbol}{total_revenue:,.2f}."
            )

        else:

            answer = (
                "Try asking about revenue, sales, products, "
                "countries, customers or monthly performance."
            )

        st.success(answer)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RetailIQ AI | Smart Retail Analytics & "
    "Business Decision Support | Portfolio MVP"
)
