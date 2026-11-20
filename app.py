
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
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
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
    name = _clean_name(column)
    identifier_tokens = (
        "id", "code", "sku", "uuid", "key", "zipcode", "postalcode",
        "phonenumber", "invoice", "ordernumber", "transactionnumber"
    )
    return any(token in name for token in identifier_tokens)


def _profile_table(frame):
    profile = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "numeric": [],
        "dates": [],
        "categorical": [],
        "identifiers": [],
        "missing": {}
    }
    for col in frame.columns:
        series = frame[col]
        profile["missing"][col] = int(series.isna().sum())

        if pd.api.types.is_datetime64_any_dtype(series):
            profile["dates"].append(col)
            continue

        name = _clean_name(col)
        date_name = any(token in name for token in (
            "date", "time", "timestamp", "created", "updated", "month", "year"
        ))
        if date_name and not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            if len(series.dropna()) and parsed.notna().mean() >= 0.65:
                profile["dates"].append(col)
                continue

        converted = pd.to_numeric(series, errors="coerce")
        numeric_ratio = converted.notna().mean() if len(series) else 0
        if pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.85:
            if _is_identifier_name(col):
                profile["identifiers"].append(col)
            else:
                profile["numeric"].append(col)
        elif series.nunique(dropna=True) <= max(50, int(len(series) * 0.20)):
            profile["categorical"].append(col)
        elif _is_identifier_name(col):
            profile["identifiers"].append(col)

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


def _render_adaptive_workspace(tables):
    """Generic, evidence-led analytics for non-legacy and multi-table uploads."""
    st.title("RetailIQ AI")
    st.subheader("Adaptive Analytics & Data Explorer")
    st.caption(
        "This dataset does not match the original Online Retail transaction schema. "
        "RetailIQ is profiling the actual columns and will only create analyses supported by the data."
    )

    st.success("Adaptive mode enabled — the original Online Retail dashboard logic has not been applied.")

    table_names = list(tables.keys())
    if not table_names:
        st.error("No readable tables were found.")
        return

    # Show a compact inventory of all uploaded tables.
    inventory = pd.DataFrame([
        {"Table": name, "Rows": len(tables[name]), "Columns": len(tables[name].columns)}
        for name in table_names
    ])
    with st.expander("Uploaded data inventory", expanded=len(table_names) > 1):
        st.dataframe(inventory, use_container_width=True, hide_index=True)

    working = None
    base_name = None

    if len(table_names) == 1:
        base_name = table_names[0]
        working = tables[base_name].copy()
    else:
        st.subheader("Choose a primary table")
        default_base = max(table_names, key=lambda n: len(tables[n]))
        base_name = st.selectbox(
            "Primary table for analysis",
            table_names,
            index=table_names.index(default_base),
            key="adaptive_primary_table"
        )
        working = tables[base_name].copy()

        st.subheader("Optional table relationships")
        st.caption(
            "Choose matching key columns only when they represent the same entity. "
            "Review row counts after each join; many-to-many joins can multiply records."
        )
        for idx, other_name in enumerate([n for n in table_names if n != base_name]):
            other = tables[other_name]
            with st.expander(f"Join {other_name}", expanded=False):
                common = [c for c in working.columns if c in other.columns]
                left_key_options = list(working.columns)
                right_key_options = list(other.columns)
                include = st.checkbox(
                    f"Include {other_name}",
                    value=False,
                    key=f"adaptive_join_include_{idx}"
                )
                if not include:
                    st.caption(
                        f"Potential common columns: {', '.join(common[:8]) if common else 'None detected'}"
                    )
                    continue

                suggested_left = common[0] if common else left_key_options[0]
                suggested_right = suggested_left if suggested_left in right_key_options else right_key_options[0]
                c1, c2 = st.columns(2)
                left_key = c1.selectbox(
                    "Primary-table key",
                    left_key_options,
                    index=left_key_options.index(suggested_left),
                    key=f"adaptive_left_key_{idx}"
                )
                right_key = c2.selectbox(
                    "Related-table key",
                    right_key_options,
                    index=right_key_options.index(suggested_right),
                    key=f"adaptive_right_key_{idx}"
                )
                join_how = st.selectbox(
                    "Join type",
                    ["left", "inner", "outer"],
                    key=f"adaptive_join_how_{idx}"
                )
                if working[left_key].duplicated().any() and other[right_key].duplicated().any():
                    st.warning(
                        "Both selected keys contain duplicates. This is a many-to-many join "
                        "and may multiply rows. Confirm that this relationship is intended."
                    )
                if st.button(f"Apply join: {other_name}", key=f"adaptive_apply_join_{idx}"):
                    try:
                        working = working.merge(
                            other,
                            left_on=left_key,
                            right_on=right_key,
                            how=join_how,
                            suffixes=("", f"_{_clean_name(other_name)[:12]}")
                        )
                        st.session_state["adaptive_joined_frame"] = working
                        st.session_state["adaptive_joined_base"] = base_name
                        st.success(
                            f"Join applied. Result: {len(working):,} rows and {len(working.columns):,} columns."
                        )
                    except Exception as exc:
                        st.error(f"Join could not be completed: {exc}")
                    st.rerun()

        if (
            st.session_state.get("adaptive_joined_base") == base_name
            and "adaptive_joined_frame" in st.session_state
        ):
            working = st.session_state["adaptive_joined_frame"].copy()

    if working is None or working.empty:
        st.warning("The selected table is empty.")
        return

    # Normalize obvious date and numeric columns for analysis without changing source columns.
    profile = _profile_table(working)
    date_options = profile["dates"]
    numeric_options = profile["numeric"]
    category_options = profile["categorical"]

    st.subheader("Data quality and structure")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(working):,}")
    k2.metric("Columns", f"{len(working.columns):,}")
    k3.metric("Numeric measures", f"{len(numeric_options):,}")
    k4.metric("Date fields", f"{len(date_options):,}")

    profile_rows = []
    for col in working.columns:
        profile_rows.append({
            "Column": str(col),
            "Detected type": (
                "Date" if col in profile["dates"] else
                "Numeric measure" if col in profile["numeric"] else
                "Identifier" if col in profile["identifiers"] else
                "Category / text"
            ),
            "Missing": profile["missing"][col],
            "Unique values": int(working[col].nunique(dropna=True))
        })
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    if not numeric_options and not category_options and not date_options:
        st.info("The uploaded table has no suitable numeric, date, or categorical fields for charts.")
        st.subheader("Data preview")
        st.dataframe(working.head(100), use_container_width=True)
        return

    st.subheader("Adaptive overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total records", f"{len(working):,}")
    metric_cols[1].metric("Distinct categories", f"{sum(working[c].nunique(dropna=True) for c in category_options):,}")
    metric_cols[2].metric("Numeric fields", f"{len(numeric_options):,}")
    metric_cols[3].metric("Missing cells", f"{int(working.isna().sum().sum()):,}")

    # A numeric field is selected by the user from detected measures; no revenue/profit
    # label is invented when the uploaded data does not contain one.
    if numeric_options:
        measure = st.selectbox(
            "Choose a numeric measure to analyse",
            numeric_options,
            key="adaptive_measure"
        )
        measure_values = pd.to_numeric(working[measure], errors="coerce").dropna()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Total {measure}", f"{measure_values.sum():,.2f}")
        m2.metric(f"Average {measure}", f"{measure_values.mean():,.2f}" if len(measure_values) else "—")
        m3.metric(f"Median {measure}", f"{measure_values.median():,.2f}" if len(measure_values) else "—")
    else:
        measure = None
        st.info("No numeric measure was detected. Numeric KPIs and measure-based charts are hidden.")

    tabs = st.tabs(["Visual analysis", "Relationships", "Data preview"])

    with tabs[0]:
        if date_options and measure:
            date_col = st.selectbox("Date field", date_options, key="adaptive_date")
            temp = working[[date_col, measure]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                trend = temp.set_index(date_col).resample("MS")[measure].sum().reset_index()
                st.plotly_chart(
                    px.line(trend, x=date_col, y=measure, markers=True,
                            title=f"{measure} over time", template="plotly_white"),
                    use_container_width=True
                )
            else:
                st.caption("The selected date and measure fields do not contain enough valid paired values.")

        if category_options and measure:
            category_col = st.selectbox(
                "Category field",
                category_options,
                key="adaptive_category"
            )
            limit = st.slider("Number of categories to display", 5, 25, 10, key="adaptive_category_limit")
            temp = working[[category_col, measure]].copy()
            temp[measure] = pd.to_numeric(temp[measure], errors="coerce")
            grouped = (
                temp.dropna(subset=[category_col, measure])
                .groupby(category_col, dropna=False)[measure]
                .sum()
                .sort_values(ascending=False)
                .head(limit)
                .reset_index()
            )
            if not grouped.empty:
                st.plotly_chart(
                    px.bar(grouped, x=category_col, y=measure, title=f"{measure} by {category_col}",
                           template="plotly_white"),
                    use_container_width=True
                )
                if 2 <= len(grouped) <= 8:
                    st.plotly_chart(
                        px.pie(grouped, names=category_col, values=measure, hole=0.5,
                               title=f"{measure} share by {category_col}"),
                        use_container_width=True
                    )

        if measure:
            st.plotly_chart(
                px.histogram(
                    working,
                    x=measure,
                    nbins=30,
                    title=f"Distribution of {measure}",
                    template="plotly_white"
                ),
                use_container_width=True
            )

        if len(numeric_options) >= 2:
            corr = working[numeric_options].apply(pd.to_numeric, errors="coerce").corr()
            if corr.shape[0] >= 2:
                st.plotly_chart(
                    px.imshow(corr, text_auto=".2f", aspect="auto",
                              title="Correlation between numeric measures",
                              color_continuous_scale="RdBu", zmin=-1, zmax=1),
                    use_container_width=True
                )

        if not date_options and not category_options and not numeric_options:
            st.info("No chart is displayed because the data does not support a meaningful visual.")

    with tabs[1]:
        st.caption("Detected potential join keys and identifier fields.")
        if profile["identifiers"]:
            st.write("Identifier fields:", ", ".join(map(str, profile["identifiers"])))
        else:
            st.info("No obvious identifier fields were detected in this table.")
        if len(table_names) > 1:
            st.write("Uploaded tables:", ", ".join(table_names))
            st.info("Use the optional table-join controls above. Verify key meaning and row counts before relying on joined metrics.")
        else:
            st.info("Upload related tables together to explore possible relationships.")

    with tabs[2]:
        st.dataframe(working.head(500), use_container_width=True)
        st.download_button(
            "Download analysed table",
            data=working.to_csv(index=False).encode("utf-8"),
            file_name="retailiq_analysed_data.csv",
            mime="text/csv"
        )


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
    missing_olist = []
    for label, table in [
        ("olist_orders_dataset.csv", olist_orders),
        ("olist_order_items_dataset.csv", olist_items),
        ("olist_products_dataset.csv", olist_products),
        ("olist_customers_dataset.csv", olist_customers),
    ]:
        if table is None:
            missing_olist.append(label)

    if missing_olist:
        st.error(
            "Olist upload is incomplete. Please upload these required files together: "
            + ", ".join(missing_olist)
        )
        st.stop()

    required_schemas = {
        "orders": {"order_id", "customer_id", "order_purchase_timestamp", "order_status"},
        "items": {"order_id", "order_item_id", "product_id", "seller_id", "price"},
        "products": {"product_id", "product_category_name"},
        "customers": {"customer_id", "customer_unique_id", "customer_city", "customer_state"},
    }
    supplied_tables = {
        "orders": olist_orders,
        "items": olist_items,
        "products": olist_products,
        "customers": olist_customers,
    }
    schema_errors = []
    for table_name, required in required_schemas.items():
        missing = required - set(supplied_tables[table_name].columns)
        if missing:
            schema_errors.append(f"{table_name}: {sorted(missing)}")
    if schema_errors:
        st.error("The uploaded Olist files have unexpected columns: " + "; ".join(schema_errors))
        st.stop()

    # Keep the order-item grain: one row represents one product line in an order.
    # Join only one-to-one / many-to-one tables to avoid multiplying sales rows.
    orders_for_join = olist_orders.copy()
    items_for_join = olist_items.copy()
    products_for_join = olist_products.copy()
    customers_for_join = olist_customers.copy()

    orders_for_join["order_purchase_timestamp"] = pd.to_datetime(
        orders_for_join["order_purchase_timestamp"], errors="coerce"
    )
    orders_for_join = orders_for_join.dropna(subset=["order_id"])
    items_for_join = items_for_join.dropna(subset=["order_id", "product_id"])

    # Use delivered orders for completed-sales KPIs; report the status filter to the user.
    delivered_orders = orders_for_join[
        orders_for_join["order_status"].astype(str).str.lower().eq("delivered")
    ].copy()
    if delivered_orders.empty:
        st.error("No delivered orders were found. Check the Olist orders file.")
        st.stop()

    # Remove accidental duplicate dimension keys before joining.
    products_for_join = products_for_join.drop_duplicates(subset=["product_id"])
    customers_for_join = customers_for_join.drop_duplicates(subset=["customer_id"])

    combined = items_for_join.merge(
        delivered_orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]],
        on="order_id", how="inner", validate="many_to_one"
    )
    combined = combined.merge(
        products_for_join, on="product_id", how="left", validate="many_to_one"
    )
    combined = combined.merge(
        customers_for_join[[
            "customer_id", "customer_unique_id", "customer_city", "customer_state"
        ]],
        on="customer_id", how="left", validate="many_to_one"
    )

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
    else:
        category_name = combined["product_category_name"]

    # Seller and review data are supplementary; review rows are aggregated first
    # so they cannot multiply order-item records.
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
    # Preserve useful Olist attributes for later customer/region/seller analysis.
    for optional_col in [
        "customer_city", "customer_state", "seller_id", "seller_city",
        "seller_state", "freight_value", "order_item_id", "ReviewScore",
        "product_category_name"
    ]:
        if optional_col in combined.columns:
            raw_df[optional_col] = combined[optional_col].values

    st.success(
        f"Olist dataset integrated successfully: {len(raw_df):,} order-item rows "
        f"from {raw_df['InvoiceNo'].nunique():,} delivered orders."
    )
    st.caption(
        "Olist preparation: one row per order item; quantity is represented as 1 per item row; "
        "sales revenue uses item price and excludes freight. Only delivered orders are included."
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
