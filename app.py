
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
    "Select Dataset Currency",
    options=list(currency_options.keys()),
    format_func=lambda x: currency_options[x]["name"]
)

currency_symbol = currency_options[selected_currency]["symbol"]

uploaded_file = st.sidebar.file_uploader(
    "Upload your retail dataset",
    type=["csv", "xlsx"]
)


st.sidebar.info(
    "Upload your retail dataset in CSV or Excel format. "
    "RetailIQ AI automatically detects common column names "
    "and lets you map columns manually when needed. "
    "Required data: Order ID, Product, Quantity, Date, "
    "Unit Price and Country."
)


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data(file_bytes, file_name):

    import io

    if file_name.lower().endswith(".csv"):
        data = pd.read_csv(io.BytesIO(file_bytes))
    else:
        data = pd.read_excel(io.BytesIO(file_bytes))

    return data


if uploaded_file is None:

    st.warning(
        "Please upload a CSV or Excel retail dataset "
        "using the sidebar to begin."
    )

    st.markdown("""
    ### Welcome to RetailIQ AI

    Analyse your retail business with:

    - Sales and revenue intelligence
    - Product performance analytics
    - Market analysis
    - Sales forecasting
    - AI-assisted business recommendations

    **Getting started:** Upload your retail transaction data.
    """)

    st.stop()


try:
    raw_df = load_data(
        uploaded_file.getvalue(),
        uploaded_file.name
    )

except Exception as e:
    st.error(f"Unable to read the file: {e}")
    st.stop()



# ==========================================
# SMART DATASET COLUMN MAPPING
# ==========================================

required_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country"
]

# Normalize column names for comparison
def normalize_column(col):
    return "".join(
        ch.lower() for ch in str(col) if ch.isalnum()
    )

column_lookup = {
    normalize_column(col): col
    for col in raw_df.columns
}

# Alternative names for common retail columns
column_aliases = {
    "InvoiceNo": [
        "invoiceno", "invoice", "invoiceid",
        "orderid", "ordernumber", "transactionid"
    ],
    "StockCode": [
        "stockcode", "productid", "productcode",
        "sku", "itemcode"
    ],
    "Description": [
        "description", "productname",
        "itemname", "product"
    ],
    "Quantity": [
        "quantity", "qty", "unitssold",
        "numberofitems"
    ],
    "InvoiceDate": [
        "invoicedate", "orderdate",
        "transactiondate", "date"
    ],
    "UnitPrice": [
        "unitprice", "sellingprice",
        "priceperunit", "price"
    ],
    "Country": [
        "country", "customercountry",
        "market", "region"
    ]
}

rename_map = {}
used_columns = set()
unmapped_columns = []

# Automatically detect matching columns
for target in required_columns:

    matched_column = None

    for alias in column_aliases[target]:

        normalized_alias = normalize_column(alias)

        if normalized_alias in column_lookup:

            candidate = column_lookup[normalized_alias]

            if candidate not in used_columns:
                matched_column = candidate
                break

    if matched_column is not None:

        rename_map[matched_column] = target
        used_columns.add(matched_column)

    else:
        unmapped_columns.append(target)


# Ask the user to map columns that were not detected
if unmapped_columns:

    st.subheader("Map Your Dataset Columns")

    st.info(
        "Some columns could not be detected automatically. "
        "Please select the matching columns below."
    )

    for target in unmapped_columns:

        available_columns = [
            col for col in raw_df.columns
            if col not in used_columns
        ]

        selected_column = st.selectbox(
            f"Select column for {target}",
            ["-- Select a column --"] + available_columns,
            key=f"mapping_{target}"
        )

        if selected_column != "-- Select a column --":

            rename_map[selected_column] = target
            used_columns.add(selected_column)


# Stop if required columns are still missing
mapped_targets = list(rename_map.values())

missing_columns = [
    col for col in required_columns
    if col not in mapped_targets
]

if missing_columns:

    st.warning(
        f"Please map these required fields: {missing_columns}"
    )

    st.stop()


# Rename uploaded columns to standard RetailIQ AI names
raw_df = raw_df.rename(columns=rename_map)

st.success("Dataset columns successfully mapped!")

with st.expander("View detected column mapping"):

    mapping_df = pd.DataFrame(
        list(rename_map.items()),
        columns=["Original Column", "RetailIQ AI Column"]
    )

    st.dataframe(
        mapping_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# DATA CLEANING
# ==========================================

df = raw_df.copy()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

cancelled = (
    df["InvoiceNo"].astype(str).str.startswith("C")
)

returns = df["Quantity"] < 0

return_count = int(returns.sum())
cancelled_count = int(cancelled.sum())

df = df[
    (~cancelled) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["Description"].notna()) &
    (df["InvoiceDate"].notna()) &
    (df["Country"].notna())
].copy()

df["Revenue"] = (
    df["Quantity"] * df["UnitPrice"]
)

if df.empty:
    st.error("No valid sales records found.")
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
    # COUNTRY REVENUE DONUT
    # ==========================================

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

    st.plotly_chart(
        fig_donut,
        use_container_width=True
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

    st.header("Market Intelligence")

    navy = "#14213D"
    teal = "#14B8A6"

    # ==========================================
    # COUNTRY KPI METRICS
    # ==========================================

    top_country = countries.iloc[0]

    total_markets = countries["Country"].nunique()

    market_orders = countries["Orders"].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Countries Served",
        f"{total_markets:,}"
    )

    c2.metric(
        "Leading Market",
        top_country["Country"]
    )

    c3.metric(
        "Total Market Revenue",
        f"{currency_symbol}{total_revenue:,.2f}"
    )

    # ==========================================
    # TOP 10 COUNTRIES BY REVENUE
    # ==========================================

    st.subheader("Top 10 Countries by Revenue")

    top_countries = countries.head(10).copy()

    fig = px.bar(
        top_countries.sort_values("Revenue"),
        x="Revenue",
        y="Country",
        orientation="h",
        color="Revenue",
        color_continuous_scale=["#14B8A6", "#14213D"],
        template="plotly_white"
    )

    fig.update_layout(
        title="Highest Revenue Generating Markets",
        xaxis_title=f"Revenue ({selected_currency})",
        xaxis_tickprefix=currency_symbol,
        yaxis_title="Country",
        title_font_color=navy,
        font=dict(color=navy),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_traces(
        hovertemplate=(
            "Country: %{y}<br>"
            f"Revenue: {currency_symbol}%{{x:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # COUNTRY REVENUE DONUT
    # ==========================================

    st.subheader("Revenue Contribution by Country")

    country_donut = countries.head(8).copy()

    if len(countries) > 8:

        other_revenue = countries.iloc[8:]["Revenue"].sum()

        country_donut = pd.concat(
            [
                country_donut,
                pd.DataFrame({
                    "Country": ["Other Countries"],
                    "Revenue": [other_revenue]
                })
            ],
            ignore_index=True
        )

    fig2 = px.pie(
        country_donut,
        names="Country",
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
            "Country: %{label}<br>"
            f"Revenue: {currency_symbol}%{{value:,.2f}}"
            "<br>Share: %{percent}"
            "<extra></extra>"
        )
    )

    fig2.update_layout(
        title="Country Revenue Share",
        title_font_color=navy,
        font=dict(color=navy),
        paper_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # COUNTRY ORDER COMPARISON
    # ==========================================

    st.subheader("Orders by Country")

    fig3 = px.bar(
        countries.head(10).sort_values("Orders"),
        x="Orders",
        y="Country",
        orientation="h",
        color_discrete_sequence=[teal],
        template="plotly_white"
    )

    fig3.update_layout(
        title="Top 10 Markets by Number of Orders",
        xaxis_title="Total Orders",
        yaxis_title="Country",
        title_font_color=navy,
        font=dict(color=navy),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # COUNTRY PERFORMANCE TABLE
    # ==========================================

    st.subheader("Country Performance Details")

    display_countries = countries.copy()

    display_countries["Revenue"] = (
        display_countries["Revenue"].apply(
            lambda x: f"{currency_symbol}{x:,.2f}"
        )
    )

    st.dataframe(
        display_countries,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # UK REVENUE CONTRIBUTION
    # ==========================================

    uk_revenue = countries.loc[
        countries["Country"] == "United Kingdom",
        "Revenue"
    ].sum()

    if total_revenue > 0:

        st.metric(
            "UK Revenue Contribution",
            f"{uk_revenue / total_revenue * 100:.1f}%"
        )


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
