
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

st.title("📊 RetailIQ AI")
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
    "Upload a dataset containing InvoiceNo, "
    "StockCode, Description, Quantity, "
    "InvoiceDate, UnitPrice and Country."
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


required_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country"
]

missing_columns = [
    col for col in required_columns
    if col not in raw_df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {missing_columns}"
    )
    st.stop()


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

    st.subheader("Monthly Revenue Trend")

    fig = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Performing Products")

    st.dataframe(
        products.head(10),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 2: SALES ANALYTICS
# ==========================================

elif page == "Sales Analytics":

    st.header("Sales Analytics")

    fig = px.bar(
        monthly,
        x="Month",
        y="Revenue",
        title="Monthly Revenue",
        template="plotly_white"
    )

    fig.update_layout(
        yaxis_title=f"Revenue ({selected_currency})",
        yaxis_tickprefix=currency_symbol
    )
    
    fig.update_traces(
        hovertemplate=(
            f"Month: %{{x}}<br>"
            f"Revenue: {currency_symbol}%{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        monthly,
        x="Month",
        y="Orders",
        markers=True,
        title="Monthly Orders",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Performance Data")

    st.dataframe(
        monthly,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 3: PRODUCT INTELLIGENCE
# ==========================================

elif page == "Product Intelligence":

    st.header("Product Intelligence")

    st.subheader("Top 10 Products by Revenue")

    fig = px.bar(
        products.head(10),
        x="Revenue",
        y="Description",
        orientation="h",
        template="plotly_white"
    )


    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title=f"Revenue ({selected_currency})",
        xaxis_tickprefix=currency_symbol
    )

    fig.update_traces(
        hovertemplate=(
            f"Product: %{{y}}<br>"
            f"Revenue: {currency_symbol}%{{x:,.2f}}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Products by Quantity")


    quantity_top = products.nlargest(
        10, "Units"
    )

    fig2 = px.bar(
        quantity_top,
        x="Units",
        y="Description",
        orientation="h",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Complete Product Performance")

    st.dataframe(
        products,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 4: MARKET ANALYSIS
# ==========================================

elif page == "Market Analysis":

    st.header("Country-wise Market Analysis")

    fig = px.bar(
        countries.head(10),
        x="Country",
        y="Revenue",
        title="Top 10 Countries by Revenue",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        countries,
        use_container_width=True,
        hide_index=True
    )

    if total_revenue:
        uk_revenue = countries.loc[
            countries["Country"] == "United Kingdom",
            "Revenue"
        ].sum()

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
        f"with £{best_month['Revenue']:,.2f}."
    )

    st.info(
        f"Your leading market is {best_country['Country']} "
        f"with £{best_country['Revenue']:,.2f} revenue."
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
                f"£{best_country['Revenue']:,.2f} revenue."
            )

        elif "product" in q:

            answer = (
                f"{best_product['Description']} is the "
                f"highest-revenue product with "
                f"£{best_product['Revenue']:,.2f} revenue."
            )

        elif "month" in q:

            answer = (
                f"{best_month['Month'].strftime('%B %Y')} "
                f"was the best month, generating "
                f"£{best_month['Revenue']:,.2f}."
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
                f"Total revenue is £{total_revenue:,.2f}."
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
