import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="EV Penetration India Dashboard",
    layout="wide",
    page_icon="EV",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FORECAST_END_YEAR = 2030
NEON_COLORS = [
    "#00F5FF",
    "#FF2ED1",
    "#39FF14",
    "#FFF700",
    "#FF5F1F",
    "#9D4EDD",
    "#00FFA3",
    "#FF006E",
]

NEON_SCALE = [
    [0.0, "#101020"],
    [0.2, "#00F5FF"],
    [0.4, "#39FF14"],
    [0.6, "#FFF700"],
    [0.8, "#FF2ED1"],
    [1.0, "#9D4EDD"],
]

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = NEON_COLORS
px.defaults.color_continuous_scale = NEON_SCALE

def neon_layout(fig):
    fig.update_layout(
        plot_bgcolor="#070A18",
        paper_bgcolor="#070A18",
        font=dict(color="#F8FAFC"),
        title_font=dict(color="#F8FAFC", size=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# -----------------------------
# Common helper functions
# -----------------------------

def make_unique_columns(columns):
    seen = {}
    unique_cols = []

    for col in columns:
        col = str(col).strip()

        if col not in seen:
            seen[col] = 0
            unique_cols.append(col)
        else:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")

    return unique_cols


def clean_column_names(df):
    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("%", "Percent", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("-", "_", regex=False)
    )

    df.columns = make_unique_columns(df.columns)
    return df


def clean_num(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("NA", "", regex=False)
        .str.replace("na", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def find_col(df, keywords):
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords):
            return col
    return None


def normalize_state_name(value):
    text = str(value).strip().replace("&", "and")
    text = " ".join(text.split()).title()

    fixes = {
        "Andaman And Nicobar Island": "Andaman And Nicobar Islands",
        "Ut Of Dnh And Dd": "Dadra And Nagar Haveli And Daman And Diu",
        "Nct Of Delhi": "Delhi",
        "Jammu And Kashmir": "Jammu And Kashmir",
    }

    return fixes.get(text, text)


def normalize_fuel(value):
    return str(value).strip().lower().replace(" ", "")


def is_ev_fuel(value):
    fuel = normalize_fuel(value)

    if "hybrid" in fuel:
        return False

    return fuel in [
        "ev",
        "electric",
        "electric(bov)",
        "electricbov",
    ]


def remove_total_rows(df):
    df = df.copy()
    mask = pd.Series(False, index=df.index)

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            vals = (
                df[col]
                .astype(str)
                .str.replace("\u00a0", " ", regex=False)
                .str.replace("\u200b", "", regex=False)
                .str.lower()
                .str.strip()
            )
            mask = mask | vals.isin(["total", "grand total"])

    return df.loc[~mask].reset_index(drop=True)


def safe_pct_change(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return ((current - previous) / previous) * 100


def cagr(start, end, years):
    if pd.isna(start) or pd.isna(end) or start <= 0 or years <= 0:
        return np.nan
    return ((end / start) ** (1 / years) - 1) * 100


def fmt_num(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def fmt_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


# -----------------------------
# Dataset preparation
# -----------------------------

def normalize_category_name(col):
    text = str(col).replace("_", " ").strip().lower()
    text = text.replace("electric", "").strip()

    if "2w" in text or "2 w" in text or "two" in text:
        return "Electric 2W"
    if "3w" in text or "3 w" in text or "three" in text:
        return "Electric 3W"
    if "4w" in text or "4 w" in text or "car" in text:
        return "Electric Cars"
    if "goods" in text:
        return "Electric Goods"
    if "bus" in text:
        return "Electric Bus"

    return str(col).replace("_", " ").strip().title()


def normalize_wheel_category(value):
    text = str(value).strip()
    low = text.lower()

    if "all vehicle" in low:
        return "All Vehicles"
    if low == "2w" or "two wheeler" in low:
        return "2W"
    if "3w passenger" in low:
        return "3W Passenger"
    if "3w goods" in low:
        return "3W Goods"
    if "car" in low or "cab" in low:
        return "Cars"
    if "bus" in low:
        return "Buses"
    if "lgv" in low or "mgv" in low or "hgv" in low or "tonnes" in low:
        return "Goods Vehicles"
    if "other" in low:
        return "Others"

    return text.title()


def prepare_wheel_fuel_yearly(path):
    raw = pd.read_excel(path, sheet_name="FY2011-2025", engine="openpyxl")

    year_cols = [col for col in raw.columns if str(col).startswith("FY")]

    national = raw.iloc[0:4][["Unnamed: 0", "Unnamed: 1"] + year_cols].copy()
    national.columns = ["Category", "Fuel_Type"] + year_cols
    national["Category"] = "All Vehicles"

    category = raw.iloc[7:].copy()
    category = category[["Unnamed: 0", "Unnamed: 1"] + year_cols]
    category.columns = ["Category", "Fuel_Type"] + year_cols
    category = category.dropna(subset=["Category", "Fuel_Type"])

    yearly = pd.concat([national, category], ignore_index=True)
    yearly["Category"] = yearly["Category"].apply(normalize_wheel_category)

    yearly_long = yearly.melt(
        id_vars=["Category", "Fuel_Type"],
        value_vars=year_cols,
        var_name="Financial_Year",
        value_name="Registrations",
    )

    yearly_long["Year"] = yearly_long["Financial_Year"].str.extract(r"(\d{4})").astype(int)
    yearly_long["Registrations"] = clean_num(yearly_long["Registrations"]).fillna(0)
    yearly_long["Is_EV"] = yearly_long["Fuel_Type"].apply(is_ev_fuel)

    return yearly_long


def prepare_penetration(yearly_long):
    total_regs = (
        yearly_long[yearly_long["Fuel_Type"].astype(str).str.lower().str.strip() == "total"]
        .groupby(["Category", "Year"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "Total_Registrations"})
    )

    ev_regs = (
        yearly_long[yearly_long["Is_EV"]]
        .groupby(["Category", "Year"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "EV_Registrations"})
    )

    penetration = total_regs.merge(ev_regs, on=["Category", "Year"], how="left")
    penetration["EV_Registrations"] = penetration["EV_Registrations"].fillna(0)

    penetration["EV_Penetration_Percent"] = (
        penetration["EV_Registrations"]
        / penetration["Total_Registrations"].replace(0, np.nan)
    ) * 100

    return penetration


def prepare_monthly_category(path):
    df = pd.read_csv(path)
    df = remove_total_rows(clean_column_names(df))

    year_col = find_col(df, ["Year"])
    month_col = find_col(df, ["Month"])
    date_col = find_col(df, ["Date"])
    total_col = find_col(df, ["Total_Registration", "Total Registration", "Total"])

    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif year_col and month_col:
        years = df[year_col].astype(str).str.extract(r"(\d{4})")[0]
        months = df[month_col].astype(str).str.strip()
        df["Date"] = pd.to_datetime(years + "-" + months + "-01", errors="coerce")
    else:
        raise ValueError("Monthly category file must contain Year + Month or Date columns.")

    category_cols = []

    for col in df.columns:
        low = str(col).lower()

        if col in ["Date", year_col, month_col, date_col]:
            continue
        if "total" in low:
            continue
        if "electric" in low:
            category_cols.append(col)

    if not category_cols:
        raise ValueError("Could not detect columns like Electric 2W, Electric 3W, Electric 4W.")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in category_cols:
        df[col] = clean_num(df[col]).fillna(0).clip(lower=0)

    category_monthly = df.melt(
        id_vars=["Date"],
        value_vars=category_cols,
        var_name="Category",
        value_name="EV_Registrations",
    )

    category_monthly["Category"] = category_monthly["Category"].apply(normalize_category_name)

    category_monthly = (
        category_monthly
        .groupby(["Date", "Category"], as_index=False)["EV_Registrations"]
        .sum()
    )

    if total_col and total_col in df.columns:
        df[total_col] = clean_num(df[total_col]).fillna(0).clip(lower=0)
        monthly_total = (
            df.groupby("Date", as_index=False)[total_col]
            .sum()
            .rename(columns={total_col: "EV_Registrations"})
        )
    else:
        monthly_total = (
            category_monthly
            .groupby("Date", as_index=False)["EV_Registrations"]
            .sum()
        )

    monthly_total["Year"] = monthly_total["Date"].dt.year
    monthly_total["Month"] = monthly_total["Date"].dt.month
    monthly_total["Month_Name"] = monthly_total["Date"].dt.strftime("%b")

    category_monthly["Year"] = category_monthly["Date"].dt.year
    category_monthly["Month"] = category_monthly["Date"].dt.month
    category_monthly["Month_Name"] = category_monthly["Date"].dt.strftime("%b")

    return monthly_total, category_monthly


def prepare_state_monthly(path):
    df = pd.read_csv(path)
    df = clean_column_names(df)

    date_col = find_col(df, ["date"])
    state_col = find_col(df, ["state_name", "state"])
    fuel_col = find_col(df, ["fuel_type", "fuel"])
    reg_col = find_col(df, ["registrations", "registration"])

    if not all([date_col, state_col, fuel_col, reg_col]):
        raise ValueError("State monthly dataset must contain date, state, fuel_type, and registrations columns.")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["State"] = df[state_col].apply(normalize_state_name)
    df["Fuel_Type"] = df[fuel_col].astype(str).str.strip()
    df["Registrations"] = clean_num(df[reg_col]).fillna(0).clip(lower=0)
    df["Is_EV"] = df["Fuel_Type"].apply(is_ev_fuel)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    df = df.dropna(subset=["Date"])

    state_ev_monthly = (
        df[df["Is_EV"]]
        .groupby(["State", "Date", "Year", "Month"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "EV_Registrations"})
    )

    return df, state_ev_monthly


# -----------------------------
# Forecasting
# -----------------------------

def forecast_monthly(actual_df, value_col="EV_Registrations", degree=2):
    actual = actual_df[["Date", value_col]].copy().sort_values("Date")
    actual[value_col] = clean_num(actual[value_col]).fillna(0).clip(lower=0)

    actual = (
        actual
        .groupby("Date", as_index=False)[value_col]
        .sum()
        .sort_values("Date")
    )

    actual = actual[actual["Date"].notna()]

    if actual.empty:
        raise ValueError("No monthly records are available for forecasting.")

    full_dates = pd.date_range(actual["Date"].min(), actual["Date"].max(), freq="MS")
    actual = pd.DataFrame({"Date": full_dates}).merge(actual, on="Date", how="left")
    actual[value_col] = actual[value_col].fillna(0)

    last_date = actual["Date"].max()

    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        f"{FORECAST_END_YEAR}-12-01",
        freq="MS",
    )

    all_dates = pd.concat([actual["Date"], pd.Series(future_dates)], ignore_index=True)

    nonzero_months = int((actual[value_col] > 0).sum())

    if len(actual) >= 6 and nonzero_months >= 4:
        x_actual = np.arange(len(actual)).reshape(-1, 1)
        x_all = np.arange(len(all_dates)).reshape(-1, 1)

        model = make_pipeline(
            PolynomialFeatures(degree),
            Ridge(alpha=1.0),
        )

        model.fit(x_actual, actual[value_col])

        fitted = model.predict(x_actual)
        predicted = np.clip(model.predict(x_all), 0, None)

        r2 = r2_score(actual[value_col], fitted) if len(actual) > 1 else np.nan
        residual_std = float(np.std(actual[value_col] - fitted))
        method = "Polynomial Ridge Regression"
    else:
        base = actual[value_col].tail(min(6, len(actual))).mean()
        future_values = [base] * len(future_dates)
        predicted = np.array(list(actual[value_col]) + future_values)

        r2 = np.nan
        residual_std = float(np.std(actual[value_col].tail(min(6, len(actual)))))
        method = "Fallback Average"

    ci = 1.96 * residual_std

    forecast = pd.DataFrame({
        "Date": all_dates,
        "Actual": list(actual[value_col]) + [np.nan] * len(future_dates),
        "Predicted": predicted,
        "CI_Lower": np.clip(predicted - ci, 0, None),
        "CI_Upper": predicted + ci,
    })

    forecast["Year"] = forecast["Date"].dt.year
    forecast["Month"] = forecast["Date"].dt.month

    return forecast, r2, last_date, method


def forecast_categories(category_monthly):
    frames = []
    metrics = []

    for category, group in category_monthly.groupby("Category"):
        pred, r2, last_date, method = forecast_monthly(group)

        pred["Category"] = category
        frames.append(pred)

        metrics.append({
            "Category": category,
            "Model_Fit_Score": r2,
            "Last_Actual_Month": last_date.strftime("%b %Y"),
            "Forecast_Method": method,
        })

    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)


def forecast_states(state_ev_monthly):
    frames = []
    metrics = []

    for state, group in state_ev_monthly.groupby("State"):
        pred, r2, last_date, method = forecast_monthly(group)

        nonzero_months = int((group["EV_Registrations"] > 0).sum())

        if nonzero_months >= 36 and pd.notna(r2) and r2 >= 0.50:
            confidence = "High"
        elif nonzero_months >= 18:
            confidence = "Medium"
        else:
            confidence = "Low"

        pred["State"] = state
        frames.append(pred)

        metrics.append({
            "State": state,
            "Model_Fit_Score": r2,
            "Last_Actual_Month": last_date.strftime("%b %Y"),
            "Nonzero_EV_Months": nonzero_months,
            "Forecast_Method": method,
            "Forecast_Confidence": confidence,
        })

    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)


# -----------------------------
# Load data
# -----------------------------

@st.cache_data
def load_data():
    yearly_long = prepare_wheel_fuel_yearly(DATA_DIR / "Dataset_Wheel_Fuel_Yearly_2011-2025.xlsx")
    penetration = prepare_penetration(yearly_long)

    monthly_total, category_monthly = prepare_monthly_category(DATA_DIR / "Ev_monthlywith_category.csv")

    state_raw, state_ev_monthly = prepare_state_monthly(
        DATA_DIR / "Dataset_state_Fuel_Month_2019-2024.csv"
    )

    national_forecast, national_r2, last_actual_month, national_method = forecast_monthly(monthly_total)
    category_forecast, category_metrics = forecast_categories(category_monthly)
    state_forecast, state_metrics = forecast_states(state_ev_monthly)

    return {
        "yearly_long": yearly_long,
        "penetration": penetration,
        "monthly_total": monthly_total,
        "category_monthly": category_monthly,
        "state_raw": state_raw,
        "state_ev_monthly": state_ev_monthly,
        "national_forecast": national_forecast,
        "national_r2": national_r2,
        "last_actual_month": last_actual_month,
        "national_method": national_method,
        "category_forecast": category_forecast,
        "category_metrics": category_metrics,
        "state_forecast": state_forecast,
        "state_metrics": state_metrics,
    }


try:
    data = load_data()

    yearly_long = data["yearly_long"]
    penetration = data["penetration"]
    monthly_total = data["monthly_total"]
    category_monthly = data["category_monthly"]
    state_raw = data["state_raw"]
    state_ev_monthly = data["state_ev_monthly"]
    national_forecast = data["national_forecast"]
    category_forecast = data["category_forecast"]
    category_metrics = data["category_metrics"]
    state_forecast = data["state_forecast"]
    state_metrics = data["state_metrics"]

except Exception as e:
    st.error("Failed to load or prepare dashboard datasets.")
    st.exception(e)
    st.stop()


# -----------------------------
# Dashboard
# -----------------------------

st.title("Electric Vehicle Penetration - India")
st.caption("Client-ready analytics and forecast dashboard using EV registration datasets")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "State Analysis",
    "Penetration",
    "Class / Category",
    "Monthly Trends",
    "Forecast to 2030",
    "State Forecast",
])


# -----------------------------
# Tab 1: Overview
# -----------------------------

with tab1:
    st.subheader("National EV Growth Overview")

    national_ev = (
        yearly_long[
            (yearly_long["Category"] == "All Vehicles")
            & (yearly_long["Is_EV"])
            & (yearly_long["Year"].between(2015, 2025))
        ]
        .groupby("Year", as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "EV_Registrations"})
    )

    start_year = 2015
    latest_year = 2025

    first_total = national_ev.loc[national_ev["Year"] == start_year, "EV_Registrations"].sum()
    latest_total = national_ev.loc[national_ev["Year"] == latest_year, "EV_Registrations"].sum()

    growth = ((latest_total - first_total) / first_total * 100) if first_total > 0 else np.nan
    growth_cagr = cagr(first_total, latest_total, latest_year - start_year)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(f"Latest EV Registrations ({latest_year})", fmt_num(latest_total))
    c2.metric(f"Starting EV Registrations ({start_year})", fmt_num(first_total))
    c3.metric("Overall Growth", fmt_pct(growth))
    c4.metric("CAGR", fmt_pct(growth_cagr))

    st.info(
        "Formula: Overall Growth = ((Latest Year EVs - First Year EVs) / First Year EVs) x 100. "
        "CAGR = ((Latest Year EVs / First Year EVs) ^ (1 / number of years) - 1) x 100."
    )

    fig = px.area(
        national_ev,
        x="Year",
        y="EV_Registrations",
        markers=True,
        title="Annual EV Registrations in India (2015 to 2025)",
    )

    fig.update_layout(showlegend=False, yaxis_title="EV Registrations")
    fig = neon_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.write("CAGR details")

    cagr_summary = pd.DataFrame([{
        "Start_Year": start_year,
        "End_Year": latest_year,
        "Start_EV_Registrations": first_total,
        "End_EV_Registrations": latest_total,
        "Overall_Growth_Percent": growth,
        "CAGR_Percent": growth_cagr,
    }])

    st.dataframe(cagr_summary, use_container_width=True)


# -----------------------------
# Tab 2: State Analysis
# -----------------------------

with tab2:
    st.subheader("State-wise EV Registration Analysis")

    state_annual = (
        state_ev_monthly
        .groupby(["State", "Year"], as_index=False)["EV_Registrations"]
        .sum()
    )

    complete_state_years = sorted(state_annual["Year"].dropna().unique())
    default_state_year = 2023 if 2023 in complete_state_years else complete_state_years[-1]

    selected_state_year = st.selectbox(
        "Select year for state ranking",
        complete_state_years,
        index=complete_state_years.index(default_state_year),
    )

    st.info(
        "This view uses only Electric(Bov) records from the new state fuel monthly dataset. "
        "Other fuel categories are excluded for EV state analysis."
    )

    top_states = (
        state_annual[state_annual["Year"] == selected_state_year]
        .sort_values("EV_Registrations", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_states,
        x="State",
        y="EV_Registrations",
        title=f"Top 10 States/UTs by EV Registrations ({selected_state_year})",
        color="EV_Registrations",
    )

    fig.update_layout(xaxis_tickangle=-45)
    fig = neon_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    state_choice = st.selectbox(
        "Select a state for historical trend",
        sorted(state_annual["State"].unique()),
    )

    state_view = state_annual[state_annual["State"] == state_choice]

    fig2 = px.line(
        state_view,
        x="Year",
        y="EV_Registrations",
        markers=True,
        title=f"Historical EV Trend - {state_choice}",
    )

    fig = neon_layout(fig2)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("State EV dataset checks"):
        st.write("Shape:", state_ev_monthly.shape)
        st.dataframe(state_ev_monthly.head(), use_container_width=True)
        st.dataframe(state_ev_monthly.tail(), use_container_width=True)


# -----------------------------
# Tab 3: Penetration
# -----------------------------

with tab3:
    st.subheader("EV Penetration Percentage")

    st.info("Formula: EV Penetration % = (EV registrations / total vehicle registrations) x 100.")

    national_pen = (
        penetration[
            (penetration["Category"] == "All Vehicles")
            & (penetration["Year"].between(2015, 2025))
        ]
        .sort_values("Year")
    )

    latest_pen_year = int(national_pen["Year"].max())
    latest_pen = national_pen[national_pen["Year"] == latest_pen_year].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(f"EV Penetration ({latest_pen_year})", fmt_pct(latest_pen["EV_Penetration_Percent"]))
    c2.metric(f"EV Registrations ({latest_pen_year})", fmt_num(latest_pen["EV_Registrations"]))
    c3.metric(f"Total Registrations ({latest_pen_year})", fmt_num(latest_pen["Total_Registrations"]))

    fig = px.line(
        national_pen,
        x="Year",
        y="EV_Penetration_Percent",
        markers=True,
        title="Year-wise EV Penetration in India (2015 to 2025)",
    )

    fig.update_layout(yaxis_title="EV Penetration %")
    fig = neon_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    selected_pen_year = st.selectbox(
        "Select year for category-wise penetration",
        sorted(penetration["Year"].unique()),
        index=len(sorted(penetration["Year"].unique())) - 1,
    )

    category_pen = (
        penetration[
            (penetration["Category"] != "All Vehicles")
            & (penetration["Year"] == selected_pen_year)
        ]
        .sort_values("EV_Penetration_Percent", ascending=False)
    )

    fig2 = px.bar(
        category_pen,
        x="Category",
        y="EV_Penetration_Percent",
        title=f"Category-wise EV Penetration ({selected_pen_year})",
        color="EV_Penetration_Percent",
    )

    fig2 = neon_layout(fig2)
    fig2.update_layout(xaxis_tickangle=-45, yaxis_title="EV Penetration %")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("Category-wise penetration table")
    st.dataframe(category_pen, use_container_width=True)


# -----------------------------
# Tab 4: Class / Category
# -----------------------------

with tab4:
    st.subheader("Vehicle Class / Category-wise EV Analysis")

    category_yearly = (
        yearly_long[
            (yearly_long["Is_EV"])
            & (yearly_long["Category"] != "All Vehicles")
            & (yearly_long["Year"].between(2015, 2025))
        ]
        .groupby(["Category", "Year"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "EV_Registrations"})
    )

    category_summary = (
        category_yearly
        .groupby("Category", as_index=False)["EV_Registrations"]
        .sum()
        .sort_values("EV_Registrations", ascending=False)
    )

    fig = px.bar(
        category_summary,
        x="Category",
        y="EV_Registrations",
        title="Total EV Registrations by Class / Category (2015 to 2025)",
        color="Category",
    )

    fig = neon_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        category_yearly,
        x="Year",
        y="EV_Registrations",
        color="Category",
        markers=True,
        title="Year-wise EV Trend by Class / Category",
    )
    fig2 = neon_layout(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    start = category_yearly[category_yearly["Year"] == 2015][["Category", "EV_Registrations"]]
    end = category_yearly[category_yearly["Year"] == 2025][["Category", "EV_Registrations"]]

    category_cagr = start.merge(end, on="Category", suffixes=("_2015", "_2025"))
    category_cagr["CAGR_Percent"] = category_cagr.apply(
        lambda row: cagr(row["EV_Registrations_2015"], row["EV_Registrations_2025"], 10),
        axis=1,
    )

    category_cagr = category_cagr.sort_values("CAGR_Percent", ascending=False)

    st.write("Category-wise CAGR")
    st.dataframe(category_cagr, use_container_width=True)

    st.write("Category summary")
    st.dataframe(category_summary, use_container_width=True)


# -----------------------------
# Tab 5: Monthly Trends
# -----------------------------

with tab5:
    st.subheader("Monthly EV Registration Trends")

    monthly_total = monthly_total.sort_values("Date").copy()

    monthly_total["YoY_Growth_Pct"] = (
        monthly_total["EV_Registrations"].pct_change(periods=12) * 100
    )

    st.info(
        "Formula: Monthly YoY Growth % = ((Current month EVs - Same month previous year EVs) / "
        "Same month previous year EVs) x 100."
    )

    fig1 = px.line(
        monthly_total,
        x="Date",
        y="EV_Registrations",
        title="Monthly Total EV Registrations",
    )
    fig1 = neon_layout(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        monthly_total.dropna(subset=["YoY_Growth_Pct"]),
        x="Date",
        y="YoY_Growth_Pct",
        title="Year-over-Year Monthly EV Growth",
        color="YoY_Growth_Pct",
        color_continuous_midpoint=0,
    )
    fig2 = neon_layout(fig2)

    st.plotly_chart(fig2, use_container_width=True)

    pivot = (
        monthly_total.pivot_table(
            index="Year",
            columns="Month",
            values="EV_Registrations",
            aggfunc="sum",
        )
        .reindex(columns=range(1, 13))
    )

    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig3 = px.imshow(
        pivot,
        aspect="auto",
        title="Seasonality Heatmap - Monthly EV Registrations",
        color_continuous_scale="Greens",
    )
    fig3 = neon_layout(fig3)
    st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# Tab 6: Forecast to 2030
# -----------------------------

with tab6:
    st.subheader("Monthly and Category Forecast to 2030")

    latest_actual_year = int(monthly_total["Year"].max())

    st.info(
        f"Forecast period: actual monthly data is used through "
        f"{data['last_actual_month'].strftime('%B %Y')}. "
        "Predictions continue through December 2030. "
        f"Model used: {data['national_method']}."
    )

    st.caption(
        "Model Fit Score shows how closely the model follows historical data. "
        "A value closer to 1 means stronger historical fit. It does not guarantee future accuracy."
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=national_forecast["Date"],
        y=national_forecast["Actual"],
        name="Actual",
        mode="lines+markers",
    ))

    fig.add_trace(go.Scatter(
        x=national_forecast["Date"],
        y=national_forecast["Predicted"],
        name="Predicted",
        mode="lines",
        line=dict(dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=national_forecast["Date"],
        y=national_forecast["CI_Upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=national_forecast["Date"],
        y=national_forecast["CI_Lower"],
        name="Indicative range",
        mode="lines",
        fill="tonexty",
        line=dict(width=0),
    ))

    fig.update_layout(
        title="National EV Monthly Forecast to 2030",
        yaxis_title="EV Registrations",
    )
    fig = neon_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    actual_annual = monthly_total.groupby("Year", as_index=False)["EV_Registrations"].sum()

    predicted_annual = (
        national_forecast.groupby("Year", as_index=False)["Predicted"]
        .sum()
        .rename(columns={"Predicted": "Predicted_EV_Registrations"})
    )

    annual_forecast = predicted_annual.merge(
        actual_annual.rename(columns={"EV_Registrations": "Actual_EV_Registrations"}),
        on="Year",
        how="left",
    )

    annual_forecast["Data_Type"] = np.where(
        annual_forecast["Year"] <= latest_actual_year,
        "Actual",
        "Forecast",
    )

    annual_forecast["Forecast_YoY_Growth_Percent"] = (
        annual_forecast["Predicted_EV_Registrations"].pct_change() * 100
    )

    st.write("Annual national forecast")
    st.dataframe(annual_forecast.tail(10), use_container_width=True)

    category_yearly_forecast = (
        category_forecast.groupby(["Year", "Category"], as_index=False)["Predicted"]
        .sum()
        .rename(columns={"Predicted": "Predicted_EV_Registrations"})
    )

    forecast_year_options = list(range(latest_actual_year + 1, FORECAST_END_YEAR + 1))

    selected_category_year = st.selectbox(
        "Select category forecast year",
        forecast_year_options,
        index=len(forecast_year_options) - 1,
    )

    selected_category_table = (
        category_yearly_forecast[
            category_yearly_forecast["Year"] == selected_category_year
        ]
        .sort_values("Predicted_EV_Registrations", ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "National Model Fit Score",
            f"{data['national_r2']:.3f}" if pd.notna(data["national_r2"]) else "N/A",
        )

    with c2:
        top_category = selected_category_table.iloc[0]["Category"]
        top_category_value = selected_category_table.iloc[0]["Predicted_EV_Registrations"]

        st.metric(
            f"Top Category ({selected_category_year})",
            top_category,
            f"{top_category_value:,.0f} predicted EVs",
        )

    st.write(f"Category-wise prediction table ({selected_category_year})")
    st.dataframe(selected_category_table, use_container_width=True)

    selected_categories = st.multiselect(
        "Select categories to display",
        sorted(category_forecast["Category"].unique()),
        default=sorted(category_forecast["Category"].unique()),
    )

    fig2 = px.line(
        category_forecast[category_forecast["Category"].isin(selected_categories)],
        x="Date",
        y="Predicted",
        color="Category",
        title="Separate Category-wise Monthly Forecast to 2030",
    )
    fig2 = neon_layout(fig2)

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Category model diagnostics"):
        st.dataframe(
            category_metrics.sort_values("Model_Fit_Score", ascending=False),
            use_container_width=True,
        )


# -----------------------------
# Tab 7: State Forecast
# -----------------------------

with tab7:
    st.subheader("State-wise EV Forecast to 2030")

    state_actual_latest_month = state_ev_monthly["Date"].max()
    state_actual_latest_year = int(state_ev_monthly["Year"].max())

    st.info(
        f"This forecast uses only Electric(Bov) records from Dataset_state_Fuel_Month_2019-2024.csv. "
        f"Actual state monthly data is available through {state_actual_latest_month.strftime('%B %Y')}. "
        "Every state is included. States with limited history are marked with lower forecast confidence."
    )

    state_forecast_annual = (
        state_forecast.groupby(["State", "Year"], as_index=False)["Predicted"]
        .sum()
        .rename(columns={"Predicted": "Predicted_EV_Registrations"})
    )

    selected_state_forecast_year = st.selectbox(
        "Select state forecast year",
        list(range(2025, FORECAST_END_YEAR + 1)),
        index=len(list(range(2025, FORECAST_END_YEAR + 1))) - 1,
    )

    latest_actual_state = (
        state_ev_monthly[state_ev_monthly["Year"] == 2023]
        .groupby("State", as_index=False)["EV_Registrations"]
        .sum()
        .rename(columns={"EV_Registrations": "Actual_EV_Registrations_2023"})
    )

    year_view = (
        state_forecast_annual[
            state_forecast_annual["Year"] == selected_state_forecast_year
        ]
        .merge(latest_actual_state, on="State", how="left")
        .merge(state_metrics, on="State", how="left")
    )

    year_view["Growth_vs_2023_Percent"] = year_view.apply(
        lambda row: safe_pct_change(
            row["Predicted_EV_Registrations"],
            row["Actual_EV_Registrations_2023"],
        ),
        axis=1,
    )

    year_view = year_view.sort_values("Predicted_EV_Registrations", ascending=False)

    c1, c2, c3 = st.columns(3)

    c1.metric("States Forecasted", year_view["State"].nunique())
    c2.metric("Latest Actual Month", state_actual_latest_month.strftime("%b %Y"))
    c3.metric(
        "High Confidence States",
        int((year_view["Forecast_Confidence"] == "High").sum()),
    )

    fig = px.bar(
        year_view.head(12),
        x="State",
        y="Predicted_EV_Registrations",
        title=f"Top Predicted States by EV Registrations ({selected_state_forecast_year})",
        color="Predicted_EV_Registrations",
    )

    fig = neon_layout(fig)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    top_state_name = year_view.iloc[0]["State"]
    top_state_value = year_view.iloc[0]["Predicted_EV_Registrations"]

    st.success(
        f"For {selected_state_forecast_year}, {top_state_name} is projected to have the highest "
        f"EV registrations with approximately {top_state_value:,.0f} registrations."
    )

    st.write(f"State-wise prediction table ({selected_state_forecast_year})")
    st.dataframe(year_view, use_container_width=True)

    state_choice = st.selectbox(
        "Select a state for forecast trend",
        sorted(state_forecast["State"].unique()),
    )

    actual_state_trend = state_ev_monthly[state_ev_monthly["State"] == state_choice].copy()
    actual_state_trend = actual_state_trend.rename(
        columns={"EV_Registrations": "EV_Registrations"}
    )
    actual_state_trend["Data_Type"] = "Actual"

    forecast_state_trend = state_forecast[state_forecast["State"] == state_choice].copy()
    forecast_state_trend = forecast_state_trend.rename(
        columns={"Predicted": "EV_Registrations"}
    )
    forecast_state_trend["Data_Type"] = np.where(
        forecast_state_trend["Date"] <= state_actual_latest_month,
        "Fitted",
        "Forecast",
    )

    combined_state_trend = pd.concat(
        [
            actual_state_trend[["State", "Date", "EV_Registrations", "Data_Type"]],
            forecast_state_trend[["State", "Date", "EV_Registrations", "Data_Type"]],
        ],
        ignore_index=True,
    )

    fig2 = px.line(
        combined_state_trend,
        x="Date",
        y="EV_Registrations",
        color="Data_Type",
        title=f"Actual and Forecast EV Trend - {state_choice}",
    )

    fig2 = neon_layout(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("State forecast model diagnostics"):
        st.dataframe(
            state_metrics.sort_values(
                ["Forecast_Confidence", "Model_Fit_Score"],
                ascending=[True, False],
            ),
            use_container_width=True,
        )