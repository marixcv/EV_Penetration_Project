
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


def read_statewise_file(path):
    raw = pd.read_csv(path, header=None)

    header_row_idx = None

    for i in range(min(8, len(raw))):
        row_values = raw.iloc[i].fillna("").astype(str).str.lower().tolist()

        if any("state" in val for val in row_values):
            header_row_idx = i
            break

    if header_row_idx is None:
        df = pd.read_csv(path)
        df.columns = make_unique_columns(df.columns)
        return df

    fuel_row_idx = max(header_row_idx - 1, 0)

    fuel_headers = raw.iloc[fuel_row_idx].ffill().fillna("").astype(str).str.strip()
    year_headers = raw.iloc[header_row_idx].fillna("").astype(str).str.strip()

    final_cols = []

    for fuel, year in zip(fuel_headers, year_headers):
        fuel = str(fuel).strip()
        year = str(year).strip()

        if fuel.lower() in ["nan", "none", ""]:
            fuel = ""
        if year.lower() in ["nan", "none", ""]:
            year = ""

        if "state" in year.lower():
            final_cols.append("State")
        elif fuel and year:
            final_cols.append(f"{fuel}_{year}")
        elif year:
            final_cols.append(year)
        elif fuel:
            final_cols.append(fuel)
        else:
            final_cols.append("Blank")

    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = make_unique_columns(final_cols)
    df = df.dropna(how="all").reset_index(drop=True)

    return df


@st.cache_data
def load_data():
    return {
        "monthly_category": pd.read_csv(DATA_DIR / "Ev_monthlywith_category.csv"),
        "statewise": read_statewise_file(DATA_DIR / "IMP_Ev_statewise.csv"),
        "penetration": pd.read_csv(DATA_DIR / "Ev_penetration%wise.csv"),
    }


def clean_column_names(df):
    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("%", "Percent")
        .str.replace(".", "", regex=False)
    )

    df.columns = make_unique_columns(df.columns)
    return df


def clean_num(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("NA", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def find_col(df, keywords):
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None


def remove_total_rows(df):
    df = df.copy()
    mask = pd.Series(False, index=df.index)

    for i in range(len(df.columns)):
        series = df.iloc[:, i]

        if series.dtype == "object" or str(series.dtype).startswith("string"):
            vals = (
                series.astype(str)
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


def show_dataset_checks(df, name):
    with st.expander(f"{name} - dataset checks"):
        st.write("Shape:", df.shape)

        c1, c2 = st.columns(2)

        with c1:
            st.write("First records")
            st.dataframe(df.head(), use_container_width=True)

        with c2:
            st.write("Latest records")
            st.dataframe(df.tail(), use_container_width=True)

        st.write("Missing values")
        st.dataframe(
            df.isna()
            .sum()
            .reset_index()
            .rename(columns={"index": "Column", 0: "Missing"}),
            use_container_width=True,
        )


def normalize_category_name(col):
    text = str(col).replace("_", " ").strip().lower()
    text = text.replace("electric", "").strip()

    if "2w" in text or "2 w" in text or "two" in text:
        return "Electric 2W"
    if "3w" in text or "3 w" in text or "three" in text:
        return "Electric 3W"
    if "4w" in text or "4 w" in text or "four" in text:
        return "Electric 4W"
    if "goods" in text:
        return "Electric Goods"
    if "bus" in text:
        return "Electric Bus"

    return str(col).replace("_", " ").strip().title()


def prepare_monthly_category(df):
    df = remove_total_rows(clean_column_names(df))

    year_col = find_col(df, ["Year"])
    month_col = find_col(df, ["Month"])
    date_col = find_col(df, ["Date"])
    total_col = find_col(df, ["Total_Registration", "Total Registration", "Total"])

    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif year_col and month_col:
        years = df[year_col].astype(str).str.split(".").str[0].str.strip()
        months = df[month_col].astype(str).str.strip()
        df["Date"] = pd.to_datetime(years + "-" + months + "-01", errors="coerce")
    else:
        raise ValueError("Monthly file must contain Year + Month or Date columns.")

    category_cols = []

    for col in df.columns:
        low = col.lower()

        if col in ["Date", year_col, month_col, date_col]:
            continue
        if "total" in low:
            continue
        if "electric" in low:
            category_cols.append(col)

    if not category_cols:
        raise ValueError("Could not detect category columns like Electric 2W, Electric 3W, Electric 4W.")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in category_cols:
        df[col] = clean_num(df[col]).fillna(0)

    category_monthly = df.melt(
        id_vars=["Date"],
        value_vars=category_cols,
        var_name="Category",
        value_name="EV_Registrations",
    )

    category_monthly["Category"] = category_monthly["Category"].apply(normalize_category_name)

    category_monthly = (
        category_monthly.groupby(["Date", "Category"], as_index=False)["EV_Registrations"]
        .sum()
    )

    if total_col and total_col in df.columns:
        df[total_col] = clean_num(df[total_col]).fillna(0)
        monthly_total = (
            df.groupby("Date", as_index=False)[total_col]
            .sum()
            .rename(columns={total_col: "EV_Registrations"})
        )
    else:
        monthly_total = (
            category_monthly.groupby("Date", as_index=False)["EV_Registrations"]
            .sum()
        )

    monthly_total["Year"] = monthly_total["Date"].dt.year
    monthly_total["Month"] = monthly_total["Date"].dt.month
    monthly_total["Month_Name"] = monthly_total["Date"].dt.strftime("%b")

    monthly_raw = category_monthly.copy()
    monthly_raw["Year"] = monthly_raw["Date"].dt.year
    monthly_raw["Month"] = monthly_raw["Date"].dt.month
    monthly_raw["Month_Name"] = monthly_raw["Date"].dt.strftime("%b")

    return monthly_raw, monthly_total, category_monthly


def prepare_statewise_electric(df):
    df = remove_total_rows(clean_column_names(df))

    state_col = find_col(df, ["State"])
    electric_cols = [
        col for col in df.columns
        if "electric" in col.lower() or "bov" in col.lower()
    ]

    if state_col is None:
        raise ValueError("State column was not found in IMP_Ev_statewise.csv.")

    if not electric_cols:
        raise ValueError("ELECTRIC(BOV) columns were not found in IMP_Ev_statewise.csv.")

    keep = df[[state_col] + electric_cols].copy()
    keep = keep.rename(columns={state_col: "State"})
    keep["State"] = keep["State"].astype(str).str.strip().str.title()

    rows = []

    for col in electric_cols:
        year_digits = "".join(ch for ch in str(col) if ch.isdigit())

        if len(year_digits) >= 4:
            temp = keep[["State", col]].copy()
            temp["Year"] = int(year_digits[:4])
            temp["EV_Registrations"] = clean_num(temp[col]).fillna(0)
            rows.append(temp[["State", "Year", "EV_Registrations"]])

    if not rows:
        raise ValueError("Electric columns were found, but year labels could not be parsed.")

    state_long = pd.concat(rows, ignore_index=True)
    state_long = (
        state_long.groupby(["State", "Year"], as_index=False)["EV_Registrations"]
        .sum()
    )

    latest_year = int(state_long["Year"].max())

    latest_state = state_long[state_long["Year"] == latest_year].rename(
        columns={"EV_Registrations": "Latest_EV_Registrations"}
    )

    return state_long, latest_state


def forecast_monthly(actual_df, degree=2):
    actual = actual_df[["Date", "EV_Registrations"]].copy().sort_values("Date")
    actual = actual.groupby("Date", as_index=False)["EV_Registrations"].sum()
    actual = actual[actual["EV_Registrations"] > 0]

    if len(actual) < 3:
        raise ValueError("At least 3 non-zero monthly records are required for forecasting.")

    last_date = actual["Date"].max()

    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        f"{FORECAST_END_YEAR}-12-01",
        freq="MS",
    )

    all_dates = pd.concat([actual["Date"], pd.Series(future_dates)], ignore_index=True)

    x_actual = np.arange(len(actual)).reshape(-1, 1)
    x_all = np.arange(len(all_dates)).reshape(-1, 1)

    model = make_pipeline(
        PolynomialFeatures(degree),
        Ridge(alpha=1.0),
    )

    model.fit(x_actual, actual["EV_Registrations"])

    fitted = model.predict(x_actual)
    predicted = np.clip(model.predict(x_all), 0, None)

    r2 = r2_score(actual["EV_Registrations"], fitted) if len(actual) > 1 else np.nan
    ci = 1.96 * float(np.std(actual["EV_Registrations"] - fitted))

    forecast = pd.DataFrame({
        "Date": all_dates,
        "Actual": list(actual["EV_Registrations"]) + [np.nan] * len(future_dates),
        "Predicted": predicted,
        "CI_Lower": np.clip(predicted - ci, 0, None),
        "CI_Upper": predicted + ci,
    })

    forecast["Year"] = forecast["Date"].dt.year
    forecast["Month"] = forecast["Date"].dt.month

    return forecast, r2, last_date


def forecast_categories(category_monthly):
    frames = []
    metrics = []

    for category, group in category_monthly.groupby("Category"):
        if len(group) < 6 or group["EV_Registrations"].sum() <= 0:
            continue

        pred, r2, last_date = forecast_monthly(group, degree=2)
        pred["Category"] = category
        frames.append(pred)

        metrics.append({
            "Category": category,
            "Model_Fit_Score": r2,
            "Last_Actual_Month": last_date.strftime("%b %Y"),
        })

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)


def simulated_state_forecast(state_long, annual_forecast, start_year):
    first_year = int(state_long["Year"].min())
    latest_year = int(state_long["Year"].max())

    first = state_long[state_long["Year"] == first_year][["State", "EV_Registrations"]]
    first = first.rename(columns={"EV_Registrations": "First_EV"})

    latest = state_long[state_long["Year"] == latest_year][["State", "EV_Registrations"]]
    latest = latest.rename(columns={"EV_Registrations": "Latest_EV"})

    weights = latest.merge(first, on="State", how="left").fillna(0)

    latest_total = weights["Latest_EV"].sum()
    weights["Latest_Share"] = weights["Latest_EV"] / latest_total if latest_total > 0 else 0

    weights["Growth_Momentum"] = (
        np.log1p(weights["Latest_EV"]) - np.log1p(weights["First_EV"])
    )

    if weights["Growth_Momentum"].max() > weights["Growth_Momentum"].min():
        weights["Growth_Momentum_Normalized"] = (
            (weights["Growth_Momentum"] - weights["Growth_Momentum"].min())
            / (weights["Growth_Momentum"].max() - weights["Growth_Momentum"].min())
        )
    else:
        weights["Growth_Momentum_Normalized"] = 0

    base_weight = (
        0.75 * weights["Latest_Share"]
        + 0.25 * (
            weights["Growth_Momentum_Normalized"]
            / max(weights["Growth_Momentum_Normalized"].sum(), 1)
        )
    )

    weights["Base_Simulation_Weight"] = base_weight / base_weight.sum()

    annual_future = annual_forecast[
        (annual_forecast["Year"] >= start_year)
        & (annual_forecast["Year"] <= FORECAST_END_YEAR)
    ]

    rows = []

    for _, state_row in weights.iterrows():
        for _, year_row in annual_future.iterrows():
            years_ahead = int(year_row["Year"]) - start_year
            momentum_boost = 1 + (state_row["Growth_Momentum_Normalized"] * 0.03 * years_ahead)

            rows.append({
                "State": state_row["State"],
                "Year": int(year_row["Year"]),
                "Raw_Adjusted_Weight": state_row["Base_Simulation_Weight"] * momentum_boost,
                "National_Predicted_EV": float(year_row["Predicted"]),
            })

    simulated = pd.DataFrame(rows)

    simulated["Simulation_Weight"] = simulated.groupby("Year")["Raw_Adjusted_Weight"].transform(
        lambda x: x / x.sum()
    )

    simulated["Simulated_EV_Registrations"] = (
        simulated["National_Predicted_EV"] * simulated["Simulation_Weight"]
    ).round()

    simulated = simulated[[
        "State",
        "Year",
        "Simulated_EV_Registrations",
        "Simulation_Weight",
    ]]

    return simulated, weights.sort_values("Base_Simulation_Weight", ascending=False)


st.title("Electric Vehicle Penetration - India")
st.caption("Client-ready analytics and forecast dashboard using EV registration datasets")

try:
    raw = load_data()
    monthly_raw, monthly_total, category_monthly = prepare_monthly_category(raw["monthly_category"])
    state_long, latest_state = prepare_statewise_electric(raw["statewise"])
    penetration = remove_total_rows(clean_column_names(raw["penetration"]))

except Exception as e:
    st.error("Failed to load or prepare dashboard datasets.")
    st.exception(e)
    st.stop()


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "State Analysis",
    "Penetration",
    "Class / Category",
    "Monthly Trends",
    "Forecast to 2030",
    "Simulated State Forecast",
])


with tab1:
    st.subheader("National EV Growth Overview")

    annual = monthly_total.groupby("Year", as_index=False)["EV_Registrations"].sum()

    start_year = int(annual["Year"].min())
    latest_year = int(annual["Year"].max())

    first_total = annual.loc[annual["Year"] == start_year, "EV_Registrations"].sum()
    latest_total = annual.loc[annual["Year"] == latest_year, "EV_Registrations"].sum()

    growth = ((latest_total - first_total) / first_total * 100) if first_total > 0 else np.nan
    growth_cagr = cagr(first_total, latest_total, latest_year - start_year)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(f"Latest EV Registrations ({latest_year})", f"{latest_total:,.0f}")
    c2.metric(f"Starting EV Registrations ({start_year})", f"{first_total:,.0f}")
    c3.metric("Overall Growth", f"{growth:.1f}%" if pd.notna(growth) else "N/A")
    c4.metric("CAGR", f"{growth_cagr:.1f}%" if pd.notna(growth_cagr) else "N/A")

    st.info(
        "Formula: Overall Growth = ((Latest Year EVs - First Year EVs) / First Year EVs) x 100. "
        "CAGR = ((Latest Year EVs / First Year EVs) ^ (1 / number of years) - 1) x 100."
    )

    fig = px.area(
        annual,
        x="Year",
        y="EV_Registrations",
        markers=True,
        title=f"Annual EV Registrations in India ({start_year} to {latest_year})",
    )

    fig.update_layout(showlegend=False, yaxis_title="EV Registrations")
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("State-wise EV Registration Analysis")

    latest_state_year = int(state_long["Year"].max())

    st.info(
        f"This view uses only ELECTRIC(BOV) records from IMP_Ev_statewise.csv. "
        f"Other fuel categories are excluded. Current ranking is based on {latest_state_year}."
    )

    top_states = latest_state.sort_values(
        "Latest_EV_Registrations",
        ascending=False,
    ).head(10)

    fig = px.bar(
        top_states,
        x="State",
        y="Latest_EV_Registrations",
        title=f"Top 10 States/UTs by ELECTRIC(BOV) Registrations ({latest_state_year})",
        color="Latest_EV_Registrations",
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    state_choice = st.selectbox(
        "Select a state for historical trend",
        sorted(state_long["State"].unique()),
    )

    state_view = state_long[state_long["State"] == state_choice]

    if len(state_view) <= 1:
        st.warning(
            "Only one historical year is available for this state in the cleaned statewise data, "
            "so a trend line cannot be interpreted."
        )

    fig2 = px.line(
        state_view,
        x="Year",
        y="EV_Registrations",
        markers=True,
        title=f"ELECTRIC(BOV) Historical Trend - {state_choice}",
    )

    st.plotly_chart(fig2, use_container_width=True)
    show_dataset_checks(state_long, "State-wise ELECTRIC(BOV) cleaned data")


with tab3:
    st.subheader("EV Penetration Percentage")

    pen_state_col = find_col(penetration, ["State"])
    percent_cols = [c for c in penetration.columns if "Percent" in c or "Share" in c]
    pen_col = percent_cols[-1] if percent_cols else penetration.columns[-1]

    penetration[pen_col] = clean_num(penetration[pen_col])

    st.info("Formula: EV Penetration % = (EV registrations / total vehicle registrations) x 100.")

    top_pen = penetration.sort_values(pen_col, ascending=False).head(10)

    fig = px.bar(
        top_pen,
        x=pen_state_col,
        y=pen_col,
        title="Top 10 States/UTs by EV Penetration",
        color=pen_col,
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.subheader("Vehicle Category-wise EV Analysis")

    st.info(
        "This section uses separate monthly category columns: Electric 2W, Electric 3W, "
        "Electric 4W, Electric Goods, and Electric Bus."
    )

    category_summary = (
        category_monthly.groupby("Category", as_index=False)["EV_Registrations"]
        .sum()
        .sort_values("EV_Registrations", ascending=False)
    )

    fig = px.bar(
        category_summary,
        x="Category",
        y="EV_Registrations",
        title="Total EV Registrations by Category",
        color="Category",
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        category_monthly,
        x="Date",
        y="EV_Registrations",
        color="Category",
        title="Monthly EV Trend by Category",
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(category_summary, use_container_width=True)


with tab5:
    st.subheader("Monthly EV Registration Trends")

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

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        monthly_total.dropna(subset=["YoY_Growth_Pct"]),
        x="Date",
        y="YoY_Growth_Pct",
        title="Year-over-Year Monthly EV Growth",
        color="YoY_Growth_Pct",
        color_continuous_midpoint=0,
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "How to read YoY growth: each bar compares a month with the same month in the previous year. "
        "Positive bars mean registrations increased year-over-year. Negative bars mean registrations declined. "
        "Very tall bars usually appear when the previous year's same month had a very small base."
    )

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

    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "How to read the heatmap: rows are years and columns are months. Darker green means higher EV registrations. "
        "Seasonality means repeated monthly patterns. If some months are darker across many years, those months are "
        "stronger registration periods."
    )


with tab6:
    st.subheader("Monthly and Category Forecast to 2030")

    national_forecast, national_r2, last_actual_month = forecast_monthly(monthly_total, degree=2)
    category_forecast, category_metrics = forecast_categories(category_monthly)

    latest_actual_year = int(monthly_total["Year"].max())

    st.info(
        f"Forecast period: actual data is used through {last_actual_month.strftime('%B %Y')}. "
        "Predictions continue from the next month through December 2030. "
        "Model used: fixed polynomial ridge regression."
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

    annual_forecast.loc[
        annual_forecast["Data_Type"] == "Forecast",
        "Actual_EV_Registrations",
    ] = np.nan

    annual_forecast["Forecast_YoY_Growth_Percent"] = (
        annual_forecast["Predicted_EV_Registrations"].pct_change() * 100
    )

    latest_actual_total = actual_annual.loc[
        actual_annual["Year"] == latest_actual_year,
        "EV_Registrations",
    ].sum()

    annual_forecast[f"Growth_vs_Latest_Actual_{latest_actual_year}_Percent"] = annual_forecast[
        "Predicted_EV_Registrations"
    ].apply(lambda x: safe_pct_change(x, latest_actual_total))

    annual_forecast_display = annual_forecast.copy()
    annual_forecast_display["Actual_EV_Registrations"] = annual_forecast_display[
        "Actual_EV_Registrations"
    ].round(0)
    annual_forecast_display["Predicted_EV_Registrations"] = annual_forecast_display[
        "Predicted_EV_Registrations"
    ].round(0)
    annual_forecast_display["Forecast_YoY_Growth_Percent"] = annual_forecast_display[
        "Forecast_YoY_Growth_Percent"
    ].round(2)
    annual_forecast_display[f"Growth_vs_Latest_Actual_{latest_actual_year}_Percent"] = annual_forecast_display[
        f"Growth_vs_Latest_Actual_{latest_actual_year}_Percent"
    ].round(2)

    category_yearly_forecast = (
        category_forecast.groupby(["Year", "Category"], as_index=False)["Predicted"]
        .sum()
        .rename(columns={"Predicted": "Predicted_EV_Registrations"})
    )

    category_actual_latest = (
        category_monthly[category_monthly["Date"].dt.year == latest_actual_year]
        .groupby("Category", as_index=False)["EV_Registrations"]
        .sum()
        .rename(columns={"EV_Registrations": f"Latest_Actual_EV_Registrations_{latest_actual_year}"})
    )

    forecast_year_options = list(range(latest_actual_year + 1, FORECAST_END_YEAR + 1))

    selected_category_year = st.selectbox(
        "Select category forecast year",
        forecast_year_options,
        index=len(forecast_year_options) - 1,
    )

    selected_category_table = category_yearly_forecast[
        category_yearly_forecast["Year"] == selected_category_year
    ].copy()

    previous_category_table = category_yearly_forecast[
        category_yearly_forecast["Year"] == selected_category_year - 1
    ][["Category", "Predicted_EV_Registrations"]].rename(
        columns={"Predicted_EV_Registrations": "Previous_Year_Predicted_EV_Registrations"}
    )

    selected_category_table = selected_category_table.merge(
        previous_category_table,
        on="Category",
        how="left",
    )

    selected_category_table = selected_category_table.merge(
        category_actual_latest,
        on="Category",
        how="left",
    )

    selected_category_table["YoY_Growth_Percent"] = selected_category_table.apply(
        lambda row: safe_pct_change(
            row["Predicted_EV_Registrations"],
            row["Previous_Year_Predicted_EV_Registrations"],
        ),
        axis=1,
    )

    latest_actual_col = f"Latest_Actual_EV_Registrations_{latest_actual_year}"

    selected_category_table[f"Growth_vs_Latest_Actual_{latest_actual_year}_Percent"] = selected_category_table.apply(
        lambda row: safe_pct_change(
            row["Predicted_EV_Registrations"],
            row[latest_actual_col],
        ),
        axis=1,
    )

    selected_category_table = selected_category_table.sort_values(
        "Predicted_EV_Registrations",
        ascending=False,
    )

    category_prediction_display = selected_category_table.copy()
    category_prediction_display["Predicted_EV_Registrations"] = category_prediction_display[
        "Predicted_EV_Registrations"
    ].round(0)
    category_prediction_display["Previous_Year_Predicted_EV_Registrations"] = category_prediction_display[
        "Previous_Year_Predicted_EV_Registrations"
    ].round(0)
    category_prediction_display[latest_actual_col] = category_prediction_display[
        latest_actual_col
    ].round(0)
    category_prediction_display["YoY_Growth_Percent"] = category_prediction_display[
        "YoY_Growth_Percent"
    ].round(2)
    category_prediction_display[f"Growth_vs_Latest_Actual_{latest_actual_year}_Percent"] = category_prediction_display[
        f"Growth_vs_Latest_Actual_{latest_actual_year}_Percent"
    ].round(2)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "National Model Fit Score",
            f"{national_r2:.3f}" if pd.notna(national_r2) else "N/A",
        )

    with c2:
        top_category = selected_category_table.iloc[0]["Category"]
        top_category_value = selected_category_table.iloc[0]["Predicted_EV_Registrations"]

        st.metric(
            f"Top Category ({selected_category_year})",
            top_category,
            f"{top_category_value:,.0f} predicted EVs",
        )

    st.write("Annual national forecast")
    st.dataframe(annual_forecast_display.tail(8), use_container_width=True)

    st.write(f"Category-wise prediction table ({selected_category_year})")
    st.dataframe(category_prediction_display, use_container_width=True)

    second_category = (
        selected_category_table.iloc[1]["Category"]
        if len(selected_category_table) > 1
        else "the next category"
    )

    st.success(
        f"For {selected_category_year}, {top_category} is projected to be the largest EV category, "
        f"followed by {second_category}. The YoY Growth % compares {selected_category_year} with "
        f"{selected_category_year - 1}, while Growth vs Latest Actual compares the selected forecast year "
        f"with the latest actual year ({latest_actual_year})."
    )

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

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Model diagnostics"):
        st.write(
            "This table is used to check model reliability. A Model Fit Score closer to 1 means the model "
            "matches the historical category trend more closely. It is not the main business output."
        )
        st.dataframe(
            category_metrics.sort_values("Model_Fit_Score", ascending=False),
            use_container_width=True,
        )


with tab7:
    st.subheader("Simulated State Forecast to 2030")

    national_forecast, _, _ = forecast_monthly(monthly_total, degree=2)

    annual_forecast = national_forecast.groupby("Year", as_index=False).agg(
        Predicted=("Predicted", "sum"),
    )

    state_actual_start_year = int(state_long["Year"].min())
    state_actual_latest_year = int(state_long["Year"].max())

    monthly_actual_start_year = int(monthly_total["Year"].min())
    monthly_actual_latest_year = int(monthly_total["Year"].max())

    start_year = monthly_actual_latest_year + 1

    simulated, weights = simulated_state_forecast(
        state_long,
        annual_forecast,
        start_year,
    )

    st.warning(
        "This is a simulated state allocation forecast. It is not direct observed data and not a direct "
        "state-level ML forecast. The national EV forecast is distributed across states using ELECTRIC(BOV) "
        "share and growth momentum."
    )

    st.info(
    f"Monthly national actual data is available from {monthly_actual_start_year} to "
    f"{monthly_actual_latest_year}. Statewise ELECTRIC(BOV) actual data is available from "
    f"{state_actual_start_year} to {state_actual_latest_year} and is used only to calculate "
    "state allocation weights. The simulated state forecast is displayed from "
    f"{start_year} to {FORECAST_END_YEAR}.")

    selected_year = st.selectbox(
        "Select simulated forecast year",
        sorted(simulated["Year"].unique()),
        index=len(sorted(simulated["Year"].unique())) - 1,
    )

    latest_actual_state = (
        state_long[state_long["Year"] == state_actual_latest_year]
        .groupby("State", as_index=False)["EV_Registrations"]
        .sum()
        .rename(columns={"EV_Registrations": f"Actual_EV_Registrations_{state_actual_latest_year}"})
    )

    previous_year_simulated = (
        simulated[simulated["Year"] == selected_year - 1]
        .groupby("State", as_index=False)["Simulated_EV_Registrations"]
        .sum()
        .rename(columns={"Simulated_EV_Registrations": "Previous_Year_Simulated_EV_Registrations"})
    )

    year_view = (
        simulated[simulated["Year"] == selected_year]
        .groupby(["State", "Year"], as_index=False)
        .agg(
            Simulated_EV_Registrations=("Simulated_EV_Registrations", "sum"),
            Simulation_Weight=("Simulation_Weight", "mean"),
        )
    )

    year_view = year_view.merge(
        latest_actual_state,
        on="State",
        how="left",
    )

    year_view = year_view.merge(
        previous_year_simulated,
        on="State",
        how="left",
    )

    actual_col = f"Actual_EV_Registrations_{state_actual_latest_year}"

    year_view["YoY_Simulated_Growth_Percent"] = year_view.apply(
        lambda row: safe_pct_change(
            row["Simulated_EV_Registrations"],
            row["Previous_Year_Simulated_EV_Registrations"],
        ),
        axis=1,
    )

    year_view[f"Growth_vs_Actual_{state_actual_latest_year}_Percent"] = year_view.apply(
        lambda row: safe_pct_change(
            row["Simulated_EV_Registrations"],
            row[actual_col],
        ),
        axis=1,
    )

    year_view = year_view.sort_values(
        "Simulated_EV_Registrations",
        ascending=False,
    )

    year_view_display = year_view.copy()
    year_view_display["Simulated_EV_Registrations"] = year_view_display[
        "Simulated_EV_Registrations"
    ].round(0)
    year_view_display["Simulation_Weight"] = year_view_display[
        "Simulation_Weight"
    ].round(4)
    year_view_display[actual_col] = year_view_display[actual_col].round(0)
    year_view_display["Previous_Year_Simulated_EV_Registrations"] = year_view_display[
        "Previous_Year_Simulated_EV_Registrations"
    ].round(0)
    year_view_display["YoY_Simulated_Growth_Percent"] = year_view_display[
        "YoY_Simulated_Growth_Percent"
    ].round(2)
    year_view_display[f"Growth_vs_Actual_{state_actual_latest_year}_Percent"] = year_view_display[
        f"Growth_vs_Actual_{state_actual_latest_year}_Percent"
    ].round(2)

    fig = px.bar(
        year_view.head(12),
        x="State",
        y="Simulated_EV_Registrations",
        title=f"Top Simulated State EV Registrations ({selected_year})",
        color="Simulated_EV_Registrations",
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    top_state_name = year_view.iloc[0]["State"]
    top_state_value = year_view.iloc[0]["Simulated_EV_Registrations"]
    top_state_growth = year_view.iloc[0][f"Growth_vs_Actual_{state_actual_latest_year}_Percent"]

    st.success(
        f"For {selected_year}, {top_state_name} is projected to have the highest simulated EV count "
        f"with approximately {top_state_value:,.0f} EV registrations. "
        f"This is a simulated increase of {top_state_growth:.1f}% compared with its actual "
        f"{state_actual_latest_year} EV registrations."
    )

    st.write(f"State-wise simulated prediction table ({selected_year})")
    st.dataframe(year_view_display, use_container_width=True)

    state_choice = st.selectbox(
        "Select a state for simulated forecast trend",
        sorted(simulated["State"].unique()),
    )

    selected_state_actual = state_long[state_long["State"] == state_choice].copy()
    selected_state_actual = selected_state_actual.rename(
        columns={"EV_Registrations": "EV_Registrations"}
    )
    selected_state_actual["Data_Type"] = "Actual"

    selected_state_simulated = simulated[simulated["State"] == state_choice].copy()
    selected_state_simulated = selected_state_simulated.rename(
        columns={"Simulated_EV_Registrations": "EV_Registrations"}
    )
    selected_state_simulated["Data_Type"] = "Simulated Forecast"

    state_trend_combined = pd.concat(
        [
            selected_state_actual[["State", "Year", "EV_Registrations", "Data_Type"]],
            selected_state_simulated[["State", "Year", "EV_Registrations", "Data_Type"]],
        ],
        ignore_index=True,
    )

    fig2 = px.line(
        state_trend_combined,
        x="Year",
        y="EV_Registrations",
        color="Data_Type",
        markers=True,
        title=f"Actual and Simulated EV Trend - {state_choice}",
    )

    st.plotly_chart(fig2, use_container_width=True)

    state_latest_actual_value = selected_state_actual[
        selected_state_actual["Year"] == state_actual_latest_year
    ]["EV_Registrations"].sum()

    state_2030_value = selected_state_simulated[
        selected_state_simulated["Year"] == FORECAST_END_YEAR
    ]["EV_Registrations"].sum()

    state_growth_to_2030 = safe_pct_change(state_2030_value, state_latest_actual_value)

    st.caption(
        f"For {state_choice}, the line chart combines actual ELECTRIC(BOV) data from "
        f"{state_actual_start_year} to {state_actual_latest_year} with simulated forecast values from "
        f"{start_year} to {FORECAST_END_YEAR}. The simulated 2030 value represents an estimated "
        f"{state_growth_to_2030:.1f}% increase compared with actual {state_actual_latest_year} registrations."
    )

    with st.expander("State simulation weights"):
        st.write(
            "Simulation weights show how the national forecast was allocated across states. "
            "Higher weight means the state receives a larger share of the national forecast."
        )
        st.dataframe(weights, use_container_width=True)