import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="EV Penetration India Dashboard",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_DIR = OUTPUT_DIR / "cleaned"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data
def load_data():
    return {
        "yearwise": pd.read_csv(DATA_DIR / "Ev_yearwise.csv"),
        "statewise": pd.read_csv(DATA_DIR / "Ev_statewise.csv"),
        "penetration": pd.read_csv(DATA_DIR / "Ev_penetration%wise.csv"),
        "categorywise": pd.read_csv(DATA_DIR / "Ev_categorywise.csv")
    }


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("%", "Percent")
        .str.replace(".", "", regex=False)
    )
    return df


def clean_numeric_columns(df):
    df = df.copy()
    text_keywords = ["State", "Category", "Class", "Vehicle", "Year"]

    for col in df.columns:
        is_text_col = any(keyword.lower() in col.lower() for keyword in text_keywords)

        if not is_text_col:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("NA", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def standardize_text_columns(df):
    df = df.copy()
    text_keywords = ["State", "Category", "Class", "Vehicle", "Year"]

    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in text_keywords):
            df[col] = df[col].astype(str).str.strip().str.title()

    return df


def clean_all(data):
    cleaned = {}

    for name, df in data.items():
        temp = clean_column_names(df)
        temp = standardize_text_columns(temp)
        temp = clean_numeric_columns(temp)
        cleaned[name] = temp
        temp.to_csv(CLEAN_DIR / f"{name}_cleaned.csv", index=False)

    return cleaned


def remove_total_rows(df):
    df = df.copy()

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    remove_mask = pd.Series(False, index=df.index)

    for col in text_cols:
        values = (
            df[col]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace("\u200b", "", regex=False)
            .str.lower()
            .str.strip()
        )

        remove_mask = remove_mask | values.isin(["grand total", "total"])

    return df.loc[~remove_mask].reset_index(drop=True)


def find_col(df, keywords):
    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in keywords):
            return col
    return None


def show_dataset_checks(df, dataset_name):
    with st.expander(f"{dataset_name} dataset checks"):
        st.write("Shape:", df.shape)

        c1, c2 = st.columns(2)
        with c1:
            st.write("Head")
            st.dataframe(df.head())
        with c2:
            st.write("Tail")
            st.dataframe(df.tail())

        st.write("Missing values")
        st.dataframe(
            df.isnull()
            .sum()
            .reset_index()
            .rename(columns={"index": "Column", 0: "Missing"})
        )


st.title("Electric Vehicle Penetration in India")
st.caption("Government vehicle registration data based analytics dashboard")

try:
    raw = load_data()
    data = clean_all(raw)

    yearwise = data["yearwise"]
    statewise = remove_total_rows(data["statewise"])
    penetration = remove_total_rows(data["penetration"])
    categorywise = remove_total_rows(data["categorywise"])

except Exception as e:
    st.error("App stopped while loading or cleaning data.")
    st.exception(e)
    st.stop()


with st.sidebar:
    st.header("Source Note")
    st.write(
        "Original CSV files are kept unchanged in the data folder. "
        "Cleaned copies are saved separately in outputs/cleaned."
    )
    st.write(
        "Dataset checks are shown inside each related tab."
    )


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "State Analysis",
    "Penetration",
    "Class / Category Analysis",
    "Forecast",
    "Limitations"
])


with tab1:
    year_col = find_col(yearwise, ["Year"])
    ev_col = [col for col in yearwise.columns if col != year_col][0]
    yearwise[ev_col] = pd.to_numeric(yearwise[ev_col], errors="coerce")

    start_year = str(yearwise[year_col].iloc[0])
    end_year = str(yearwise[year_col].iloc[-1])

    st.subheader(f"National EV Growth ({start_year} to {end_year})")
    show_dataset_checks(yearwise, "Year-wise")

    total_ev_latest = int(yearwise[ev_col].iloc[-1])
    total_ev_first = int(yearwise[ev_col].iloc[0])
    growth = ((total_ev_latest - total_ev_first) / total_ev_first) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Latest EV Registrations ({end_year})", f"{total_ev_latest:,}")
    m2.metric(f"First Year EV Registrations ({start_year})", f"{total_ev_first:,}")
    m3.metric(f"Overall Growth ({start_year} to {end_year})", f"{growth:.2f}%")

    fig = px.line(
        yearwise,
        x=year_col,
        y=ev_col,
        markers=True,
        title=f"Year-wise Registered Electric Vehicles in India ({start_year} to {end_year})"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"This graph shows the total number of registered EVs in India for each year from "
        f"{start_year} to {end_year}."
    )


with tab2:
    st.subheader("State-wise EV Registration Analysis")
    show_dataset_checks(statewise, "State-wise")

    state_col = find_col(statewise, ["State"])
    total_col = "Grand_Total" if "Grand_Total" in statewise.columns else find_col(statewise, ["Grand", "Total"])

    statewise = statewise[
        ~statewise[state_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.contains("grand total|total", na=False)
    ]

    statewise[total_col] = pd.to_numeric(statewise[total_col], errors="coerce")
    top_n = 10
    top_states = statewise.sort_values(total_col, ascending=False).head(top_n)

    st.markdown("**Formula:** `Grand_Total = 2020 + 2021 + 2022 + 2023`")

    fig = px.bar(
        top_states,
        x=state_col,
        y=total_col,
        title="Top 10 States/UTs by Total EV Registrations",
        color=total_col,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    col_2020 = find_col(statewise, ["2020"])
    col_2023 = find_col(statewise, ["2023"])

    if col_2020 and col_2023:
        statewise[col_2020] = pd.to_numeric(statewise[col_2020], errors="coerce")
        statewise[col_2023] = pd.to_numeric(statewise[col_2023], errors="coerce")

        statewise["Growth_2020_2023"] = statewise[col_2023] - statewise[col_2020]
        growth_top = statewise.sort_values("Growth_2020_2023", ascending=False).head(top_n)

        st.markdown("**Formula:** `Growth = EV registrations in 2023 - EV registrations in 2020`")

        fig2 = px.bar(
            growth_top,
            x=state_col,
            y="Growth_2020_2023",
            title="Top 10 States/UTs by EV Growth (2020 to 2023)",
            color="Growth_2020_2023",
            color_continuous_scale="Blues"
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "The state-wise graph shows which states/UTs have the highest EV registrations. "
    )


with tab3:
    st.subheader("EV Penetration Percentage")
    st.markdown("**Formula:** `EV Penetration % = (EV Registrations / Total Vehicle Registrations) * 100`")

    penetration = remove_total_rows(penetration)
    show_dataset_checks(penetration, "Penetration-wise")

    pen_state_col = find_col(penetration, ["State"])
    percent_cols = [col for col in penetration.columns if "Percent" in col or "Share" in col]
    pen_col = percent_cols[-1] if percent_cols else penetration.columns[-1]

    penetration = penetration[
        ~penetration[pen_state_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.contains("grand total|total", na=False)
    ]

    penetration[pen_col] = pd.to_numeric(penetration[pen_col], errors="coerce")
    top_pen = penetration.sort_values(pen_col, ascending=False).head(10)

    fig = px.bar(
        top_pen,
        x=pen_state_col,
        y=pen_col,
        title="Top 10 States/UTs by EV Penetration Percentage",
        color=pen_col,
        color_continuous_scale="Teal"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    numeric_cols = penetration.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        st.markdown(
            "**Reading this chart:** Each dot is one state/UT. "
            "X-axis shows total EV registrations, and Y-axis shows EV penetration percentage."
        )

        fig2 = px.scatter(
            penetration,
            x="Total_EV" if "Total_EV" in penetration.columns else numeric_cols[-2],
            y=pen_col,
            hover_name=pen_state_col,
            title="Total EV Registrations vs EV Penetration Percentage",
            size=pen_col
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "EV penetration percentage means EV registrations as a share of total vehicle registrations. "
        "It is not the same as EV growth percentage."
    )

with tab4:
    st.subheader("Vehicle Class / Category-wise EV Analysis")

    categorywise = remove_total_rows(categorywise)
    show_dataset_checks(categorywise, "Class / Category-wise")

    st.write(
        "This section shows which EV vehicle classes/categories contribute most across all available years "
        "in the category dataset."
    )

    category_col = "Category" if "Category" in categorywise.columns else find_col(categorywise, ["Category", "Class"])

    year_cols = [
        col for col in categorywise.columns
        if str(col).isdigit()
    ]

    for col in year_cols:
        categorywise[col] = pd.to_numeric(categorywise[col], errors="coerce")

    if category_col and year_cols:
        categorywise["Overall_Total"] = categorywise[year_cols].sum(axis=1)
        category_plot = categorywise.sort_values("Overall_Total", ascending=False)

        fig = px.bar(
            category_plot,
            x=category_col,
            y="Overall_Total",
            title="Vehicle Class / Category-wise Total EV Registrations/Sales Across Available Years",
            color="Overall_Total",
            color_continuous_scale="Plasma"
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.pie(
            category_plot,
            names=category_col,
            values="Overall_Total",
            title="Overall Share of EV Registrations/Sales by Vehicle Class / Category"
        )
        st.plotly_chart(fig2, use_container_width=True)

        if "Percent_Growth" in categorywise.columns:
            categorywise["Percent_Growth"] = pd.to_numeric(categorywise["Percent_Growth"], errors="coerce")
            growth_plot = categorywise.sort_values("Percent_Growth", ascending=False)

            fig3 = px.bar(
                growth_plot,
                x=category_col,
                y="Percent_Growth",
                title="Vehicle Class / Category-wise Percent Growth",
                color="Percent_Growth",
                color_continuous_scale="Blues"
            )
            fig3.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "The bar and pie chart use overall category totals across available year columns. ")


with tab5:
    st.subheader("Simple EV Forecast")
    show_dataset_checks(yearwise, "Forecast input: year-wise")

    model_df = yearwise.copy()
    model_df.columns = ["Year", "EV_Count"]
    model_df["EV_Count"] = pd.to_numeric(model_df["EV_Count"], errors="coerce")
    model_df = model_df.dropna(subset=["EV_Count"])
    model_df["Year_Index"] = np.arange(len(model_df))

    X = model_df[["Year_Index"]]
    y = model_df["EV_Count"]

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_DIR / "ev_forecast_model.pkl", "wb") as f:
        pickle.dump(model, f)

    future_years = ["2024-25", "2025-26", "2026-27", "2027-28"]
    future_index = pd.DataFrame({
        "Year_Index": [
            len(model_df),
            len(model_df) + 1,
            len(model_df) + 2,
            len(model_df) + 3
        ]
    })
    preds = model.predict(future_index).astype(int)

    forecast_df = pd.DataFrame({
        "Year": future_years,
        "EV_Count": preds,
        "Type": "Predicted"
    })

    actual_df = pd.DataFrame({
        "Year": model_df["Year"],
        "EV_Count": model_df["EV_Count"],
        "Type": "Actual"
    })

    final_forecast = pd.concat([actual_df, forecast_df], ignore_index=True)
    final_forecast.to_csv(OUTPUT_DIR / "forecast_output.csv", index=False)

    fig = px.line(
        final_forecast,
        x="Year",
        y="EV_Count",
        color="Type",
        markers=True,
        title="Actual and Predicted EV Registrations"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast_df)

    st.info(
        "The forecast starts after the latest available actual year, 2023-24. "
        "Values for 2024-25 and 2025-26 are estimated because actual data is not included in this dataset. "
        "Values for 2026-27 and 2027-28 are future projections. "
        "Because only limited yearly observations are available, this is a trend estimate."
    )


with tab6:
    st.subheader("Project Limitations")

    st.write(
        "Manufacturer-wise analysis was considered, but it was not included because the selected official "
        "downloadable data.gov.in VAHAN/e-Vahan datasets do not contain manufacturer-level information."
    )

    st.write(
        "Official monthly EV registration data may be available in dashboard form through VAHAN/Parivahan, "
        "but a clean downloadable monthly CSV was not available for this project. Therefore, the forecast uses "
        "year-wise EV registration data."
    )

    st.write(
        "The prediction model is intentionally simple because the year-wise dataset has only a small number "
        "of observations. The forecast should be interpreted as a trend estimate."
    )

    st.write(
        "The project focuses on government registration data, so private datasets such as Kaggle were not used "
        "as the main source."
    )