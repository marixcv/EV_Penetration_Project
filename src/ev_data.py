from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


FORECAST_END_YEAR = 2030


STATE_NAME_FIXES = {
    "andaman and nicobar island": "Andaman And Nicobar Islands",
    "andaman and nicobar islands": "Andaman And Nicobar Islands",
    "dadra and nagar haveli and daman and diu": "Dadra And Nagar Haveli And Daman And Diu",
    "ut of dnh and dd": "Dadra And Nagar Haveli And Daman And Diu",
    "jammu and kashmir": "Jammu And Kashmir",
    "nct of delhi": "Delhi",
}


def make_unique_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []

    for col in columns:
        col = str(col).strip()
        if col not in seen:
            seen[col] = 0
            output.append(col)
        else:
            seen[col] += 1
            output.append(f"{col}_{seen[col]}")

    return output


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
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


def clean_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("NA", "", regex=False)
        .str.replace("nan", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        if any(keyword.lower() in str(col).lower() for keyword in keywords):
            return col
    return None


def normalize_state_name(value) -> str:
    text = str(value).replace("\u00a0", " ").replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    key = text.lower()
    return STATE_NAME_FIXES.get(key, text.title())


def standardize_wheel_category(value) -> str:
    text = str(value).strip()
    low = text.lower()

    if "all vehicle" in low:
        return "All Vehicles"
    if low == "2w" or "two wheeler" in low:
        return "2W"
    if "3w passenger" in low or "three wheeler (passenger" in low:
        return "3W Passenger"
    if "3w goods" in low or "three wheeler (goods" in low:
        return "3W Goods"
    if low == "cars" or "car" in low or "cab" in low:
        return "Cars"
    if low in {"bus", "buses"} or "bus" in low:
        return "Buses"
    if "lgv" in low or "mgv" in low or "hgv" in low or "tonnes" in low:
        return "Goods Vehicles"
    if "other" in low:
        return "Others"

    return text.title()


def normalize_fuel(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).title()


def is_ev_fuel_value(value) -> bool:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)

    if "hybrid" in text:
        return False
    return text in {"ev", "electric", "electric(bov)", "electric (bov)", "electric bov"}


def safe_pct_change(current, previous) -> float:
    if pd.isna(current) or pd.isna(previous) or float(previous) == 0:
        return np.nan
    return ((float(current) - float(previous)) / float(previous)) * 100


def cagr(start, end, years) -> float:
    if pd.isna(start) or pd.isna(end) or float(start) <= 0 or years <= 0:
        return np.nan
    return ((float(end) / float(start)) ** (1 / years) - 1) * 100


def remove_total_rows(df: pd.DataFrame) -> pd.DataFrame:
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


def load_wheel_fuel_yearly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="FY2011-2025")
    year_cols = [col for col in raw.columns if re.fullmatch(r"FY\d{4}", str(col))]

    national = raw.iloc[0:4][["Unnamed: 0", "Unnamed: 1"] + year_cols].copy()
    national = national.rename(columns={"Unnamed: 0": "Category", "Unnamed: 1": "Fuel_Type"})
    national["Category"] = "All Vehicles"

    category = raw.iloc[7:].copy()
    category = category.rename(columns={"Unnamed: 0": "Category", "Unnamed: 1": "Fuel_Type"})
    category = category[["Category", "Fuel_Type"] + year_cols]
    category = category.dropna(subset=["Category", "Fuel_Type"])
    category = category[
        ~category["Category"].astype(str).str.contains("source|category", case=False, na=False)
    ]

    yearly = pd.concat([national, category], ignore_index=True)
    yearly["Category"] = yearly["Category"].apply(standardize_wheel_category)
    yearly["Fuel_Type"] = yearly["Fuel_Type"].apply(normalize_fuel)

    yearly_long = yearly.melt(
        id_vars=["Category", "Fuel_Type"],
        value_vars=year_cols,
        var_name="Financial_Year",
        value_name="Registrations",
    )

    yearly_long["Year"] = yearly_long["Financial_Year"].str.extract(r"(\d{4})").astype(int)
    yearly_long["Registrations"] = clean_num(yearly_long["Registrations"]).fillna(0)
    yearly_long["Is_EV"] = yearly_long["Fuel_Type"].apply(is_ev_fuel_value)

    yearly_long = (
        yearly_long.groupby(["Category", "Fuel_Type", "Financial_Year", "Year", "Is_EV"], as_index=False)
        ["Registrations"]
        .sum()
    )

    return yearly_long.sort_values(["Category", "Fuel_Type", "Year"]).reset_index(drop=True)


def compute_cagr_tables(yearly: pd.DataFrame, start_year: int = 2015, end_year: int = 2025):
    ev = yearly[yearly["Is_EV"]].copy()

    national = ev[ev["Category"] == "All Vehicles"].groupby("Year", as_index=False)["Registrations"].sum()
    start_value = national.loc[national["Year"] == start_year, "Registrations"].sum()
    end_value = national.loc[national["Year"] == end_year, "Registrations"].sum()

    overall = {
        "Start_Year": start_year,
        "End_Year": end_year,
        "Start_EV_Registrations": start_value,
        "End_EV_Registrations": end_value,
        "CAGR_Percent": cagr(start_value, end_value, end_year - start_year),
    }

    category = (
        ev[ev["Category"] != "All Vehicles"]
        .groupby(["Category", "Year"], as_index=False)["Registrations"]
        .sum()
    )

    start = category[category["Year"] == start_year][["Category", "Registrations"]].rename(
        columns={"Registrations": "Start_EV_Registrations"}
    )
    end = category[category["Year"] == end_year][["Category", "Registrations"]].rename(
        columns={"Registrations": "End_EV_Registrations"}
    )

    category_cagr = start.merge(end, on="Category", how="outer").fillna(0)
    category_cagr["CAGR_Percent"] = category_cagr.apply(
        lambda row: cagr(row["Start_EV_Registrations"], row["End_EV_Registrations"], end_year - start_year),
        axis=1,
    )
    category_cagr = category_cagr.sort_values("CAGR_Percent", ascending=False, na_position="last")

    return overall, category_cagr


def compute_penetration_tables(yearly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = (
        yearly[yearly["Fuel_Type"].str.lower() == "total"]
        .groupby(["Category", "Year"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "Total_Registrations"})
    )

    ev = (
        yearly[yearly["Is_EV"]]
        .groupby(["Category", "Year"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "EV_Registrations"})
    )

    penetration = total.merge(ev, on=["Category", "Year"], how="left").fillna({"EV_Registrations": 0})
    penetration["EV_Penetration_Percent"] = (
        penetration["EV_Registrations"] / penetration["Total_Registrations"].replace(0, np.nan)
    ) * 100

    national = penetration[penetration["Category"] == "All Vehicles"].sort_values("Year").reset_index(drop=True)
    category = penetration[penetration["Category"] != "All Vehicles"].sort_values(
        ["Year", "EV_Penetration_Percent"], ascending=[True, False]
    )

    return national, category.reset_index(drop=True)


def normalize_category_name(col) -> str:
    text = str(col).replace("_", " ").strip().lower()
    text = text.replace("electric", "").strip()

    if "2w" in text or "2 w" in text or "two" in text:
        return "2W"
    if "3w" in text or "3 w" in text or "three" in text:
        return "3W"
    if "4w" in text or "4 w" in text or "car" in text:
        return "Cars"
    if "goods" in text:
        return "Goods Vehicles"
    if "bus" in text:
        return "Buses"

    return str(col).replace("_", " ").strip().title()


def prepare_monthly_category(path: Path | pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path) if isinstance(path, Path) else path.copy()
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
        raise ValueError("Monthly category file needs Year + Month or Date columns.")

    category_cols = []
    for col in df.columns:
        low = str(col).lower()
        if col in ["Date", year_col, month_col, date_col] or "total" in low:
            continue
        if "electric" in low:
            category_cols.append(col)

    if not category_cols:
        raise ValueError("No electric category columns were found.")

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
        category_monthly.groupby(["Date", "Category"], as_index=False)["EV_Registrations"].sum()
    )

    if total_col and total_col in df.columns:
        df[total_col] = clean_num(df[total_col]).fillna(0).clip(lower=0)
        monthly_total = df.groupby("Date", as_index=False)[total_col].sum().rename(
            columns={total_col: "EV_Registrations"}
        )
    else:
        monthly_total = category_monthly.groupby("Date", as_index=False)["EV_Registrations"].sum()

    for frame in [category_monthly, monthly_total]:
        frame["Year"] = frame["Date"].dt.year
        frame["Month"] = frame["Date"].dt.month
        frame["Month_Name"] = frame["Date"].dt.strftime("%b")

    monthly_raw = category_monthly.copy()
    return monthly_raw, monthly_total, category_monthly


def load_state_monthly(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = clean_column_names(pd.read_csv(path))

    date_col = find_col(df, ["date"])
    state_col = find_col(df, ["state_name", "state"])
    fuel_col = find_col(df, ["fuel_type", "fuel"])
    registration_col = find_col(df, ["registrations", "registration"])

    required = {
        "date": date_col,
        "state": state_col,
        "fuel": fuel_col,
        "registrations": registration_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"State monthly dataset is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["State"] = df[state_col].apply(normalize_state_name)
    df["Fuel_Type"] = df[fuel_col].apply(normalize_fuel)
    df["Registrations"] = clean_num(df[registration_col]).fillna(0).clip(lower=0)
    df["Is_EV"] = df["Fuel_Type"].apply(is_ev_fuel_value)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    df = df.dropna(subset=["Date"])

    monthly_total = (
        df.groupby(["State", "Date", "Year", "Month"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "Total_Registrations"})
    )

    monthly_ev = (
        df[df["Is_EV"]]
        .groupby(["State", "Date", "Year", "Month"], as_index=False)["Registrations"]
        .sum()
        .rename(columns={"Registrations": "EV_Registrations"})
    )

    return df, monthly_ev, monthly_total


def read_old_statewise_reference(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    header_row_idx = None

    for i in range(min(8, len(raw))):
        row_values = raw.iloc[i].fillna("").astype(str).str.lower().tolist()
        if any("state" in value for value in row_values):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Could not find state header row in old statewise dataset.")

    fuel_row_idx = max(header_row_idx - 1, 0)
    fuel_headers = raw.iloc[fuel_row_idx].ffill().fillna("").astype(str).str.strip()
    year_headers = raw.iloc[header_row_idx].fillna("").astype(str).str.strip()

    final_cols = []
    for fuel, year in zip(fuel_headers, year_headers):
        fuel = "" if fuel.lower() in {"nan", "none"} else fuel
        year = "" if year.lower() in {"nan", "none"} else year

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
    df = remove_total_rows(df.dropna(how="all").reset_index(drop=True))

    state_col = find_col(df, ["State"])
    electric_cols = [col for col in df.columns if "electric" in str(col).lower() and "bov" in str(col).lower()]

    rows = []
    for col in electric_cols:
        year_match = re.search(r"(20\d{2})", str(col))
        if not year_match:
            continue
        temp = df[[state_col, col]].copy()
        temp["State"] = temp[state_col].apply(normalize_state_name)
        temp["Year"] = int(year_match.group(1))
        temp["Reference_EV_Registrations"] = clean_num(temp[col]).fillna(0)
        rows.append(temp[["State", "Year", "Reference_EV_Registrations"]])

    if not rows:
        raise ValueError("No ELECTRIC(BOV) year columns were found in old statewise dataset.")

    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["State", "Year"], as_index=False)["Reference_EV_Registrations"]
        .sum()
    )


def compare_state_sources(monthly_ev: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    annual_new = monthly_ev.groupby(["State", "Year"], as_index=False)["EV_Registrations"].sum()
    annual_new = annual_new.rename(columns={"EV_Registrations": "Monthly_File_EV_Registrations"})

    comparison = reference.merge(annual_new, on=["State", "Year"], how="outer")
    comparison[["Reference_EV_Registrations", "Monthly_File_EV_Registrations"]] = comparison[
        ["Reference_EV_Registrations", "Monthly_File_EV_Registrations"]
    ].fillna(0)
    comparison["Difference"] = (
        comparison["Monthly_File_EV_Registrations"] - comparison["Reference_EV_Registrations"]
    )
    comparison["Difference_Percent_vs_Reference"] = comparison.apply(
        lambda row: safe_pct_change(row["Monthly_File_EV_Registrations"], row["Reference_EV_Registrations"]),
        axis=1,
    )
    comparison["Needs_Review"] = (
        comparison["Difference"].abs().gt(1000)
        & comparison["Difference_Percent_vs_Reference"].abs().gt(25)
        & comparison["Reference_EV_Registrations"].gt(0)
    )

    return comparison.sort_values(["Needs_Review", "Difference"], ascending=[False, False]).reset_index(drop=True)


def apply_reference_adjustment(
    monthly_ev: pd.DataFrame,
    reference: pd.DataFrame,
    threshold_abs: float = 1000,
    threshold_pct: float = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_new = monthly_ev.groupby(["State", "Year"], as_index=False)["EV_Registrations"].sum()
    annual_new = annual_new.rename(columns={"EV_Registrations": "Monthly_File_EV_Registrations"})
    annual = reference.merge(annual_new, on=["State", "Year"], how="outer").fillna(0)

    annual["Difference"] = annual["Monthly_File_EV_Registrations"] - annual["Reference_EV_Registrations"]
    annual["Difference_Percent_vs_Reference"] = annual.apply(
        lambda row: safe_pct_change(row["Monthly_File_EV_Registrations"], row["Reference_EV_Registrations"]),
        axis=1,
    )
    annual["Use_Reference_Adjustment"] = (
        annual["Reference_EV_Registrations"].gt(0)
        & annual["Monthly_File_EV_Registrations"].gt(0)
        & annual["Difference"].abs().gt(threshold_abs)
        & annual["Difference_Percent_vs_Reference"].abs().gt(threshold_pct)
    )
    annual["Adjustment_Factor"] = np.where(
        annual["Use_Reference_Adjustment"],
        annual["Reference_EV_Registrations"] / annual["Monthly_File_EV_Registrations"].replace(0, np.nan),
        1.0,
    )
    annual["Adjustment_Factor"] = annual["Adjustment_Factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    adjusted = monthly_ev.merge(
        annual[["State", "Year", "Adjustment_Factor", "Use_Reference_Adjustment"]],
        on=["State", "Year"],
        how="left",
    )
    adjusted["Adjustment_Factor"] = adjusted["Adjustment_Factor"].fillna(1.0)
    adjusted["Use_Reference_Adjustment"] = adjusted["Use_Reference_Adjustment"].fillna(False)
    adjusted["EV_Registrations_Original"] = adjusted["EV_Registrations"]
    adjusted["EV_Registrations_Adjusted"] = adjusted["EV_Registrations"] * adjusted["Adjustment_Factor"]

    return adjusted, annual


def latest_complete_year(monthly_df: pd.DataFrame) -> int:
    months = monthly_df[["Date"]].drop_duplicates().copy()
    months["Year"] = months["Date"].dt.year
    months["Month"] = months["Date"].dt.month
    counts = months.groupby("Year")["Month"].nunique()
    complete = counts[counts >= 12]

    if not complete.empty:
        return int(complete.index.max())

    return int(months["Year"].max())


def forecast_monthly_series(
    actual_df: pd.DataFrame,
    value_col: str = "EV_Registrations",
    end_year: int = FORECAST_END_YEAR,
    degree: int = 2,
) -> tuple[pd.DataFrame, float, pd.Timestamp, str]:
    actual = actual_df[["Date", value_col]].copy()
    actual = actual.dropna(subset=["Date"])
    actual[value_col] = clean_num(actual[value_col]).fillna(0).clip(lower=0)
    actual = actual.groupby("Date", as_index=False)[value_col].sum().sort_values("Date")

    if actual.empty:
        raise ValueError("No monthly records are available for forecasting.")

    positive_dates = actual.loc[actual[value_col] > 0, "Date"]
    start_date = positive_dates.min() if not positive_dates.empty else actual["Date"].min()
    actual = actual[actual["Date"] >= start_date]

    full_dates = pd.date_range(actual["Date"].min(), actual["Date"].max(), freq="MS")
    actual = pd.DataFrame({"Date": full_dates}).merge(actual, on="Date", how="left")
    actual[value_col] = actual[value_col].fillna(0)

    last_date = actual["Date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), f"{end_year}-12-01", freq="MS")
    all_dates = pd.concat([actual["Date"], pd.Series(future_dates)], ignore_index=True)

    nonzero_months = int((actual[value_col] > 0).sum())
    can_model = len(actual) >= 6 and nonzero_months >= 4

    if can_model:
        fit_degree = min(degree, 2 if len(actual) >= 24 else 1)
        x_actual = np.arange(len(actual)).reshape(-1, 1)
        x_all = np.arange(len(all_dates)).reshape(-1, 1)

        model = make_pipeline(PolynomialFeatures(fit_degree), Ridge(alpha=1.0))
        model.fit(x_actual, actual[value_col])

        fitted = model.predict(x_actual)
        predicted = np.clip(model.predict(x_all), 0, None)
        r2 = r2_score(actual[value_col], fitted) if len(actual) > 1 else np.nan
        residual_std = float(np.std(actual[value_col] - fitted))
        method = "Polynomial Ridge Regression"
    else:
        annual = actual.copy()
        annual["Year"] = annual["Date"].dt.year
        annual = annual.groupby("Year", as_index=False)[value_col].sum()
        positive = annual[annual[value_col] > 0]

        if len(positive) >= 2:
            annual_growth = cagr(
                positive.iloc[0][value_col],
                positive.iloc[-1][value_col],
                int(positive.iloc[-1]["Year"] - positive.iloc[0]["Year"]),
            )
        else:
            annual_growth = 5.0

        if pd.isna(annual_growth):
            annual_growth = 5.0

        annual_growth = float(np.clip(annual_growth / 100, -0.10, 0.35))
        monthly_growth = (1 + annual_growth) ** (1 / 12) - 1
        base = float(actual[value_col].tail(min(6, len(actual))).mean())

        future_values = [max(base * ((1 + monthly_growth) ** (i + 1)), 0) for i in range(len(future_dates))]
        predicted = np.array(list(actual[value_col]) + future_values)
        fitted = np.array(actual[value_col])
        r2 = np.nan
        residual_std = float(np.std(actual[value_col].tail(min(12, len(actual))))) if len(actual) > 1 else 0
        method = "Growth Fallback"

    ci = 1.96 * residual_std

    forecast = pd.DataFrame(
        {
            "Date": all_dates,
            "Actual": list(actual[value_col]) + [np.nan] * len(future_dates),
            "Predicted": predicted,
            "CI_Lower": np.clip(predicted - ci, 0, None),
            "CI_Upper": predicted + ci,
        }
    )
    forecast["Year"] = forecast["Date"].dt.year
    forecast["Month"] = forecast["Date"].dt.month

    return forecast, r2, last_date, method


def forecast_categories(category_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    metrics = []

    for category, group in category_monthly.groupby("Category"):
        forecast, r2, last_date, method = forecast_monthly_series(
            group.rename(columns={"EV_Registrations": "Value"}),
            value_col="Value",
        )
        forecast["Category"] = category
        frames.append(forecast)

        metrics.append(
            {
                "Category": category,
                "Model_Fit_Score": r2,
                "Last_Actual_Month": last_date.strftime("%b %Y"),
                "Forecast_Method": method,
            }
        )

    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)


def forecast_states_to_2030(state_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    metrics = []

    for state, group in state_monthly.groupby("State"):
        value_col = "EV_Registrations_Adjusted" if "EV_Registrations_Adjusted" in group.columns else "EV_Registrations"
        forecast, r2, last_date, method = forecast_monthly_series(group, value_col=value_col)

        nonzero_months = int((group[value_col] > 0).sum())
        if nonzero_months >= 36 and (pd.isna(r2) or r2 >= 0.50):
            confidence = "High"
        elif nonzero_months >= 18:
            confidence = "Medium"
        else:
            confidence = "Low"

        forecast["State"] = state
        frames.append(forecast)
        metrics.append(
            {
                "State": state,
                "Model_Fit_Score": r2,
                "Nonzero_EV_Months": nonzero_months,
                "Last_Actual_Month": last_date.strftime("%b %Y"),
                "Forecast_Method": method,
                "Forecast_Confidence": confidence,
            }
        )

    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)
