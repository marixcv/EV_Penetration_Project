# EV Penetration Analysis India

Interactive Streamlit dashboard for analyzing EV adoption trends, penetration rates, and forecasting EV registrations in India through 2030.

## Project Overview

This project analyzes Electric Vehicle (EV) adoption trends across India using vehicle registration datasets. The analysis explores EV growth, state-wise adoption patterns, category-wise penetration, monthly trends, and forecasts EV registrations up to 2030 through an interactive Streamlit dashboard.

The objective is to provide data-driven insights into India's transition toward electric mobility and identify regions and vehicle segments leading EV adoption.

---

## Objectives

- Analyze historical EV registration trends in India.
- Measure EV penetration across vehicle categories.
- Compare EV adoption across states and union territories.
- Study monthly and yearly growth patterns.
- Forecast EV registrations through 2030.
- Build an interactive dashboard for exploration and decision-making.

---

## Datasets Used

### 1. State Fuel Monthly Dataset
**File:** `Dataset_state_Fuel_Month_2019-2024.csv`

Contains:
- State/UT
- Fuel Type
- Registration Count
- Monthly Registration Data

### 2. Wheel Fuel Yearly Dataset
**File:** `Dataset_Wheel_Fuel_Yearly_2011-2025.xlsx`

Contains:
- Vehicle Categories
- Fuel Types
- Annual Registration Data

### 3. Monthly EV Category Dataset
**File:** `Ev_monthlywith_category.csv`

Contains:
- Monthly EV Registrations
- Vehicle Categories
- Registration Volumes

### 4. EV Penetration Dataset
**File:** `Ev_penetration%wise.csv`

Contains:
- EV Penetration Metrics
- Category-wise Adoption Information

---

## Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-Learn
- OpenPyXL
- Jupyter Notebook

---

## Project Structure

```text
EV_PENETRATION/
│
├── data/
│   ├── Dataset_state_Fuel_Month_2019-2024.csv
│   ├── Dataset_Wheel_Fuel_Yearly_2011-2025.xlsx
│   ├── Ev_monthlywith_category.csv
│   └── Ev_penetration%wise.csv
│
├── notebooks/
│   └── Main_EDA.ipynb
│
├── outputs/
│   ├── cleaned/
│   ├── state_forecast_2030_summary.csv
│   └── state_forecast_monthly_2030.csv
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── ev_data.py
│
├── README.md
└── requirements.txt
```

---

## Methodology

### Data Preparation
- Data cleaning and preprocessing
- Missing value handling
- State and category standardization
- EV fuel type identification
- Aggregation of monthly and yearly registrations

### Exploratory Data Analysis
- National EV growth analysis
- State-wise EV registration analysis
- Category-wise EV adoption analysis
- Monthly trend analysis
- EV penetration assessment

### Forecasting
- Polynomial Regression
- Ridge Regression
- Monthly EV registration forecasting
- State-wise EV forecasting
- Category-wise EV forecasting
- Confidence interval estimation

Forecasts are generated through the year 2030.

---

## Dashboard Features

### National EV Growth Overview
- Annual EV registration trends
- Growth rate analysis
- CAGR calculation
- Key performance indicators

### State-wise Analysis
- Top EV-adopting states
- Historical state trends
- State ranking dashboard

### EV Penetration Analysis
- National EV penetration rate
- Category-wise penetration percentage
- Penetration trend visualization

### Vehicle Category Analysis
- Two-Wheeler EV trends
- Three-Wheeler EV trends
- Electric Car adoption
- Category growth comparison

### Monthly Trends
- Monthly EV registration patterns
- Year-over-Year growth analysis
- Seasonality heatmaps

### Forecasting to 2030
- National EV forecast
- Category-wise forecast
- State-wise forecast
- Model diagnostics
- Forecast confidence assessment

---

## Key Insights

- EV registrations have grown significantly over recent years.
- EV adoption varies considerably across Indian states.
- Two-wheelers account for a major share of EV registrations.
- EV penetration continues to increase across vehicle categories.
- Forecast results indicate sustained growth in EV adoption through 2030.

---

## Running the Project

### Clone the Repository

```bash
git clone https://github.com/marixcv/EV_Penetration_Project.git
cd EV_PENETRATION
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit Dashboard

```bash
streamlit run src/app.py
```

---

## Output Files

The project generates forecasting outputs including:

- State-wise EV forecast summaries
- Monthly forecast datasets
- Cleaned datasets for analysis
- Interactive dashboard visualizations

Generated files are stored in the `outputs/` directory.

---

## Future Improvements

- Integration with Power BI dashboards
- Advanced machine learning forecasting models
- Manufacturer-wise market share analysis
- Charging infrastructure analysis
- Real-time data integration
- Deployment on cloud platforms

---

## Author

Maria Shaikh

EV Penetration Analysis India – Data Analytics Project