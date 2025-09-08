import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="SBE Enrollments – EDA & Forecast", layout="wide")

def find_date_col(df):
    for c in ["Created Date", "Created date", "Registration Date", "Enroll Date", "Enrollment Date"]:
        if c in df.columns:
            return c
    return None

def find_cor_col(df):
    for c in ["COR", "Country of Residence", "Country of residence", "Country"]:
        if c in df.columns:
            return c
    return None

def read_excel(file_or_path, sheet_name):
    # Accepts either an uploaded file buffer or a filesystem path
    try:
        df = pd.read_excel(file_or_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"Could not read Excel (sheet: {sheet_name}). Details: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = find_date_col(df)
    if not date_col:
        raise RuntimeError("No enrollment date-like column found. Expected one of: Created Date / Registration Date / Enroll Date / Enrollment Date")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    cor_col = find_cor_col(df)
    if not cor_col:
        df["COR"] = np.nan
        cor_col = "COR"
    return df, date_col, cor_col

def to_monthly_counts(df, date_col):
    s = (df
         .set_index(df[date_col].dt.to_period("M"))
         .groupby(level=0)
         .size()
         .sort_index())
    s.index = s.index.to_timestamp()
    s = s.asfreq("MS").fillna(0).astype(int)
    return s

def forecast_series(y, periods=12):
    # Robust monthly forecast with safe fallbacks
    if len(y.dropna()) < 6:
        last = float(y.iloc[-1]) if len(y) else 0.0
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
        return pd.Series([last]*periods, index=idx)
    try:
        m = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        fc = res.forecast(steps=periods)
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
        return pd.Series(np.maximum(0, np.round(fc, 0)), index=idx).astype(int)
    except Exception:
        try:
            hw = ExponentialSmoothing(y, trend="add", seasonal="add",
                                      seasonal_periods=12, initialization_method="estimated")
            res = hw.fit(optimized=True)
            fc = res.forecast(periods)
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            return pd.Series(np.maximum(0, np.round(fc, 0)), index=idx).astype(int)
        except Exception:
            last = float(y.iloc[-1]) if len(y) else 0.0
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            return pd.Series([last]*periods, index=idx)

def plot_series_with_forecast(y, y_fc, title="Total Enrollments – Actual & Forecast"):
    fig, ax = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax, label="Actual")
    if y_fc is not None and len(y_fc):
        y_fc.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Enrollments")
    ax.legend()
    st.pyplot(fig)

def build_country_pivot(df, date_col, cor_col, start, end, top_n=10):
    m = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    d = df.loc[m].copy()
    if d.empty:
        return None
    d["ym"] = d[date_col].dt.to_period("M").dt.to_timestamp()
    pivot = (d.pivot_table(index="ym", columns=cor_col, values=cor_col, aggfunc="count")
               .fillna(0)
               .astype(int))
    # Keep top N countries, bucket the rest into "Others"
    totals = pivot.sum(0).sort_values(ascending=False)
    keep = totals.head(top_n).index.tolist()
    others = [c for c in pivot.columns if c not in keep]
    if others:
        pivot["Others"] = pivot[others].sum(1)
        pivot = pivot.drop(columns=others)
    return pivot

def forecast_countries(pivot, horizon_months=12):
    if pivot is None or pivot.empty:
        return None, None
    window = min(12, len(pivot))
    recent = pivot.tail(window)
    totals = recent.sum(1).replace(0, np.nan)
    shares = (recent.div(totals, axis=0)).fillna(0.0)
    mean_share = shares.mean(0)  # average share per country (robust)
    # Forecast total
    total_series = pivot.sum(1).asfreq("MS").fillna(0)
    y_fc_total = forecast_series(total_series, periods=horizon_months if horizon_months > 0 else 12)
    # Allocate total forecast by average shares
    fc_df = pd.DataFrame({c: np.round(mean_share.get(c, 0.0)*y_fc_total.values, 0).astype(int)
                          for c in mean_share.index}, index=y_fc_total.index)
    # Reconcile row sums to total (assign remainder to max-share)
    row_diff = y_fc_total - fc_df.sum(1)
    if len(fc_df.columns) > 0:
        max_c = mean_share.idxmax()
        for i, idx in enumerate(fc_df.index):
            diff = int(row_diff.iloc[i])
            if diff != 0:
                fc_df.loc[idx, max_c] = int(max(0, fc_df.loc[idx, max_c] + diff))
    return fc_df, y_fc_total

# ========== UI ==========
st.title("SBE – Enrollments Monitoring & Forecasting")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
default_path = "data/SBE Enrolled Students Course Status 25 -7-2025.xlsx"  # for Streamlit Cloud repo use
excel_path = st.sidebar.text_input("Or path (leave if using upload)", value=default_path)
sheet = st.sidebar.selectbox("Sheet", ["All Enrolled(2)", "All Enrolled"], index=0)

# Load data (prefer uploaded file if present)
source = uploaded if uploaded is not None else excel_path
try:
    df, date_col, cor_col = read_excel(source, sheet)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

# Dynamic default date window
min_d = df[date_col].min().date()
max_d = max(df[date_col].max().date(), date.today())

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range (actuals window)",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_d, date_range

future_end_default = date.today() + relativedelta(years=1)
future_end_date = st.sidebar.date_input("Forecast end date", value=future_end_default)
top_n = st.sidebar.slider("Countries: Top N to show", min_value=3, max_value=15, value=8, step=1)

st.caption(f"Using date column: **{date_col}** | Country column: **{cor_col}**")

# Filtered frame
mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
df_win = df.loc[mask].copy()

# --- SECTION 1: Total actuals + forecast ---
st.subheader("Total Enrollments per Month — Actuals & Forecast")
monthly = to_monthly_counts(df_win, date_col)
last_actual = monthly.index.max() if len(monthly) else pd.to_datetime(end_date).replace(day=1)
future_end_ts = pd.to_datetime(future_end_date).to_period("M").to_timestamp()
months_ahead = max(0, (future_end_ts.year - last_actual.year)*12 + (future_end_ts.month - last_actual.month))
y_fc = forecast_series(monthly, periods=months_ahead) if months_ahead > 0 else pd.Series(dtype=int)
plot_series_with_forecast(monthly, y_fc, title="Total Enrollments – Actual & Forecast")

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Total Actuals (selected window)", int(monthly.sum()) if len(monthly) else 0)
c2.metric("Last Actual Month", last_actual.strftime("%b %Y") if len(monthly) else "N/A")
c3.metric("Forecast Months", months_ahead)

st.divider()

# --- SECTION 2: COR actuals + forecast allocation ---
st.subheader("Countries of Residence by Month — Actuals & Forecast Allocation")
pivot = build_country_pivot(df, date_col, cor_col, start_date, end_date, top_n=top_n)
if pivot is None:
    st.info("No data for the selected period.")
else:
    # Actuals: stacked area
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot.area(ax=ax)
    ax.set_title(f"Country of Residence (Top {top_n} + Others)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Enrollments")
    st.pyplot(fig)

    # Forecast allocation
    fc_countries, fc_total = forecast_countries(pivot, horizon_months=months_ahead if months_ahead > 0 else 12)
    if fc_countries is not None:
        st.markdown("**Forecasted Enrollments by Country (allocated from average recent shares)**")
        st.dataframe(fc_countries)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fc_countries.plot.area(ax=ax2)
        ax2.set_title("Countries of Residence — Forecast Allocation")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Enrollments (allocated)")
        st.pyplot(fig2)

        combined_actual = pivot.copy()
        combined_actual["Total"] = combined_actual.sum(1)
        combined_fc = fc_countries.copy()
        combined_fc["Total"] = combined_fc.sum(1)
        out = pd.concat([combined_actual.assign(_type="actual"),
                         combined_fc.assign(_type="forecast")])
        st.download_button("Download actual + forecast by country (CSV)",
                           data=out.to_csv().encode("utf-8"),
                           file_name="sbe_country_actuals_forecast.csv",
                           mime="text/csv")

st.info("Tip: Use the **Date range** to set actuals, and **Forecast end date** to extend predictions.")
st.caption("Models: SARIMAX(1,1,1)(1,1,1,12) with Holt-Winters fallback. Country forecasts allocated by recent average shares.")
