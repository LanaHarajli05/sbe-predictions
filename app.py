import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="SBE Enrollments – EDA & Forecast", layout="wide")

# ---------- helpers ----------
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

    # Normalize COR for grouping
    df[cor_col] = df[cor_col].astype(str).str.strip()
    df.loc[df[cor_col].eq("") | df[cor_col].eq("nan"), cor_col] = "Unknown"
    return df, date_col, cor_col

def to_monthly_counts(df, date_col):
    s = (
        df
        .assign(_ym=df[date_col].dt.to_period("M").dt.to_timestamp())
        .groupby("_ym")
        .size()
        .sort_index()
    )
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
    # Filter window
    m = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    d = df.loc[m].copy()
    if d.empty:
        return None

    d["ym"] = d[date_col].dt.to_period("M").dt.to_timestamp()

    # SAFER than pivot_table(values=.., aggfunc="count"): use crosstab to count rows
    pivot = pd.crosstab(d["ym"], d[cor_col]).sort_index()

    # Keep top N and bucket the rest
    totals = pivot.sum(0).sort_values(ascending=False)
    keep = totals.head(top_n).index.tolist()
    others = [c for c in pivot.columns if c not in keep]
    if others:
        pivot["Others"] = pivot[others].sum(1)
        pivot = pivot.drop(columns=others)

    # Ensure monthly frequency
    pivot = pivot.asfreq("MS").fillna(0).astype(int)
    return pivot

def forecast_countries(pivot, horizon_months=12):
    if pivot is None or pivot.empty:
        return None, None

    # Compute recent average shares (last 12 months or all available if <12)
    window = min(12, len(pivot))
    recent = pivot.tail(window)
    totals = recent.sum(1).replace(0, np.nan)
    shares = (recent.div(totals, axis=0)).fillna(0.0)
    mean_share = shares.mean(0)  # average share per country

    # Forecast total series
    total_series = pivot.sum(1).asfreq("MS").fillna(0)
    y_fc_total = forecast_series(total_series, periods=max(1, horizon_months))

    # Allocate total to countries by mean_share
    fc_df = pd.DataFrame(
        {c: np.round(mean_share.get(c, 0.0) * y_fc_total.values, 0).astype(int)
         for c in mean_share.index},
        index=y_fc_total.index
    )

    # Reconcile rounding to match total per month (assign remainder to max-share)
    row_diff = y_fc_total - fc_df.sum(1)
    if len(fc_df.columns) > 0:
        max_c = mean_share.idxmax()
        for i, idx in enumerate(fc_df.index):
            diff = int(row_diff.iloc[i])
            if diff != 0:
                fc_df.loc[idx, max_c] = int(max(0, fc_df.loc[idx, max_c] + diff))

    return fc_df, y_fc_total

# ---------- UI ----------
st.title("SBE – Enrollments Monitoring & Forecasting")

st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
excel_path = st.sidebar.text_input("Or path (if file is in repo)", value="")
sheet = st.sidebar.selectbox("Sheet", ["All Enrolled(2)", "All Enrolled"], index=0)

# Choose source (prefer upload)
source = None
if uploaded is not None:
    source = uploaded
elif excel_path.strip():
    if os.path.exists(excel_path.strip()):
        source = excel_path.strip()
    else:
        st.warning(f"Path not found: `{excel_path.strip()}`. Upload the Excel on the left or enter a valid path.")
        st.stop()
else:
    st.info("Please upload the Excel on the left **or** enter a valid path from your repo (e.g., `data/sbe_enrolled.xlsx`).")
    st.stop()

# Load
try:
    df, date_col, cor_col = read_excel(source, sheet)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

# Build calendar limits from actual data
min_d = df[date_col].min().date()
max_d = df[date_col].max().date()

# Default to last 36 months (but allow full range via the picker)
default_start = max(min_d, (max_d - relativedelta(years=3)))

st.sidebar.header("Filters – Actuals & Forecast")
start_date = st.sidebar.date_input("Actuals FROM", value=default_start, min_value=min_d, max_value=max_d)
end_date   = st.sidebar.date_input("Actuals TO", value=max_d, min_value=start_date, max_value=max_d)

# Forecast UNTIL (up to +5 years ahead)
forecast_max = max_d + relativedelta(years=5)
forecast_until = st.sidebar.date_input("Forecast UNTIL", value=(max_d + relativedelta(years=1)), min_value=end_date, max_value=forecast_max)

top_n = st.sidebar.slider("Countries: Top N to show", min_value=3, max_value=15, value=8, step=1)

st.caption(f"Using date column: **{date_col}** | Country column: **{cor_col}**")

# ---- SECTION 1: Total actuals + forecast ----
st.subheader("Total Enrollments per Month — Actuals & Forecast")

# Actuals window
mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
df_win = df.loc[mask].copy()

monthly = to_monthly_counts(df_win, date_col)
last_actual = monthly.index.max() if len(monthly) else pd.to_datetime(end_date).replace(day=1)
forecast_until_ts = pd.to_datetime(forecast_until).to_period("M").to_timestamp()
months_ahead = max(0, (forecast_until_ts.year - last_actual.year)*12 + (forecast_until_ts.month - last_actual.month))
y_fc = forecast_series(monthly, periods=months_ahead) if months_ahead > 0 else pd.Series(dtype=int)

plot_series_with_forecast(monthly, y_fc, title="Total Enrollments – Actual & Forecast")

c1, c2, c3 = st.columns(3)
c1.metric("Total Actuals (selected window)", int(monthly.sum()) if len(monthly) else 0)
c2.metric("Last Actual Month", last_actual.strftime("%b %Y") if len(monthly) else "N/A")
c3.metric("Forecast Months", months_ahead)

st.download_button("Download monthly actuals (CSV)", data=monthly.rename("Enrollments").to_csv().encode("utf-8"),
                   file_name="monthly_actuals.csv", mime="text/csv")

if months_ahead > 0 and len(y_fc):
    st.download_button("Download monthly forecast (CSV)", data=y_fc.rename("Forecast").to_csv().encode("utf-8"),
                       file_name="monthly_forecast.csv", mime="text/csv")

st.divider()

# ---- SECTION 2: COR actuals + forecast ----
st.subheader("Countries of Residence by Month — Actuals & Forecast Allocation")

pivot = build_country_pivot(df, date_col, cor_col, start=start_date, end=end_date, top_n=top_n)

if pivot is None or pivot.empty:
    st.info("No data for the selected period.")
else:
    # Actuals stacked area
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot.area(ax=ax)
    ax.set_title(f"Country of Residence (Top {top_n} + Others) — Actuals")
    ax.set_xlabel("Month")
    ax.set_ylabel("Enrollments")
    st.pyplot(fig)

    # Also show actual totals per country (bar)
    actual_totals = pivot.sum(0).sort_values(ascending=False)
    st.markdown("**Actual totals by country (selected window)**")
    st.bar_chart(actual_totals)

    # Forecast allocation per month
    fc_countries, fc_total = forecast_countries(pivot, horizon_months=months_ahead if months_ahead > 0 else 12)
    if fc_countries is not None:
        st.markdown("**Forecasted Enrollments by Country (per month)**")
        st.dataframe(fc_countries)

        # Forecast stacked area
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fc_countries.plot.area(ax=ax2)
        ax2.set_title("Countries of Residence — Forecast Allocation")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Enrollments (allocated)")
        st.pyplot(fig2)

        # Horizon totals by country (sum over forecast months)
        horizon_totals = fc_countries.sum(0).sort_values(ascending=False)
        st.markdown("**Forecast horizon totals by country (sum over forecast months)**")
        st.bar_chart(horizon_totals)

        # Downloads
        combined_actual = pivot.copy()
        combined_actual["Total"] = combined_actual.sum(1)
        combined_fc = fc_countries.copy()
        combined_fc["Total"] = combined_fc.sum(1)
        out = pd.concat([combined_actual.assign(_type="actual"),
                         combined_fc.assign(_type="forecast")])
        st.download_button("Download COR actual + forecast (CSV)",
                           data=out.to_csv().encode("utf-8"),
                           file_name="cor_actuals_forecast.csv",
                           mime="text/csv")

st.info("Use the calendars to set **Actuals FROM**, **Actuals TO**, and **Forecast UNTIL**. The country forecast splits the total forecast using recent average country shares (last 12 months when available).")
st.caption("Models: SARIMAX(1,1,1)(1,1,1,12) with Holt-Winters fallback. Country forecasts allocated by recent average shares for stability with limited per-country history.")
