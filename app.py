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
    s = s.asfreq("MS").fillna(0.0)  # keep float for forecasting; no rounding here
    return s

# -------- forecasting --------
def forecast_sarimax_hw(y, periods):
    """Try SARIMAX then Holt-Winters; return float series (no rounding)."""
    if len(y.dropna()) < 6:
        return None
    try:
        m = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        fc = res.forecast(steps=periods)
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
        return pd.Series(np.maximum(0, fc.astype(float)), index=idx)
    except Exception:
        try:
            hw = ExponentialSmoothing(y, trend="add", seasonal="add",
                                      seasonal_periods=12, initialization_method="estimated")
            res = hw.fit(optimized=True)
            fc = res.forecast(periods)
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            return pd.Series(np.maximum(0, fc.astype(float)), index=idx)
        except Exception:
            return None

def seasonal_naive(y, periods):
    """Repeat last year's same-month values when available; else last value."""
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
    fc = []
    for i, ts in enumerate(idx, start=1):
        same_m_last_year = ts - pd.DateOffset(years=1)
        val = y.get(same_m_last_year, np.nan)
        if pd.isna(val):
            val = y.iloc[-1] if len(y) else 0.0
        fc.append(val)
    return pd.Series(np.maximum(0, np.array(fc, dtype=float)), index=idx)

def recent_average(y, periods, k=6):
    """Repeat the mean of the last k months (good for small counts)."""
    if len(y) == 0:
        base = 0.0
    else:
        base = float(y.tail(max(1, min(k, len(y)))).mean())
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
    return pd.Series(np.full(periods, max(0.0, base), dtype=float), index=idx)

def forecast_series(y, periods=12, method="Auto (SARIMAX → HW → Seasonal Naive → Recent Avg)", k_avg=6):
    """Return float forecasts; rounding happens only for tables."""
    if periods <= 0:
        return pd.Series(dtype=float)

    method = method.lower()
    if method.startswith("auto"):
        fc = forecast_sarimax_hw(y, periods)
        if fc is None:
            # try seasonal naive; if still flat zeros, use recent average
            fc = seasonal_naive(y, periods)
            if fc.sum() == 0:
                fc = recent_average(y, periods, k=k_avg)
        return fc

    if method.startswith("sarimax"):
        fc = forecast_sarimax_hw(y, periods)
        if fc is None:
            st.warning("SARIMAX/Holt-Winters failed on this series; falling back to Recent Average.")
            fc = recent_average(y, periods, k=k_avg)
        return fc

    if method.startswith("holt"):
        # try HW only; fallback to recent avg
        try:
            hw = ExponentialSmoothing(y, trend="add", seasonal="add",
                                      seasonal_periods=12, initialization_method="estimated")
            res = hw.fit(optimized=True)
            fc_vals = res.forecast(periods)
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            return pd.Series(np.maximum(0, fc_vals.astype(float)), index=idx)
        except Exception:
            return recent_average(y, periods, k=k_avg)

    if method.startswith("seasonal"):
        return seasonal_naive(y, periods)

    if method.startswith("recent"):
        return recent_average(y, periods, k=k_avg)

    return recent_average(y, periods, k=k_avg)

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
    # safer counting via crosstab
    pivot = pd.crosstab(d["ym"], d[cor_col]).sort_index()
    totals = pivot.sum(0).sort_values(ascending=False)
    keep = totals.head(top_n).index.tolist()
    others = [c for c in pivot.columns if c not in keep]
    if others:
        pivot["Others"] = pivot[others].sum(1)
        pivot = pivot.drop(columns=others)
    pivot = pivot.asfreq("MS").fillna(0.0)
    return pivot

def forecast_countries(pivot, total_forecast, share_window_months=12):
    """Allocate total forecasts to countries by recent mean shares (float frames)."""
    if pivot is None or pivot.empty or total_forecast is None or len(total_forecast) == 0:
        return None

    window = min(share_window_months, len(pivot))
    recent = pivot.tail(window)
    totals = recent.sum(1).replace(0, np.nan)
    shares = (recent.div(totals, axis=0)).fillna(0.0)
    mean_share = shares.mean(0)  # average share per country

    fc_df = pd.DataFrame(
        {c: mean_share.get(c, 0.0) * total_forecast.values for c in mean_share.index},
        index=total_forecast.index
    )

    # reconcile tiny numerical drift to match total
    row_diff = total_forecast - fc_df.sum(1)
    if len(fc_df.columns) > 0:
        max_c = mean_share.idxmax()
        for i, idx in enumerate(fc_df.index):
            fc_df.loc[idx, max_c] = max(0.0, fc_df.loc[idx, max_c] + float(row_diff.iloc[i]))
    return fc_df

# ---------- UI ----------
st.title("SBE – Enrollments Monitoring & Forecasting")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
excel_path = st.sidebar.text_input("Or path (if file is in repo)", value="")
sheet = st.sidebar.selectbox("Sheet", ["All Enrolled(2)", "All Enrolled"], index=0)

source = None
if uploaded is not None:
    source = uploaded
elif excel_path.strip():
    if os.path.exists(excel_path.strip()):
        source = excel_path.strip()
    else:
        st.warning(f"Path not found: `{excel_path.strip()}`. Upload or enter a valid path.")
        st.stop()
else:
    st.info("Upload the Excel or enter a valid repo path (e.g., `data/sbe_enrolled.xlsx`).")
    st.stop()

try:
    df, date_col, cor_col = read_excel(source, sheet)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

# Calendar limits
min_d = df[date_col].min().date()
max_d = df[date_col].max().date()
default_start = max(min_d, (max_d - relativedelta(years=3)))

st.sidebar.header("Filters – Actuals & Forecast")
start_date = st.sidebar.date_input("Actuals FROM", value=default_start, min_value=min_d, max_value=max_d)
end_date   = st.sidebar.date_input("Actuals TO", value=max_d,       min_value=start_date, max_value=max_d)
forecast_max = max_d + relativedelta(years=5)
forecast_until = st.sidebar.date_input("Forecast UNTIL", value=(max_d + relativedelta(years=1)), min_value=end_date, max_value=forecast_max)

st.sidebar.header("Forecast options")
method = st.sidebar.selectbox(
    "Totals forecast method",
    ["Auto (SARIMAX → HW → Seasonal Naive → Recent Avg)",
     "SARIMAX/Holt-Winters",
     "Seasonal Naive (last year same month)",
     "Recent Average (last K months)"],
    index=0
)
k_avg = st.sidebar.slider("K for Recent Average", 3, 12, 6)

top_n = st.sidebar.slider("Countries: Top N to show", 3, 15, 8, 1)

st.caption(f"Using date column: **{date_col}** | Country column: **{cor_col}**")

# ---- SECTION 1: Total actuals + forecast ----
st.subheader("Total Enrollments per Month — Actuals & Forecast")

mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
df_win = df.loc[mask].copy()

monthly = to_monthly_counts(df_win, date_col)
last_actual = monthly.index.max() if len(monthly) else pd.to_datetime(end_date).replace(day=1)
forecast_until_ts = pd.to_datetime(forecast_until).to_period("M").to_timestamp()
months_ahead = max(0, (forecast_until_ts.year - last_actual.year)*12 + (forecast_until_ts.month - last_actual.month))

y_fc = forecast_series(
    monthly,
    periods=months_ahead if months_ahead > 0 else 0,
    method=method,
    k_avg=k_avg
)

plot_series_with_forecast(monthly, y_fc, title="Total Enrollments – Actual & Forecast")

c1, c2, c3 = st.columns(3)
c1.metric("Total Actuals (selected window)", int(monthly.sum()) if len(monthly) else 0)
c2.metric("Last Actual Month", last_actual.strftime("%b %Y") if len(monthly) else "N/A")
c3.metric("Forecast Months", months_ahead)

# Totals table (actuals + forecast) with Year/Month
totals_table = monthly.rename("Actual").to_frame()
if y_fc is not None and len(y_fc):
    totals_table = totals_table.join(y_fc.rename("Forecast"), how="outer")
totals_table.index.name = "Month"
totals_table = totals_table.reset_index()
totals_table["Year"] = totals_table["Month"].dt.year
totals_table["MonthName"] = totals_table["Month"].dt.strftime("%b")
totals_table_display = totals_table[["Year", "MonthName", "Actual", "Forecast"]].fillna(0)
st.markdown("**Totals by Month (Actuals & Forecast)**")
st.dataframe(totals_table_display, hide_index=True)

st.download_button("Download monthly totals (CSV)",
                   data=totals_table_display.to_csv(index=False).encode("utf-8"),
                   file_name="monthly_totals_actuals_forecast.csv",
                   mime="text/csv")

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

    # Actual totals by country (bar)
    actual_totals = pivot.sum(0).sort_values(ascending=False)
    st.markdown("**Actual totals by country (selected window)**")
    st.bar_chart(actual_totals)

    # Allocate country forecast using the TOTAL forecast series (y_fc)
    if y_fc is not None and len(y_fc):
        fc_countries = forecast_countries(pivot, y_fc, share_window_months=12)

        if fc_countries is not None and not fc_countries.empty:
            # Display per-month table (rounded just for readability)
            table_countries = fc_countries.copy()
            table_countries.index.name = "Month"
            table_countries = table_countries.reset_index()
            table_countries["Year"] = table_countries["Month"].dt.year
            table_countries["MonthName"] = table_countries["Month"].dt.strftime("%b")
            cols = ["Year", "MonthName"] + [c for c in fc_countries.columns]
            st.markdown("**Forecasted Enrollments by Country (per month)**")
            st.dataframe(table_countries[cols].round(2), hide_index=True)

            # Forecast stacked area
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fc_countries.plot.area(ax=ax2)
            ax2.set_title("Countries of Residence — Forecast Allocation")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Enrollments (allocated)")
            st.pyplot(fig2)

            # Horizon totals by country (sum of forecast months)
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
        else:
            st.info("No forecast produced for countries (check totals forecast method or horizon).")
    else:
        st.info("Set a **Forecast UNTIL** later than the last actual month to generate country forecasts.")

st.info(
    "If the orange line looks flat, switch method to **Seasonal Naive** (uses last year’s pattern) "
    "or **Recent Average** (mean of last K months). For small counts, these are more realistic."
)
