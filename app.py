import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import load

st.set_page_config(page_title="SBE Enrollments – EDA & Forecast", layout="wide")

# =========================
# Data helpers
# =========================
def find_date_col(df: pd.DataFrame) -> str | None:
    for c in ["Created Date", "Created date", "Registration Date", "Enroll Date", "Enrollment Date"]:
        if c in df.columns:
            return c
    return None

def find_cor_col(df: pd.DataFrame) -> str | None:
    for c in ["COR", "Country of Residence", "Country of residence", "Country"]:
        if c in df.columns:
            return c
    return None

def read_excel(file_or_path, sheet_name: str):
    try:
        df = pd.read_excel(file_or_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"Could not read Excel (sheet: {sheet_name}). Details: {e}")
    df.columns = [str(c).strip() for c in df.columns]

    date_col = find_date_col(df)
    if not date_col:
        raise RuntimeError(
            "No enrollment date-like column found. "
            "Expected one of: Created Date / Registration Date / Enroll Date / Enrollment Date"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    cor_col = find_cor_col(df)
    if not cor_col:
        df["COR"] = np.nan
        cor_col = "COR"

    # Normalize COR values
    df[cor_col] = df[cor_col].astype(str).str.strip()
    df.loc[df[cor_col].eq("") | df[cor_col].eq("nan"), cor_col] = "Unknown"
    return df, date_col, cor_col

def to_monthly_counts(df: pd.DataFrame, date_col: str) -> pd.Series:
    s = (
        df.assign(_ym=df[date_col].dt.to_period("M").dt.to_timestamp())
          .groupby("_ym").size().sort_index()
          .asfreq("MS").fillna(0.0)  # keep float for forecasting
    )
    return s

# =========================
# Forecasting (statistical)
# =========================
def _sarimax_or_hw(y: pd.Series, periods: int) -> pd.Series | None:
    """Try SARIMAX then Holt-Winters; return FLOAT series (no rounding)."""
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

def _seasonal_naive(y: pd.Series, periods: int) -> pd.Series:
    """Repeat last year's same-month value; fallback to last value if missing."""
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
    fc = []
    for ts in idx:
        val = y.get(ts - pd.DateOffset(years=1), np.nan)
        if pd.isna(val):
            val = y.iloc[-1] if len(y) else 0.0
        fc.append(val)
    return pd.Series(np.maximum(0, np.array(fc, dtype=float)), index=idx)

def _recent_average(y: pd.Series, periods: int, k: int = 6) -> pd.Series:
    """Repeat mean of last k months (good when counts are small)."""
    base = float(y.tail(max(1, min(k, len(y)))).mean()) if len(y) else 0.0
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
    return pd.Series(np.full(periods, max(0.0, base), dtype=float), index=idx)

def forecast_series(y: pd.Series, periods: int,
                    method: str = "Auto (SARIMAX → HW → Seasonal Naive → Recent Avg)",
                    k_avg: int = 6) -> pd.Series:
    """Return FLOAT forecasts; charts use floats; rounding only in tables."""
    if periods <= 0:
        return pd.Series(dtype=float)

    m = method.lower()
    if m.startswith("auto"):
        fc = _sarimax_or_hw(y, periods)
        if fc is None:
            fc = _seasonal_naive(y, periods)
            if fc.sum() == 0:
                fc = _recent_average(y, periods, k=k_avg)
        return fc
    if m.startswith("sarimax"):
        fc = _sarimax_or_hw(y, periods)
        return fc if fc is not None else _recent_average(y, periods, k=k_avg)
    if m.startswith("holt"):
        try:
            hw = ExponentialSmoothing(y, trend="add", seasonal="add",
                                      seasonal_periods=12, initialization_method="estimated")
            res = hw.fit(optimized=True)
            vals = res.forecast(periods)
            idx  = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            return pd.Series(np.maximum(0, vals.astype(float)), index=idx)
        except Exception:
            return _recent_average(y, periods, k=k_avg)
    if m.startswith("seasonal"):
        return _seasonal_naive(y, periods)
    if m.startswith("recent"):
        return _recent_average(y, periods, k=k_avg)
    return _recent_average(y, periods, k=k_avg)

# =========================
# Forecasting (ML artifact)
# =========================
def load_ml_artifact(path: str):
    """Load a model artifact saved from Colab:
       dict(model=..., lags=12, roll_windows=(3,6,12), model_name='XGB'|...)"""
    art = load(path)
    return art["model"], art["lags"], art["roll_windows"], art.get("model_name", "")

def recursive_forecast_ml(y: pd.Series, model, horizon: int, lags: int, roll_windows: tuple[int, ...]) -> pd.Series:
    """Iteratively predict future months using the trained ML model and lag/rolling features."""
    history = y.copy()
    fc_values = []
    for _ in range(horizon):
        tmp = pd.DataFrame({"y": history})
        # lags
        for L in range(1, lags+1):
            tmp[f"lag_{L}"] = tmp["y"].shift(L)
        # rolling
        for w in roll_windows:
            tmp[f"roll_mean_{w}"] = tmp["y"].rolling(w).mean().shift(1)
            tmp[f"roll_sum_{w}"]  = tmp["y"].rolling(w).sum().shift(1)
        # cyclic month
        tmp["month"] = tmp.index.month
        from math import pi
        tmp["month_sin"] = np.sin(2*pi*tmp["month"]/12)
        tmp["month_cos"] = np.cos(2*pi*tmp["month"]/12)
        tmp.drop(columns=["month"], inplace=True)

        tmp = tmp.dropna()
        if tmp.empty:
            y_hat = float(history.iloc[-1]) if len(history) else 0.0
        else:
            X_next = tmp.drop(columns=["y"]).iloc[[-1]]
            y_hat = float(model.predict(X_next))
        y_hat = max(0.0, y_hat)

        next_ts = (history.index[-1] + pd.offsets.MonthBegin())
        history.loc[next_ts] = y_hat
        fc_values.append((next_ts, y_hat))

    return pd.Series({ts: val for ts, val in fc_values}).sort_index()

# =========================
# Plotting helpers
# =========================
def plot_series_with_forecast(y: pd.Series, y_fc: pd.Series | None, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(y):
        y.plot(ax=ax, label="Actual")
    if y_fc is not None and len(y_fc):
        y_fc.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Enrollments")
    ax.legend()
    st.pyplot(fig)

def build_country_pivot(df: pd.DataFrame, date_col: str, cor_col: str,
                        start, end, top_n: int = 10) -> pd.DataFrame | None:
    m = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    d = df.loc[m].copy()
    if d.empty:
        return None
    d["ym"] = d[date_col].dt.to_period("M").dt.to_timestamp()
    # safer counts
    pivot = pd.crosstab(d["ym"], d[cor_col]).sort_index()
    totals = pivot.sum(0).sort_values(ascending=False)
    keep = totals.head(top_n).index.tolist()
    others = [c for c in pivot.columns if c not in keep]
    if others:
        pivot["Others"] = pivot[others].sum(1)
        pivot = pivot.drop(columns=others)
    pivot = pivot.asfreq("MS").fillna(0.0)
    return pivot

def allocate_country_forecast(pivot: pd.DataFrame, total_forecast: pd.Series,
                              share_window_months: int = 12) -> pd.DataFrame | None:
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
    # Reconcile tiny diffs so rows equal the total
    row_diff = total_forecast - fc_df.sum(1)
    if len(fc_df.columns) > 0:
        max_c = mean_share.idxmax()
        for i, idx in enumerate(fc_df.index):
            fc_df.loc[idx, max_c] = max(0.0, fc_df.loc[idx, max_c] + float(row_diff.iloc[i]))
    return fc_df

# =========================
# UI
# =========================
st.title("SBE – Enrollments Monitoring & Forecasting")

# ---- Data input
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
        st.warning(f"Path not found: `{excel_path.strip()}`. Upload the Excel or enter a valid path.")
        st.stop()
else:
    st.info("Please upload the Excel on the left **or** enter a valid repo path (e.g., `data/sbe_enrolled.xlsx`).")
    st.stop()

# ---- Load data
try:
    df, date_col, cor_col = read_excel(source, sheet)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

min_d = df[date_col].min().date()
max_d = df[date_col].max().date()
default_start = max(min_d, (max_d - relativedelta(years=3)))

# ---- Calendars
st.sidebar.header("Filters – Actuals & Forecast")
start_date = st.sidebar.date_input("Actuals FROM", value=default_start, min_value=min_d, max_value=max_d)
end_date   = st.sidebar.date_input("Actuals TO",   value=max_d,       min_value=start_date, max_value=max_d)
forecast_max = max_d + relativedelta(years=5)
forecast_until = st.sidebar.date_input("Forecast UNTIL", value=(max_d + relativedelta(years=1)),
                                       min_value=end_date, max_value=forecast_max)

# ---- Forecast options
st.sidebar.header("Forecast options")
use_ml = st.sidebar.checkbox("Use trained ML model (from Colab)")
ml_model_path = st.sidebar.text_input("ML model path (.pkl)", value="models/sbe_best_model.pkl")

method = st.sidebar.selectbox(
    "If not using ML: totals forecast method",
    ["Auto (SARIMAX → HW → Seasonal Naive → Recent Avg)",
     "SARIMAX/Holt-Winters",
     "Seasonal Naive (last year same month)",
     "Recent Average (last K months)"],
    index=0
)
k_avg = st.sidebar.slider("K for Recent Average", 3, 12, 6)
top_n = st.sidebar.slider("Countries: Top N to show", 3, 15, 8, 1)

st.caption(f"Using date column: **{date_col}** | Country column: **{cor_col}**")

# =========================
# SECTION 1 — Totals
# =========================
st.subheader("Total Enrollments per Month — Actuals & Forecast")

mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
df_win = df.loc[mask].copy()

monthly = to_monthly_counts(df_win, date_col)
last_actual = monthly.index.max() if len(monthly) else pd.to_datetime(end_date).replace(day=1)
forecast_until_ts = pd.to_datetime(forecast_until).to_period("M").to_timestamp()
months_ahead = max(0, (forecast_until_ts.year - last_actual.year)*12 + (forecast_until_ts.month - last_actual.month))

y_fc = pd.Series(dtype=float)
used_source = "None"
if months_ahead > 0 and len(monthly):
    if use_ml and os.path.exists(ml_model_path):
        try:
            ml_model, LAGS, ROLLS, model_name = load_ml_artifact(ml_model_path)
            y_fc = recursive_forecast_ml(monthly, ml_model, months_ahead, LAGS, ROLLS)
            used_source = f"ML ({model_name})"
        except Exception as e:
            st.warning(f"Failed to load/use ML model: {e}. Falling back to statistical method.")
            y_fc = forecast_series(monthly, periods=months_ahead, method=method, k_avg=k_avg)
            used_source = "Statistical"
    else:
        y_fc = forecast_series(monthly, periods=months_ahead, method=method, k_avg=k_avg)
        used_source = "Statistical"

plot_series_with_forecast(monthly, y_fc, title="Total Enrollments – Actual & Forecast")

c1, c2, c3 = st.columns(3)
c1.metric("Total Actuals (selected window)", int(round(monthly.sum())) if len(monthly) else 0)
c2.metric("Last Actual Month", last_actual.strftime("%b %Y") if len(monthly) else "N/A")
c3.metric("Forecast Months", months_ahead)
if used_source != "None":
    st.caption(f"Forecast source: **{used_source}**")

# Totals table (Year / MonthName / Actual / Forecast)
totals_table = monthly.rename("Actual").to_frame()
if len(y_fc):
    totals_table = totals_table.join(y_fc.rename("Forecast"), how="outer")
totals_table.index.name = "Month"
totals_table = totals_table.reset_index()
totals_table["Year"] = totals_table["Month"].dt.year
totals_table["MonthName"] = totals_table["Month"].dt.strftime("%b")
totals_table_display = totals_table[["Year", "MonthName", "Actual", "Forecast"]].fillna(0).round(2)
st.markdown("**Totals by Month (Actuals & Forecast)**")
st.dataframe(totals_table_display, hide_index=True)

st.download_button("Download monthly totals (CSV)",
                   data=totals_table_display.to_csv(index=False).encode("utf-8"),
                   file_name="monthly_totals_actuals_forecast.csv",
                   mime="text/csv")

st.divider()

# =========================
# SECTION 2 — COR actuals & forecast allocation
# =========================
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

    # Allocate forecast by recent mean shares (if totals forecast exists)
    if len(y_fc):
        fc_countries = allocate_country_forecast(pivot, y_fc, share_window_months=12)
        if fc_countries is not None and not fc_countries.empty:
            # Per-month table (rounded for readability)
            table_countries = fc_countries.copy()
            table_countries.index.name = "Month"
            table_countries = table_countries.reset_index()
            table_countries["Year"] = table_countries["Month"].dt.year
            table_countries["MonthName"] = table_countries["Month"].dt.strftime("%b")
            cols = ["Year", "MonthName"] + [c for c in fc_countries.columns]
            st.markdown("**Forecasted Enrollments by Country (per month)**")
            st.dataframe(table_countries[cols].round(2), hide_index=True)

            # Stacked area – forecast allocation
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

            # Downloads: combined actual + forecast by country (with monthly totals)
            combined_actual = pivot.copy()
            combined_actual["Total"] = combined_actual.sum(1)
            combined_fc = fc_countries.copy()
            combined_fc["Total"] = combined_fc.sum(1)
            out = pd.concat([
                combined_actual.assign(_type="actual"),
                combined_fc.assign(_type="forecast")
            ])
            st.download_button("Download COR actual + forecast (CSV)",
                               data=out.to_csv().encode("utf-8"),
                               file_name="cor_actuals_forecast.csv",
                               mime="text/csv")
        else:
            st.info("No country forecast produced (check totals forecast method/horizon).")
    else:
        st.info("Set **Forecast UNTIL** later than the last actual month to generate country forecasts.")

st.info(
    "Tips: For small monthly counts, try **Seasonal Naive** or **Recent Average**. "
    "To use your Colab-trained model (Poisson / RF / XGB), upload `models/sbe_best_model.pkl` "
    "to your repo and enable **Use trained ML model** in the sidebar."
)
