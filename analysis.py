
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import curve_fit

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

DATA_PATH = "project/data/mumbai_power_raw.csv"
FIG_DIR = "project/figures/"
OUT_DIR = "project/data/"


# 1. LOAD & CLEAN

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

MISSING_CODE = -999.0
cols_of_interest = ["WS2M", "WS50M", "WD2M", "WD50M", "T2M", "T2M_MAX", "T2M_MIN",
                     "RH2M", "PS", "PRECTOTCORR"]
for c in cols_of_interest:
    df[c] = df[c].replace(MISSING_CODE, np.nan)

n_missing_before = df[["WS50M"]].isna().sum().sum()
# Linear interpolation for the (rare) missing daily values, then forward/back fill edges
df[cols_of_interest] = df[cols_of_interest].interpolate(method="linear", limit_direction="both")
n_missing_after = df[["WS50M"]].isna().sum().sum()

df = df.set_index("Date")

print(f"Total daily records           : {len(df)}")
print(f"Date range                    : {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing WS50M before/after    : {n_missing_before} / {n_missing_after}")


# 2. WIND-SHEAR EXTRAPOLATION TO 10 m (WS10M estimate)
#    Standard power-law wind profile: WS_h2 = WS_h1 * (h2/h1)^alpha
#    alpha estimated empirically from the two measured heights (2 m and 50 m)
#    then used to project a 10 m series for turbine-hub comparison.

h1, h2 = 2.0, 50.0
ratio = (df["WS50M"] / df["WS2M"]).replace([np.inf, -np.inf], np.nan)
alpha_series = np.log(ratio) / np.log(h2 / h1)
alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
alpha_hat = alpha_series.median()
print(f"Estimated Hellmann wind-shear exponent (alpha, median) : {alpha_hat:.3f}")

df["WS10M_est"] = df["WS2M"] * (10.0 / h1) ** alpha_hat


# 3. DESCRIPTIVE STATISTICS & WEIBULL RESOURCE ASSESSMENT (at 50 m)

ws = df["WS50M"].dropna()

desc = ws.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
desc.to_csv(OUT_DIR + "ws50m_descriptive_stats.csv")

# Weibull (2-parameter) fit via MLE
shape_k, loc_w, scale_c = stats.weibull_min.fit(ws.values, floc=0)
print(f"Weibull fit (WS50M): k={shape_k:.3f}, c={scale_c:.3f} m/s")

AIR_DENSITY = 1.225  # kg/m^3, sea-level standard
mean_power_density = 0.5 * AIR_DENSITY * (ws ** 3).mean()  # W/m^2 (direct, from data)

# Theoretical mean wind speed & power density from Weibull parameters (closed form)
from scipy.special import gamma as gammafn
weibull_mean = scale_c * gammafn(1 + 1 / shape_k)
weibull_power_density = 0.5 * AIR_DENSITY * scale_c ** 3 * gammafn(1 + 3 / shape_k)

with open(OUT_DIR + "resource_summary.txt", "w") as f:
    f.write(f"Mean WS50M (data)                : {ws.mean():.3f} m/s\n")
    f.write(f"Std WS50M (data)                 : {ws.std():.3f} m/s\n")
    f.write(f"Weibull shape k                  : {shape_k:.3f}\n")
    f.write(f"Weibull scale c                  : {scale_c:.3f} m/s\n")
    f.write(f"Weibull-implied mean speed       : {weibull_mean:.3f} m/s\n")
    f.write(f"Mean wind power density (data)   : {mean_power_density:.2f} W/m^2\n")
    f.write(f"Mean wind power density (Weibull): {weibull_power_density:.2f} W/m^2\n")
    f.write(f"Hellmann shear exponent (alpha)  : {alpha_hat:.3f}\n")
    f.write(f"Mean estimated WS10M             : {df['WS10M_est'].mean():.3f} m/s\n")

# Seasonal (monthly) climatology
monthly = df.groupby(df.index.month)["WS50M"].agg(["mean", "std"])
monthly.index.name = "Month"
monthly.to_csv(OUT_DIR + "monthly_climatology.csv")


# 4. FIGURES: EDA

# Fig 1: Full daily time series (annual mean overlay)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(df.index, df["WS50M"], lw=0.25, alpha=0.5, color="steelblue", label="Daily WS50M")
annual_mean = df["WS50M"].resample("YS").mean()
ax.plot(annual_mean.index, annual_mean.values, color="darkorange", lw=2, label="Annual mean")
ax.set_ylabel("Wind speed at 50 m (m/s)")
ax.set_xlabel("Year")
ax.set_title("Daily wind speed at 50 m, Mumbai, India (1981-2024)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR + "fig1_timeseries.png")
plt.close(fig)

# Fig 2: Monthly climatology (boxplot-like mean +/- std)
fig, ax = plt.subplots(figsize=(6, 3.6))
months = monthly.index
ax.bar(months, monthly["mean"], yerr=monthly["std"], capsize=3, color="cornflowerblue", edgecolor="k", linewidth=0.5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("Mean wind speed at 50 m (m/s)")
ax.set_title("Monthly wind-speed climatology (1981-2024)")
fig.tight_layout()
fig.savefig(FIG_DIR + "fig2_monthly_climatology.png")
plt.close(fig)

# Fig 3: Weibull fit vs empirical histogram
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(ws, bins=40, density=True, color="lightsteelblue", edgecolor="k", linewidth=0.3, label="Observed")
x = np.linspace(0, ws.max(), 300)
pdf = stats.weibull_min.pdf(x, shape_k, loc_w, scale_c)
ax.plot(x, pdf, color="crimson", lw=2, label=f"Weibull fit (k={shape_k:.2f}, c={scale_c:.2f})")
ax.set_xlabel("Wind speed at 50 m (m/s)")
ax.set_ylabel("Probability density")
ax.set_title("Empirical distribution and fitted Weibull PDF")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR + "fig3_weibull_fit.png")
plt.close(fig)

# Fig 4: Seasonal decomposition (on monthly-resampled series to keep it readable)
monthly_series = df["WS50M"].resample("MS").mean()
decomp = seasonal_decompose(monthly_series, model="additive", period=12, extrapolate_trend="freq")
fig = decomp.plot()
fig.set_size_inches(8, 6)
fig.tight_layout()
fig.savefig(FIG_DIR + "fig4_seasonal_decomposition.png")
plt.close(fig)


# 5. TRAIN / TEST SPLIT
#    Train : 1981-2022   Test : 2023-2024 

train = df.loc[:"2022-12-31"].copy()
test = df.loc["2023-01-01":].copy()
print(f"Train size: {len(train)}  Test size: {len(test)}")


# 6a. MODEL 1 - SARIMA (statistical time-series model)

def fourier_terms(index, period=365.25, order=3):
    t = np.arange(len(index))
    terms = {}
    for k in range(1, order + 1):
        terms[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        terms[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms, index=index)

exog_full = fourier_terms(df.index)
exog_train = exog_full.loc[train.index]
exog_test = exog_full.loc[test.index]

sarima_model = SARIMAX(
    train["WS50M"],
    order=(5, 0, 1),
    exog=exog_train,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = sarima_model.fit(disp=False, maxiter=200)


sarima_preds = []
current_res = sarima_fit
for i, date in enumerate(test.index):
    step_exog = exog_test.iloc[[i]]
    step_forecast = current_res.get_forecast(steps=1, exog=step_exog).predicted_mean.iloc[0]
    sarima_preds.append(step_forecast)
    true_val = pd.Series([df.loc[date, "WS50M"]], index=[date], name="WS50M")
    current_res = current_res.append(true_val, exog=step_exog, refit=False)

sarima_pred = pd.Series(sarima_preds, index=test.index)


# 6b. MODEL 2 - Random Forest Regression (machine-learning model)
#     Features: calendar (day-of-year harmonics, month), lagged wind speed,
#     rolling statistics, and same-day meteorological covariates.

def build_features(frame):
    f = pd.DataFrame(index=frame.index)
    doy = frame.index.dayofyear
    f["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    f["month"] = frame.index.month
    f["lag1"] = frame["WS50M"].shift(1)
    f["lag2"] = frame["WS50M"].shift(2)
    f["lag3"] = frame["WS50M"].shift(3)
    f["lag7"] = frame["WS50M"].shift(7)
    f["roll_mean_7"] = frame["WS50M"].shift(1).rolling(7).mean()
    f["roll_mean_30"] = frame["WS50M"].shift(1).rolling(30).mean()
    f["roll_std_7"] = frame["WS50M"].shift(1).rolling(7).std()
    f["T2M"] = frame["T2M"]
    f["RH2M"] = frame["RH2M"]
    f["PS"] = frame["PS"]
    f["PRECTOTCORR"] = frame["PRECTOTCORR"]
    f["target"] = frame["WS50M"]
    return f

feat_all = build_features(df).dropna()
feat_train = feat_all.loc[:"2022-12-31"]
feat_test = feat_all.loc["2023-01-01":]

X_train, y_train = feat_train.drop(columns="target"), feat_train["target"]
X_test, y_test = feat_test.drop(columns="target"), feat_test["target"]

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_pred = pd.Series(rf.predict(X_test), index=X_test.index)


# 6c. BASELINE - naive persistence (tomorrow = today)

persistence_pred = df["WS50M"].shift(1).loc[test.index]

# Align all predictions/actuals on the same index (RF loses first 30 obs to rolling window)
common_idx = feat_test.index
y_true = df.loc[common_idx, "WS50M"]
sarima_aligned = sarima_pred.loc[common_idx]
persistence_aligned = persistence_pred.loc[common_idx]
rf_aligned = rf_pred.loc[common_idx]

def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100
    return {"Model": label, "MAE (m/s)": mae, "RMSE (m/s)": rmse, "R2": r2, "MAPE (%)": mape}

results = pd.DataFrame([
    metrics(y_true, persistence_aligned, "Persistence (baseline)"),
    metrics(y_true, sarima_aligned, "SARIMA + Fourier annual cycle"),
    metrics(y_true, rf_aligned, "Random Forest Regression"),
])
results = results.set_index("Model")
results.to_csv(OUT_DIR + "model_comparison.csv")
print(results)


# 7. FIGURES: MODEL RESULTS


# Fig 5: Actual vs predicted, first 120 days of test period, both models
window = common_idx[:120]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(window, y_true.loc[window], color="black", lw=1.3, label="Observed")
ax.plot(window, sarima_aligned.loc[window], color="tomato", lw=1.1, ls="--", label="SARIMA forecast")
ax.plot(window, rf_aligned.loc[window], color="seagreen", lw=1.1, ls="-.", label="Random Forest forecast")
ax.set_ylabel("WS50M (m/s)")
ax.set_title("Observed vs. predicted daily wind speed (first 120 test days, 2023)")
ax.legend(fontsize=8)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR + "fig5_actual_vs_predicted.png")
plt.close(fig)

# Fig 6: Scatter actual vs predicted for both models
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharex=True, sharey=True)
for ax, pred, name, color in zip(
    axes, [sarima_aligned, rf_aligned], ["SARIMA", "Random Forest"], ["tomato", "seagreen"]
):
    ax.scatter(y_true, pred, s=4, alpha=0.3, color=color)
    lims = [0, max(y_true.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed (m/s)")
    ax.set_title(name)
axes[0].set_ylabel("Predicted (m/s)")
fig.suptitle("Predicted vs. observed wind speed on the 2023-2024 test set")
fig.tight_layout()
fig.savefig(FIG_DIR + "fig6_scatter_actual_vs_pred.png")
plt.close(fig)

# Fig 7: Random Forest feature importance
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()
fig, ax = plt.subplots(figsize=(6, 4))
importances.plot(kind="barh", ax=ax, color="seagreen", edgecolor="k", linewidth=0.4)
ax.set_xlabel("Relative importance")
ax.set_title("Random Forest feature importance")
fig.tight_layout()
fig.savefig(FIG_DIR + "fig7_rf_feature_importance.png")
plt.close(fig)

# Fig 8: Residual distributions
fig, ax = plt.subplots(figsize=(6, 3.8))
resid_sarima = y_true - sarima_aligned
resid_rf = y_true - rf_aligned
ax.hist(resid_sarima, bins=40, alpha=0.55, label="SARIMA residuals", color="tomato")
ax.hist(resid_rf, bins=40, alpha=0.55, label="Random Forest residuals", color="seagreen")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Residual (m/s)")
ax.set_ylabel("Count")
ax.set_title("Residual distribution on the test set")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR + "fig8_residuals.png")
plt.close(fig)

# Fig 9: ACF/PACF of the training series (for model justification)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
plot_acf(train["WS50M"].iloc[-2000:], lags=40, ax=axes[0])
plot_pacf(train["WS50M"].iloc[-2000:], lags=40, ax=axes[1], method="ywm")
axes[0].set_title("ACF (last 2000 training days)")
axes[1].set_title("PACF (last 2000 training days)")
fig.tight_layout()
fig.savefig(FIG_DIR + "fig9_acf_pacf.png")
plt.close(fig)

print("\nAll figures and tables written to project/figures and project/data.")
print(results.to_string())
