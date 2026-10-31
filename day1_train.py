# ============================================================
# EL NIÑO PREDICTOR — DAY 1
# Using your local CSV files:
#   - oni_historical_data.csv  (ONI index, 1950–present)
#   - mei_v2_historical_data.csv (MEI index, 1979–present)
# ============================================================

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import joblib
warnings.filterwarnings('ignore')

# Create output folders if they don't exist
os.makedirs('data',   exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================
# STEP 1 — LOAD AND PARSE ONI DATA
# ============================================================
# The ONI (Oceanic Niño Index) measures sea surface temperature
# anomalies in the central Pacific (Niño 3.4 region).
# The CSV has a wide format: each row = 1 year,
# each column = one 3-month season (DJF, JFM, FMA, ... NDJ).
# We need to convert this into long format: one row per month.
# ============================================================
print("=" * 60)
print("STEP 1: Loading ONI Data")
print("=" * 60)

oni_wide = pd.read_csv('oni_historical_data.csv')
print(f"ONI raw shape: {oni_wide.shape}")
print(f"Columns: {list(oni_wide.columns)}")

# The season columns like "DJF" represent the CENTER month.
# DJF = Jan, JFM = Feb, FMA = Mar, MAM = Apr, AMJ = May,
# MJJ = Jun, JJA = Jul, JAS = Aug, ASO = Sep, SON = Oct,
# OND = Nov, NDJ = Dec
season_to_month = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
    'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
    'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
}

# Melt: wide → long (year, season, value) → (year, month, oni)
year_col = oni_wide.columns[0]   # first column is 'Year'
season_cols = [c for c in oni_wide.columns if c in season_to_month]

oni_long = oni_wide.melt(
    id_vars=year_col,
    value_vars=season_cols,
    var_name='season',
    value_name='oni'
)
oni_long['month'] = oni_long['season'].map(season_to_month)
oni_long['year']  = oni_long[year_col].astype(int)
oni_long['oni']   = pd.to_numeric(oni_long['oni'], errors='coerce')
oni_long = oni_long.dropna(subset=['oni'])
oni_long['date']  = pd.to_datetime(
    dict(year=oni_long['year'], month=oni_long['month'], day=1)
)
oni_long = oni_long[['date','year','month','oni']].sort_values('date').reset_index(drop=True)

print(f"\nONI long format: {len(oni_long)} monthly records")
print(f"Range: {oni_long['date'].min().strftime('%b %Y')} → {oni_long['date'].max().strftime('%b %Y')}")
print(oni_long.head())

# ============================================================
# STEP 2 — LOAD AND PARSE MEI DATA
# ============================================================
# MEI (Multivariate ENSO Index) is a richer measure of El Niño
# that combines 6 variables: sea level pressure, SST, air temp,
# cloudiness, zonal and meridional wind.
# Higher MEI = stronger El Niño conditions.
# The CSV uses bimonthly seasons: DJ, JF, FM, MA, AM, MJ,
# JJ, JA, AS, SO, ON, ND — we map each to its center month.
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading MEI Data")
print("=" * 60)

mei_wide = pd.read_csv('mei_v2_historical_data.csv')
print(f"MEI raw shape: {mei_wide.shape}")
print(f"Columns: {list(mei_wide.columns)}")

# Bimonthly season → center month mapping
bimonth_to_month = {
    'DJ': 1, 'JF': 2, 'FM': 3, 'MA': 4,
    'AM': 5, 'MJ': 6, 'JJ': 7, 'JA': 8,
    'AS': 9, 'SO': 10, 'ON': 11, 'ND': 12
}

year_col_mei  = mei_wide.columns[0]
bimonth_cols  = [c for c in mei_wide.columns if c in bimonth_to_month]

mei_long = mei_wide.melt(
    id_vars=year_col_mei,
    value_vars=bimonth_cols,
    var_name='season',
    value_name='mei'
)
mei_long['month'] = mei_long['season'].map(bimonth_to_month)
mei_long['year']  = mei_long[year_col_mei].astype(int)
mei_long['mei']   = pd.to_numeric(mei_long['mei'], errors='coerce')
mei_long = mei_long.dropna(subset=['mei'])
mei_long['date']  = pd.to_datetime(
    dict(year=mei_long['year'], month=mei_long['month'], day=1)
)
mei_long = mei_long[['date','mei']].sort_values('date').reset_index(drop=True)

print(f"\nMEI long format: {len(mei_long)} monthly records")
print(f"Range: {mei_long['date'].min().strftime('%b %Y')} → {mei_long['date'].max().strftime('%b %Y')}")

# ============================================================
# STEP 3 — MERGE ONI + MEI
# ============================================================
# We join on date. ONI goes back to 1950, MEI only to 1979.
# Where MEI is missing (pre-1979), we fill with 0
# (neutral — not ideal but acceptable for a first version).
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Merging ONI + MEI")
print("=" * 60)

df = pd.merge(oni_long, mei_long, on='date', how='left')
df['mei'] = df['mei'].fillna(0)   # pre-1979 MEI = 0 (neutral)

print(f"Merged dataset: {len(df)} rows")
print(f"MEI available for {(df['mei'] != 0).sum()} months (post-1979)")
print(df[['date','oni','mei']].tail())

# ============================================================
# STEP 4 — LABEL EL NIÑO MONTHS
# ============================================================
# Official NOAA definition: El Niño = ONI ≥ +0.5°C for 5
# consecutive overlapping 3-month seasons.
# For ML purposes we use a simpler proxy that captures the
# same events: ONI ≥ +0.5 in the current month.
# label = 1 → El Niño, label = 0 → Neutral / La Niña
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Labelling El Niño Events")
print("=" * 60)

df['label'] = (df['oni'] >= 0.5).astype(int)

el_niño  = df['label'].sum()
total    = len(df)
pct      = el_niño / total * 100
print(f"El Niño months : {el_niño}  ({pct:.1f}%)")
print(f"Non-El Niño    : {total - el_niño}  ({100-pct:.1f}%)")
print(f"\nNote: ~{pct:.0f}% El Niño is class imbalance.")
print("This means accuracy is a BAD metric — we'll use ROC-AUC instead.")
print("(A model that always says 'no El Niño' gets 70% accuracy but is useless.)")

# ============================================================
# STEP 5 — EDA (Exploratory Data Analysis)
# ============================================================
# Before building any model, always visualize the data.
# This helps you understand patterns, spot anomalies,
# and choose the right features.
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: EDA — Plotting Key Patterns")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('El Niño EDA — ONI & MEI Indices', fontsize=14, fontweight='bold')

# ── Plot 1: ONI time series with El Niño shading ──
ax = axes[0, 0]
ax.plot(df['date'], df['oni'], color='#333', lw=0.8, alpha=0.7)
ax.fill_between(df['date'], df['oni'], 0,
    where=df['oni'] >= 0.5,  color='#e74c3c', alpha=0.5, label='El Niño (≥+0.5)')
ax.fill_between(df['date'], df['oni'], 0,
    where=df['oni'] <= -0.5, color='#2980b9', alpha=0.5, label='La Niña (≤−0.5)')
ax.axhline(0.5,  color='red',  lw=0.8, ls='--')
ax.axhline(-0.5, color='blue', lw=0.8, ls='--')
ax.axhline(0,    color='black',lw=0.5)
ax.set_title('ONI Over Time')
ax.set_ylabel('ONI Anomaly (°C)')
ax.legend(fontsize=8)

# ── Plot 2: ONI distribution — is it normal? ──
# Understanding the distribution helps choose the right model.
ax = axes[0, 1]
ax.hist(df['oni'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0.5,  color='red',  lw=1.5, ls='--', label='+0.5 El Niño cutoff')
ax.axvline(-0.5, color='blue', lw=1.5, ls='--', label='-0.5 La Niña cutoff')
ax.axvline(df['oni'].mean(), color='green', lw=1.5, ls='-', label=f'Mean={df["oni"].mean():.2f}')
ax.set_title('ONI Distribution')
ax.set_xlabel('ONI (°C)')
ax.legend(fontsize=7)

# ── Plot 3: Average ONI by month — seasonal pattern ──
# El Niño tends to peak in Dec–Jan (boreal winter).
# This seasonal pattern is why we include month encoding as a feature.
ax = axes[0, 2]
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_avg = df.groupby('month')['oni'].mean()
colors = ['#e74c3c' if v > 0 else '#2980b9' for v in monthly_avg.values]
ax.bar(month_names, monthly_avg.values, color=colors, alpha=0.8)
ax.axhline(0, color='black', lw=0.5)
ax.set_title('Avg ONI by Month\n(El Niño peaks in Dec–Feb)')
ax.tick_params(axis='x', rotation=45)
ax.set_ylabel('Avg ONI (°C)')

# ── Plot 4: ONI autocorrelation ──
# Autocorrelation tells us: "if ONI is high this month,
# how likely is it to still be high k months later?"
# High autocorrelation at lag 1-6 → lag features will be useful.
ax = axes[1, 0]
lags = range(1, 25)
acf  = [df['oni'].autocorr(lag=l) for l in lags]
ax.bar(lags, acf, color='steelblue', alpha=0.8)
ax.axhline(0.2,  color='red', lw=0.8, ls='--', label='±0.2 rough significance')
ax.axhline(-0.2, color='red', lw=0.8, ls='--')
ax.axhline(0, color='black', lw=0.5)
ax.set_title('ONI Autocorrelation\n(justifies lag features)')
ax.set_xlabel('Lag (months)')
ax.legend(fontsize=8)
print(f"Autocorr lag-1: {df['oni'].autocorr(1):.3f}")
print(f"Autocorr lag-3: {df['oni'].autocorr(3):.3f}")
print(f"Autocorr lag-6: {df['oni'].autocorr(6):.3f}")
print(f"High autocorrelation → lag features will help the model")

# ── Plot 5: ONI vs MEI scatter (are they correlated?) ──
# If MEI and ONI are highly correlated, MEI adds extra
# multivariate signal that ONI alone doesn't capture.
ax = axes[1, 1]
mei_available = df[df['mei'] != 0]
ax.scatter(mei_available['oni'], mei_available['mei'],
           alpha=0.3, s=10, color='purple')
ax.axvline(0.5,  color='red',  lw=0.8, ls='--')
ax.axhline(0.5,  color='red',  lw=0.8, ls='--')
ax.set_title('ONI vs MEI Correlation\n(MEI adds multivariate signal)')
ax.set_xlabel('ONI (°C)')
ax.set_ylabel('MEI')
corr = mei_available[['oni','mei']].corr().iloc[0,1]
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
        fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
print(f"\nONI–MEI correlation: {corr:.3f}")

# ── Plot 6: ONI heatmap — year × month ──
# Classic climate visualization. Warm colors = El Niño years.
# You'll recognize 1982–83, 1997–98, 2015–16 as the big ones.
ax = axes[1, 2]
pivot = df.pivot_table(values='oni', index='year', columns='month')
pivot.columns = month_names
sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label':'ONI (°C)'}, vmin=-2.5, vmax=2.5)
ax.set_title('ONI Heatmap (Year × Month)\nRed = El Niño, Blue = La Niña')
ax.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=130, bbox_inches='tight')
plt.show()
print("\n✅ eda_plots.png saved")

# ============================================================
# STEP 6 — FEATURE ENGINEERING
# ============================================================
# Raw ONI value alone is not enough. We create extra features
# that capture:
#   - Memory    : what was ONI k months ago? (lag features)
#   - Trend     : is ONI rising or falling? (diff, slope)
#   - Momentum  : how long has it been elevated? (consec_above)
#   - Seasonality: which month of the year? (sin/cos encoding)
#   - Smoothing : rolling mean/std to reduce noise
# All of these come from concepts you've studied in your course.
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Feature Engineering")
print("=" * 60)

df_f = df.copy()

# ── Lag features: ONI value t-1, t-2, t-3, t-6, t-12 months ago ──
# The model uses these to understand the "recent history" of ONI.
# shift(k) moves values DOWN by k rows — so row i gets the value
# that was in row i-k originally (i.e., k months earlier).
for lag in [1, 2, 3, 6, 12]:
    df_f[f'oni_lag{lag}'] = df_f['oni'].shift(lag)

# ── Rolling statistics (calculated on past data only, shift(1) avoids leakage) ──
# rolling(w).mean() = average of the past w months
# rolling(w).std()  = how volatile/variable ONI has been
# rolling(w).max()  = peak ONI in the past w months
# shift(1) ensures we only use data BEFORE the current month —
# this prevents "data leakage" (using future info to predict the present)
for w in [3, 6, 12]:
    past = df_f['oni'].shift(1)
    df_f[f'oni_roll_mean_{w}'] = past.rolling(w).mean()
    df_f[f'oni_roll_std_{w}']  = past.rolling(w).std()
    df_f[f'oni_roll_max_{w}']  = past.rolling(w).max()

# ── MEI lag features (MEI at t-1, t-3) ──
# MEI adds multivariate information beyond just SST.
# We lag it to avoid leakage.
df_f['mei_lag1'] = df_f['mei'].shift(1)
df_f['mei_lag3'] = df_f['mei'].shift(3)

# ── Trend: slope of ONI over last 6 months ──
# A positive slope means ONI is rising → El Niño developing.
# We compute the best-fit line slope using polyfit.
def rolling_slope(series, window=6):
    out  = [np.nan] * len(series)
    vals = series.values
    for i in range(window, len(vals)):
        y = vals[i - window:i]
        if not np.isnan(y).any():
            out[i] = np.polyfit(np.arange(window), y, 1)[0]
    return out

df_f['oni_trend_6m'] = rolling_slope(df_f['oni'])

# ── Cyclical month encoding ──
# Month 1 and month 12 are close in climate terms (both winter)
# but far apart numerically (1 vs 12). Sin/cos encoding wraps
# the year into a circle so the model understands this.
df_f['month_sin'] = np.sin(2 * np.pi * df_f['month'] / 12)
df_f['month_cos'] = np.cos(2 * np.pi * df_f['month'] / 12)

# ── Consecutive months above El Niño threshold ──
# Counts how many months in a row ONI has been ≥ 0.5.
# This captures "momentum" — a strong ongoing event vs a spike.
cnt    = 0
consec = []
for v in df_f['oni']:
    cnt = cnt + 1 if v >= 0.5 else 0
    consec.append(cnt)
df_f['consec_above'] = consec

# ── Rate of change: month-over-month and 3-month change ──
# diff(k) = current value − value k months ago
# Positive = ONI rising (El Niño developing)
# Negative = ONI falling (event ending or La Niña developing)
df_f['oni_diff1'] = df_f['oni'].diff(1)
df_f['oni_diff3'] = df_f['oni'].diff(3)

# Drop rows where rolling/lag features are NaN
# (the first 12 months won't have 12-month lag values etc.)
df_f = df_f.dropna().reset_index(drop=True)

FEATURES = [
    # Core signal
    'oni', 'mei_lag1', 'mei_lag3',
    # Memory — where was ONI in the past?
    'oni_lag1', 'oni_lag2', 'oni_lag3', 'oni_lag6', 'oni_lag12',
    # Smoothed signal — noise-reduced ONI
    'oni_roll_mean_3', 'oni_roll_mean_6', 'oni_roll_mean_12',
    # Volatility — how stable has ONI been?
    'oni_roll_std_3', 'oni_roll_std_6',
    # Peak — highest recent ONI
    'oni_roll_max_3', 'oni_roll_max_6',
    # Trend and momentum
    'oni_trend_6m', 'oni_diff1', 'oni_diff3', 'consec_above',
    # Seasonality
    'month_sin', 'month_cos',
]

X = df_f[FEATURES]
y = df_f['label']

print(f"Total features : {len(FEATURES)}")
print(f"Dataset rows   : {len(df_f)}")
print(f"\nFeature list:")
for i, f in enumerate(FEATURES, 1):
    print(f"  {i:2d}. {f}")

df_f[FEATURES + ['label','date','year','month']].to_csv('data/processed.csv', index=False)
joblib.dump(FEATURES, 'models/features.pkl')
print("\n✅ data/processed.csv and models/features.pkl saved")

# ============================================================
# STEP 7 — TRAIN/TEST SPLIT (time-based, NOT random)
# ============================================================
# For time series data, NEVER use random split.
# Why? Because if row 500 is Jan 2010 and row 501 is Feb 2010,
# a random split might put Feb in training and Jan in testing.
# The model then "sees the future" during training → cheating.
# We always split by date: train on past, test on future.
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Time-Based Train/Test Split")
print("=" * 60)

split_date = '2010-01-01'
tr = df_f['date'] < split_date
te = df_f['date'] >= split_date

X_tr, X_te = X[tr], X[te]
y_tr, y_te = y[tr], y[te]

print(f"Train : {len(X_tr)} months  (up to Dec 2009)")
print(f"Test  : {len(X_te)} months  (Jan 2010 → present)")
print(f"Train El Niño %: {y_tr.mean()*100:.1f}%")
print(f"Test  El Niño %: {y_te.mean()*100:.1f}%")

# StandardScaler: transforms each feature to mean=0, std=1
# This is required for both Logistic Regression and SVM because
# they are sensitive to feature scale (oni might range -3 to +3,
# while consec_above might range 0 to 20 — unfair comparison).
# IMPORTANT: fit scaler ONLY on training data, then apply to test.
# Fitting on test data would leak test statistics into training.
scaler  = StandardScaler()
Xtr_s   = scaler.fit_transform(X_tr)   # learn mean/std from train
Xte_s   = scaler.transform(X_te)       # apply same scale to test

# ============================================================
# STEP 8 — TRAIN LOGISTIC REGRESSION
# ============================================================
# Logistic Regression models the PROBABILITY of El Niño.
# It learns a weight for each feature and combines them into
# a score, then squashes through sigmoid to get prob ∈ [0,1].
#
# Key parameters:
#   class_weight='balanced' : automatically adjusts for the fact
#     that El Niño months are rarer. Without this, the model
#     would just predict "no El Niño" all the time.
#   C=0.1 : regularization strength. Lower C = more regularization
#     (like Ridge regression penalty). Prevents overfitting.
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Training Logistic Regression")
print("=" * 60)

lr = LogisticRegression(
    class_weight='balanced',
    C=0.1,
    max_iter=1000,
    random_state=42
)
lr.fit(Xtr_s, y_tr)

lr_prob = lr.predict_proba(Xte_s)[:, 1]   # probability of El Niño
lr_pred = lr.predict(Xte_s)
lr_auc  = roc_auc_score(y_te, lr_prob)
lr_ap   = average_precision_score(y_te, lr_prob)

# Cross-validation: train/evaluate 5 different time slices
# Gives a more stable estimate of real model performance
cv     = StratifiedKFold(n_splits=5, shuffle=False)
lr_cv  = cross_val_score(lr, Xtr_s, y_tr, cv=cv, scoring='roc_auc').mean()

print(f"ROC-AUC  (test) : {lr_auc:.4f}")
print(f"Avg Prec (test) : {lr_ap:.4f}")
print(f"CV ROC-AUC      : {lr_cv:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_te, lr_pred, target_names=['Not El Niño','El Niño']))

# ============================================================
# STEP 9 — TRAIN SVM
# ============================================================
# SVM (Support Vector Machine) finds the best decision boundary
# between El Niño and non-El Niño points in feature space.
# RBF kernel: handles non-linear patterns (the relationship
# between ONI features and El Niño is not a straight line).
#
# Key parameters:
#   kernel='rbf'      : Radial Basis Function — captures curves
#   C=1.0             : how much to penalize misclassification
#   gamma='scale'     : how far each training point's influence
#                       reaches. 'scale' = 1/(n_features * X.var())
#   probability=True  : needed to call predict_proba() for ROC curve
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Training SVM (RBF kernel)")
print("=" * 60)

svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm.fit(Xtr_s, y_tr)

svm_prob = svm.predict_proba(Xte_s)[:, 1]
svm_pred = svm.predict(Xte_s)
svm_auc  = roc_auc_score(y_te, svm_prob)
svm_ap   = average_precision_score(y_te, svm_prob)
svm_cv   = cross_val_score(svm, Xtr_s, y_tr, cv=cv, scoring='roc_auc').mean()

print(f"ROC-AUC  (test) : {svm_auc:.4f}")
print(f"Avg Prec (test) : {svm_ap:.4f}")
print(f"CV ROC-AUC      : {svm_cv:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_te, svm_pred, target_names=['Not El Niño','El Niño']))

# ============================================================
# STEP 10 — MODEL COMPARISON PLOTS
# ============================================================
# ROC Curve: plots True Positive Rate vs False Positive Rate
#   at every possible threshold. AUC = area under this curve.
#   AUC=1.0 → perfect, AUC=0.5 → random guessing.
#
# Precision-Recall Curve: more informative for imbalanced data.
#   Precision = of all "El Niño" predictions, how many were right?
#   Recall    = of all actual El Niño months, how many did we catch?
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Comparison Plots")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Logistic Regression vs SVM — El Niño Classification', fontsize=13)

# ROC Curve
ax = axes[0]
fpr_l, tpr_l, _ = roc_curve(y_te, lr_prob)
fpr_s, tpr_s, _ = roc_curve(y_te, svm_prob)
ax.plot(fpr_l, tpr_l, 'b-', lw=2, label=f'LR  AUC={lr_auc:.3f}')
ax.plot(fpr_s, tpr_s, 'r-', lw=2, label=f'SVM AUC={svm_auc:.3f}')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(fontsize=9)

# Precision-Recall Curve
ax = axes[1]
p_l, r_l, _ = precision_recall_curve(y_te, lr_prob)
p_s, r_s, _ = precision_recall_curve(y_te, svm_prob)
ax.plot(r_l, p_l, 'b-', lw=2, label=f'LR  AP={lr_ap:.3f}')
ax.plot(r_s, p_s, 'r-', lw=2, label=f'SVM AP={svm_ap:.3f}')
ax.axhline(y_te.mean(), color='gray', ls='--', lw=1,
           label=f'No-skill baseline ({y_te.mean():.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve\n(key for imbalanced classes)')
ax.legend(fontsize=9)

# Bar chart summary
ax = axes[2]
labels  = ['ROC-AUC\n(test)', 'Avg Prec\n(test)', 'CV AUC\n(5-fold)']
lr_vals  = [lr_auc,  lr_ap,  lr_cv]
svm_vals = [svm_auc, svm_ap, svm_cv]
x = np.arange(3); w = 0.3
ax.bar(x - w/2, lr_vals,  w, color='#3498db', alpha=0.85, label='Logistic Regression')
ax.bar(x + w/2, svm_vals, w, color='#e74c3c', alpha=0.85, label='SVM RBF')
for i, (a, b) in enumerate(zip(lr_vals, svm_vals)):
    ax.text(i - w/2, a + 0.02, f'{a:.3f}', ha='center', fontsize=8)
    ax.text(i + w/2, b + 0.02, f'{b:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.15)
ax.axhline(0.5, color='gray', ls='--', lw=1, label='Random baseline')
ax.set_title('Model Comparison')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
print("✅ model_comparison.png saved")

# ── Prediction timeline on test set ──
# Visual check: does the predicted probability actually rise
# during known El Niño events (2015-16, 2023-24)?
test_df = df_f[te].copy()
test_df['svm_prob'] = svm_prob
test_df['lr_prob']  = lr_prob
test_df['actual']   = y_te.values

plt.figure(figsize=(14, 4))
plt.fill_between(test_df['date'], test_df['actual'].astype(float),
                 alpha=0.2, color='red', label='Actual El Niño months')
plt.plot(test_df['date'], test_df['svm_prob'], 'r-',  lw=1.5,
         label=f'SVM probability (AUC={svm_auc:.3f})')
plt.plot(test_df['date'], test_df['lr_prob'],  'b--', lw=1.2,
         label=f'LR probability  (AUC={lr_auc:.3f})')
plt.axhline(0.5, color='gray', ls=':', lw=1, label='0.5 decision threshold')
plt.xlabel('Date')
plt.ylabel('Predicted El Niño Probability')
plt.title('Predicted Probability vs Actual — Test Period (2010 → present)')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('predictions_timeline.png', dpi=130, bbox_inches='tight')
plt.show()
print("✅ predictions_timeline.png saved")

# ============================================================
# STEP 11 — SAVE BEST MODEL
# ============================================================
# We pick the model with the higher ROC-AUC on the test set.
# We save 4 things:
#   model.pkl    — the trained model (SVM or LR)
#   scaler.pkl   — the StandardScaler (MUST use same scaler
#                  when predicting in the Flask app)
#   features.pkl — the ordered list of feature names
#   metadata.json — scores and info for the Flask app to display
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: Saving Best Model")
print("=" * 60)

if svm_auc >= lr_auc:
    best, best_name, best_auc, best_ap = svm, "SVM (RBF kernel)", svm_auc, svm_ap
else:
    best, best_name, best_auc, best_ap = lr,  "Logistic Regression", lr_auc, lr_ap

print(f"Winner → {best_name}  (ROC-AUC = {best_auc:.4f})")

joblib.dump(best,     'models/model.pkl')
joblib.dump(scaler,   'models/scaler.pkl')
joblib.dump(FEATURES, 'models/features.pkl')

metadata = {
    "model_name"    : best_name,
    "roc_auc"       : round(best_auc, 4),
    "avg_precision" : round(best_ap, 4),
    "lr_auc"        : round(lr_auc, 4),
    "svm_auc"       : round(svm_auc, 4),
    "train_period"  : "1950–2009",
    "test_period"   : "2010–present",
    "n_features"    : len(FEATURES),
    "features"      : FEATURES,
    "threshold"     : 0.5,
    "data_source"   : "NOAA ONI + MEI v2"
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ models/model.pkl")
print("✅ models/scaler.pkl")
print("✅ models/features.pkl")
print("✅ models/metadata.json")

# ============================================================
# STEP 12 — SANITY CHECK: PREDICT LATEST MONTH
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: Sanity Check — Most Recent Month")
print("=" * 60)

latest    = df_f.tail(1)
X_latest  = scaler.transform(latest[FEATURES])
prob      = best.predict_proba(X_latest)[0][1]
pred      = best.predict(X_latest)[0]
risk      = "HIGH 🔴" if prob > 0.7 else "MEDIUM 🟡" if prob > 0.4 else "LOW 🟢"

print(f"Month       : {latest['date'].values[0]}")
print(f"ONI         : {latest['oni'].values[0]:.3f} °C")
print(f"MEI (lag-1) : {latest['mei_lag1'].values[0]:.3f}")
print(f"El Niño prob: {prob*100:.1f}%")
print(f"Prediction  : {'El Niño ✅' if pred==1 else 'Not El Niño'}")
print(f"Risk level  : {risk}")

print(f"""
{'='*60}
✅ DAY 1 COMPLETE
{'='*60}
  Logistic Regression ROC-AUC : {lr_auc:.4f}  (CV: {lr_cv:.4f})
  SVM (RBF)           ROC-AUC : {svm_auc:.4f}  (CV: {svm_cv:.4f})
  Best model saved    : {best_name}

Files created:
  data/processed.csv
  models/model.pkl | scaler.pkl | features.pkl | metadata.json
  eda_plots.png | model_comparison.png | predictions_timeline.png

Note for your resume:
  "Trained {best_name} on 70+ years of NOAA ONI + MEI
   climate data achieving ROC-AUC of {best_auc:.2f}"

Next → Day 3: Flask web app (3 routes + SQLite logging)
""")
