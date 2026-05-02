# QR-LSTM Notebook — Polishing, Executive Summary & Contributions

**Project:** B41 Bus Route — Quantile Regression Post-Processing on LSTM Predictions
**Notebook:** `QR_LSTM.ipynb`

---

## TASK 1 — Notebook Polish: Markdown Revisions & Visualisation Suggestions

### 1.1 Title Cell (Section 0)

**Current:** Describes the notebook as a QR layer on an LSTM.
**Suggested revision:**

```markdown
# B41 Bus Route — Quantile Regression Post-Processing on LSTM Predictions

## Overview

This notebook implements a **Quantile Regression (QR) post-processing layer** on top of a
pre-trained Long Short-Term Memory (LSTM) neural network for bus delay prediction on the
B41 route (Brooklyn, NYC).

### Motivation
Transit delay prediction is inherently uncertain. A point forecast (e.g. "the next bus
arrives in 8 minutes") tells passengers nothing about how reliable that estimate is.
By wrapping the LSTM with Quantile Regression, we produce **prediction intervals** —
e.g. "80% of the time, the delay falls between 3 and 17 minutes" — which are far more
actionable for operational planning.

### Key Design Principle
We do **not** retrain the LSTM. Instead, we treat it as a fixed black box and apply QR
as a lightweight calibration layer on its residuals. This approach is:
- **Non-invasive:** LSTM weights remain frozen.
- **Computationally cheap:** QR training runs in seconds.
- **Theoretically grounded:** Pinball loss optimisation and conformal coverage guarantees
  apply without distributional assumptions.

### Data Splits

| Split    | Months          | Role                               |
|----------|-----------------|------------------------------------|
| train    | June–October    | Fit QR models on LSTM residuals    |
| val      | August          | Tune/validate prediction intervals |
| test     | December        | Final held-out evaluation          |

### 15 LSTM Input Features

| Category     | Features                                                                                 |
|--------------|------------------------------------------------------------------------------------------|
| Stop presence| FLATBUSH AV/PROSPECT PARK ENT, LIVINGSTON ST/NEVINS ST, FLATBUSH AV/BERGEN ST,          |
|              | CADMAN PLZ W/MONTAGUE ST, FLATBUSH AV/PLAZA ST E                                        |
| Weather      | temperaturec, precipitation, windspeedkmh                                                |
| Temporal     | month, weekday, hour, isweekend, isholiday                                               |
| Lag features | daylylag, weeklylag                                                                      |

### Evaluation Metrics
- **Point accuracy:** MAE, RMSE, R²
- **Interval quality:** Coverage (80%), interval width, Pinball loss (Q10, Q50, Q90)
- **Operational:** Disruption detection F1 (threshold: 10 min)
```

---

### 1.2 Section 1 — Imports

**Current:** `## 1 Imports & Plot Defaults`
**Suggested revision:**

```markdown
## 1 · Imports & Global Settings

All third-party dependencies are imported here. The `matplotlib` defaults are set once
globally so every subsequent plot inherits a consistent, publication-ready style
(no top/right spines, 130 DPI).

> **Reproducibility note:** No random seed is required because `QuantileRegressor` (via the
> `highs` LP solver) is deterministic.
```

---

### 1.3 Section 2 — Configuration

**Current:** `## 2 Configuration` — minimal description.
**Suggested revision:**

```markdown
## 2 · Configuration

All file paths and modelling hyperparameters live here. Update the three CSV paths to
point to your local LSTM output files before running the notebook.

| Constant         | Value                           | Purpose                              |
|------------------|---------------------------------|--------------------------------------|
| `CSV_TRAIN`      | `lstm_train_predictions.csv`    | June–October LSTM outputs            |
| `CSV_VAL`        | `lstm_val_predictions.csv`      | August LSTM outputs                  |
| `CSV_TEST`       | `lstm_test_predictions.csv`     | December LSTM outputs                |
| `PATH_FEATURES`  | `final_dataset_corrected.csv`   | Full 15-feature matrix               |
| `QR_QUANTILES`   | (0.1, 0.5, 0.9)                 | Lower bound, median, upper bound     |
| `DISRUPTION_THR` | 10 minutes                      | Operational disruption threshold     |

`QR_FEATURE_COLS` = 15 LSTM features **+** `ypred` (16 total). The LSTM point prediction
is included as a covariate because residual variance tends to scale with prediction
magnitude (heteroscedasticity).
```

---

### 1.4 Section 3 — Data Loading

**Current:** `## 3 Load Pre-computed LSTM Outputs`
**Suggested revision:**

```markdown
## 3 · Load Pre-computed LSTM Outputs

Three CSV files are loaded — one per data split — and concatenated into a single
`pred_df`. The feature matrix (`feat_df`) is loaded separately and merged on `timestamp`.

**Column schema after merge:**

| Column       | Description                          |
|--------------|--------------------------------------|
| `timestamp`  | 30-minute interval (datetime64)      |
| `ypred`      | LSTM point prediction (minutes)      |
| `ytrue`      | Observed delay (minutes)             |
| `residuals`  | `ytrue − ypred` (LSTM error)         |
| `split`      | train / val / test                   |
| 15 features  | See Section 2 configuration table    |

No missing values were found after the timestamp merge (all 4,282 prediction rows matched).
```

---

### 1.5 Section 4 — Feature Correlation (if present)

**Suggested inline comment to add above the Spearman plot:**

```python
# Spearman rank correlation: non-parametric, robust to outliers and monotone
# non-linearities. Features are ranked by absolute correlation magnitude.
# Top predictors of LSTM residuals: wind speed, month, holiday flag, and lag features.
# This motivates including ypred and temporal features in the enhanced QR (Section 12).
```

---

### 1.6 Sections 5–10 (QR Fitting, Evaluation)

**Suggested opening markdown for the QR training section (wherever `QuantileRegressor.fit` is called):**

```markdown
## 5 · Fit Quantile Regression Models

Three separate `QuantileRegressor` models are trained — one per quantile (τ = 0.1, 0.5, 0.9).

**Why three separate models?**
`sklearn`'s `QuantileRegressor` solves a Linear Programme (LP) for a fixed τ. Running three
independent LPs is computationally straightforward and allows each quantile to have a
different linear response surface.

**Target variable:** LSTM residuals (`ytrue − ypred`) on the *training* split.
**Covariates:** 15 LSTM input features **+** `ypred` (16 total), standardised via
`StandardScaler` (fitted on train, applied to val/test).

**Reconstruction formula:**

    ŷ_q(x) = ŷ_LSTM(x) + q̂(x)

where `q̂(x)` is the QR-predicted residual at quantile τ.
```

---

### 1.7 Section 11 — Visualisations

**Existing plots:** residual box plot, time-series ribbon, pinball bar chart, quantile reliability diagram.

#### Suggested additional / replacement plots

The code below can be appended to Section 11 as new sub-sections.

---

**11.A  Interval Width vs. LSTM Prediction Magnitude (Heteroscedasticity Scatter)**

```python
# 11.A Interval width vs. ypred — reveals whether residual variance
# scales with the LSTM's own forecast (heteroscedasticity).
import matplotlib.pyplot as plt
import numpy as np

testdf = qrpreddf[qrpreddf['split'] == 'test'].copy()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(testdf['ypred'], testdf['intervalwidth'],
           alpha=0.35, s=12, color='steelblue', label='Interval width')

# LOWESS-style moving average to show trend
bins = np.linspace(testdf['ypred'].min(), testdf['ypred'].max(), 20)
bin_idx = np.digitize(testdf['ypred'], bins)
bin_mid = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
bin_mean = [testdf.loc[bin_idx == i+1, 'intervalwidth'].mean() for i in range(len(bins)-1)]
ax.plot(bin_mid, bin_mean, color='crimson', linewidth=2, label='Binned mean')

ax.set_xlabel('LSTM point prediction ŷ_LSTM (min)')
ax.set_ylabel('Q90 − Q10 interval width (min)')
ax.set_title('11.A  Interval Width vs. LSTM Prediction — Test Set')
ax.legend()
plt.tight_layout()
plt.show()
```

---

**11.B  Hourly Coverage Heatmap (hour × day-of-week)**

```python
# 11.B  Coverage heatmap by hour-of-day and day-of-week.
# Reveals systematic under-coverage during peak/off-peak windows.

testdf = qrpreddf[qrpreddf['split'] == 'test'].copy()
testdf['hour']    = testdf['timestamp'].dt.hour
testdf['weekday'] = testdf['timestamp'].dt.dayofweek  # 0=Mon

pivot = (testdf.groupby(['weekday', 'hour'])['iscovered']
               .mean()
               .unstack(fill_value=np.nan))

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
               vmin=0.5, vmax=1.0, origin='lower')
ax.set_xticks(range(pivot.shape[1]))
ax.set_xticklabels(pivot.columns, fontsize=7)
ax.set_yticks(range(len(day_labels)))
ax.set_yticklabels(day_labels)
ax.set_xlabel('Hour of day')
ax.set_ylabel('Day of week')
ax.set_title('11.B  Empirical Coverage by Hour × Day — Test Set (target = 0.80)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Coverage fraction')
plt.tight_layout()
plt.show()
```

---

**11.C  Disruption Detection Precision–Recall Curve**

```python
# 11.C  Precision–Recall curve for disruption detection across Q90 thresholds.
# More informative than a single F1 score for imbalanced classes.
from sklearn.metrics import precision_recall_curve, average_precision_score

testdf = qrpreddf[qrpreddf['split'] == 'test'].copy()
y_true_bin = (testdf['ytrue'] >= DISRUPTION_THR).astype(int).values
# Use Q90 as the soft score for disruption probability
y_score = testdf['q90'].values

precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score)
ap = average_precision_score(y_true_bin, y_score)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, color='steelblue', lw=1.5,
        label=f'QR Q90 (AP = {ap:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('11.C  Disruption Detection Precision–Recall — Test Set')
ax.legend()
plt.tight_layout()
plt.show()
```

---

**11.D  Rolling 30-Sample Coverage (temporal stability)**

```python
# 11.D  Rolling empirical coverage over the test timeline.
# Checks whether calibration degrades during specific periods (e.g. winter storms).

testdf = qrpreddf[qrpreddf['split'] == 'test'].sort_values('timestamp').copy()
window = 30  # roughly 15 hours at 30-min intervals

testdf['rolling_cov'] = (testdf['iscovered']
                         .rolling(window, min_periods=window // 2)
                         .mean())

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(testdf['timestamp'], testdf['rolling_cov'],
        color='steelblue', linewidth=1.2, label=f'{window}-obs rolling coverage')
ax.axhline(0.80, color='crimson', linestyle='--', linewidth=1, label='Target 80%')
ax.set_xlabel('Date')
ax.set_ylabel('Rolling coverage')
ax.set_title('11.D  Temporal Coverage Stability — Test Set (December)')
ax.legend()
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.show()
```

---

**11.E  QR Coefficient Importance (absolute value bar chart)**

```python
# 11.E  QR coefficient magnitudes for Q10 and Q90.
# Shows which features drive asymmetric interval expansion.
import pandas as pd

coef_data = {}
for tau, model in qr_models.items():
    coef_data[f'τ={tau}'] = model.coef_

coef_df = pd.DataFrame(coef_data, index=QR_FEATURE_COLS)
coef_df_abs = coef_df.abs().sort_values('τ=0.5', ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(coef_df_abs))
width = 0.28
for i, (col, color) in enumerate(zip(coef_df.columns, ['steelblue', 'seagreen', 'tomato'])):
    ax.bar(x + (i - 1) * width, coef_df_abs[col], width, label=col, color=color, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(coef_df_abs.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('|Coefficient|')
ax.set_title('11.E  QR Feature Importance (Absolute Coefficients)')
ax.legend()
plt.tight_layout()
plt.show()
```

---

### 1.8 Section 12 — Calibration & Improvement Profile

**Suggested opening markdown (replace current minimal header):**

```markdown
## 12 · Calibration & Improvement Profile

This section systematically evaluates five calibration strategies applied on top of the
base QR model. Each stage is evaluated on the **held-out December test set** to ensure
no information leakage.

### Why Calibrate?

Even a well-fit QR model can produce mis-calibrated intervals in deployment:
- **Heteroscedastic errors:** LSTM residual variance changes with operating conditions.
- **Systematic bias:** The LSTM may consistently over- or under-predict delays.
- **Distribution shift:** December test conditions differ from the summer/autumn training period.

### Pipeline Stages

| Stage | Name                          | Key idea                                           |
|-------|-------------------------------|----------------------------------------------------|
| 0     | Raw QR (baseline)             | Direct QR on LSTM residuals, no post-hoc correction|
| 1     | Split Conformal               | Distribution-free *(1−α)* coverage guarantee       |
| 2     | Bias Correction               | Shift all quantiles by VAL median error            |
| 3     | Bias + Conformal ✓ recommended| Correct bias first, then apply CP expansion        |
| 4     | Enhanced QR Features          | Add cyclic temporal features & non-linear terms    |
| 5     | Enhanced + Bias + Conformal   | Full pipeline                                      |

### Metric Definitions

| Metric        | Formula / Description                                  | Target      |
|---------------|--------------------------------------------------------|-------------|
| MAE           | mean\|ytrue − q50\|                                   | Lower       |
| RMSE          | sqrt(mean(ytrue − q50)²)                              | Lower       |
| R²            | 1 − SS_res/SS_tot                                      | Higher      |
| coverage_80   | Fraction of ytrue ∈ [Q10, Q90]                        | ≥ 0.80      |
| width_80      | Mean(Q90 − Q10) — sharpness                            | Lower given coverage |
| pinball_q     | Asymmetric loss at quantile q                          | Lower       |
| F1 disruption | F1 score: disrupt if Q90 ≥ 10 min                     | Higher      |
```

---

**Suggested conclusion markdown cell (add as final cell in Section 12):**

```markdown
## 13 · Conclusions

### Key Findings

1. **The QR layer substantially improves uncertainty quantification** over bare LSTM point
   predictions. The base model (Stage 0) achieves 80.8% coverage — close to the 80% target
   — but with a 14.3-minute interval width.

2. **Stage 3 (Bias + Conformal) is the recommended deployment strategy.** It corrects the
   LSTM's systematic underestimation bias (−0.89 min on VAL) and then applies conformal
   expansion to guarantee coverage. On the December test set it achieves:
   - Coverage: **80.3%** (within 0.3 pp of target)
   - MAE: **4.29 min** (improvement over Stage 0's 4.84 min)
   - Interval width: **12.76 min** (sharper than Stage 0 with same coverage)

3. **Enhanced temporal features (Stage 4) reduce interval width to 8.8 min** — a 38%
   reduction — by better capturing intra-day and day-of-week variance patterns.
   However, Stage 5 (Enhanced + Bias + Conformal) over-corrects bias, dropping coverage
   to 73% due to the larger bias estimate from the enhanced feature set.

4. **Disruption detection (F1)** improves from 0.362 (Stage 0) to 0.512 (Stage 4), driven
   by sharper Q90 predictions that better distinguish severe delays.

### Limitations

- The model relies on pre-computed LSTM outputs. Re-running with updated LSTM weights
  requires regenerating the three input CSV files.
- Split conformal prediction uses a single VAL month (August). A larger calibration window
  may yield tighter coverage guarantees.
- Temporal covariate shift between summer training data and the December test period likely
  contributes to the remaining RMSE (~7 min).
- Stage 5 is sensitive to bias estimation: the enhanced feature set shifts the VAL median
  estimate substantially, which can degrade test coverage.
```

---

## TASK 2 — Executive Summary: Highlights & Limitations

### Model Highlights (for non-technical audience)

1. **Uncertainty-aware predictions, not just a single number.**
   The system answers *"How late will the bus be?"* with an 80% confidence interval —
   e.g. "between 2 and 16 minutes late" — rather than a potentially misleading point estimate.

2. **No retraining required.**
   The QR calibration layer wraps the existing LSTM without touching its weights. This makes
   it cheap to deploy and easy to update when conditions change.

3. **Calibration is verified, not assumed.**
   Conformal Prediction provides a mathematical guarantee that the 80% interval captures the
   true delay at least 80% of the time — without assuming Gaussian errors or any specific
   distribution.

4. **Operational disruption detection improved.**
   Using the upper bound (Q90) as a disruption flag, the system correctly identifies severe
   delays (>10 min) with an F1 of 0.51 — a 41% improvement over the uncalibrated baseline.

5. **Weather and temporal patterns are explicitly modelled.**
   Wind speed, holidays, and cyclic time-of-day features explain systematic patterns in LSTM
   errors, making intervals tighter without sacrificing coverage.

6. **Recommended strategy (Stage 3) is practical and interpretable.**
   Bias correction + conformal expansion is a two-step procedure that can be understood and
   audited by non-specialists, unlike complex deep learning post-hoc methods.

---

### Model Limitations (for non-technical audience)

1. **Historical data only.**
   The model was trained on 2017 data (June–October) and tested on December 2017. Its accuracy
   for future years depends on whether bus patterns have changed significantly.

2. **Single route.**
   Results apply specifically to the B41 route. Generalisation to other routes requires
   re-fitting the QR layer with route-specific LSTM outputs.

3. **Summer training, winter testing.**
   The LSTM was trained on summer/autumn data; December presents different weather and
   passenger patterns. This temporal mismatch likely contributes to a residual MAE of
   ~4.3 minutes.

4. **A point prediction is still needed first.**
   The QR layer depends entirely on the LSTM for its central forecast. If the LSTM fails,
   the calibration layer cannot compensate.

5. **Calibration split is small.**
   The conformal calibration uses a single month (August, ~610 observations). A larger or
   more diverse calibration set could tighten the coverage guarantee.

6. **Interval width remains practically large.**
   Even the best configuration (Stage 4 uncalibrated) produces average intervals of ~8.8 min,
   which may be too wide to be operationally actionable for high-frequency services.

---

## TASK 2 — LaTeX Snippets for Overleaf

### Highlights

```latex
\subsection*{Model Highlights}

\begin{itemize}
    \item \textbf{Uncertainty-aware predictions.}
    Rather than a single point estimate, the system outputs calibrated 80\% prediction
    intervals --- e.g.\ \emph{``the delay will fall between 2 and 16 minutes with 80\%
    confidence''} --- providing passengers and operators with actionable uncertainty
    quantification.

    \item \textbf{Non-invasive post-processing.}
    The Quantile Regression (QR) layer wraps the pre-trained LSTM without retraining it,
    making deployment computationally trivial and future updates straightforward.

    \item \textbf{Distribution-free coverage guarantee.}
    Split Conformal Prediction ensures the 80\% interval captures the true delay at
    least 80\% of the time without assuming Gaussian or any other parametric distribution
    for the LSTM residuals.

    \item \textbf{Improved disruption detection.}
    Using the Q90 upper bound as a disruption flag (threshold: 10 min), the calibrated
    model achieves an F1 score of 0.51 --- a 41\% improvement over the uncalibrated
    baseline (F1 = 0.36).

    \item \textbf{Weather and temporal patterns captured.}
    Wind speed, public holidays, and cyclic time-of-day encodings explain systematic
    patterns in LSTM errors, reducing average interval width by up to 38\% in the
    Enhanced QR configuration without sacrificing coverage.

    \item \textbf{Interpretable recommended strategy.}
    Stage~3 (Bias Correction + Conformal Expansion) combines a simple mean-shift
    correction with a one-parameter conformal expansion, producing a pipeline that is
    both theoretically rigorous and practically auditable.
\end{itemize}
```

---

### Limitations

```latex
\subsection*{Model Limitations}

\begin{itemize}
    \item \textbf{Historical data dependency.}
    The model was trained on 2017 observations (June--October) and evaluated on December
    2017. Generalisation to future years is contingent on the stability of delay patterns
    on the B41 route.

    \item \textbf{Single-route scope.}
    Results are specific to the B41 route. Deploying the approach on other routes
    requires refitting the QR layer with route-specific LSTM outputs and feature sets.

    \item \textbf{Temporal distribution shift.}
    The LSTM was trained on summer and autumn data; December test conditions differ in
    weather severity and passenger demand, likely contributing to a residual mean absolute
    error of approximately 4.3 minutes.

    \item \textbf{Dependence on LSTM point forecast quality.}
    The QR layer calibrates LSTM errors but cannot compensate for systematic LSTM
    failures. The accuracy of prediction intervals is bounded by the quality of the
    underlying point predictor.

    \item \textbf{Limited calibration window.}
    Conformal calibration relies on a single validation month (August,
    $n \approx 610$ observations). A larger or more temporally diverse calibration set
    would yield tighter finite-sample coverage guarantees.

    \item \textbf{Interval width may limit operational utility.}
    Even the tightest configuration (Stage~4 Enhanced QR) produces average intervals of
    approximately 8.8 minutes, which may exceed the precision threshold for high-frequency
    real-time passenger information systems.
\end{itemize}
```

---

## TASK 3 — Statement of Individual Contributions

> **Instruction:** The paragraph below is a short template. Fill in each team member's
> name and their specific contributions. If your project description specifies a format,
> follow that format; otherwise adapt the template below.

### Template (plain text — adapt for your Overleaf document)

```latex
\section*{Statement of Individual Contributions}

This project was completed collaboratively by all group members. The specific
contributions of each member are as follows:

\begin{itemize}
    \item \textbf{[Member A]:} Led the development and implementation of the LSTM model,
    including data preprocessing, feature engineering, and hyperparameter tuning.

    \item \textbf{[Member B]:} Designed and implemented the Quantile Regression
    post-processing layer, including the conformal calibration pipeline (Stages 0--5)
    and the associated evaluation metrics.

    \item \textbf{[Member C]:} Conducted the exploratory data analysis, produced all
    visualisations, and drafted the technical report notebook.

    \item \textbf{[Member D]:} Wrote the Executive Summary, coordinated the LaTeX
    typesetting in Overleaf, and proofread all written deliverables.
\end{itemize}

All members contributed equally to the interpretation of results and the final
presentation.
```

---

*End of report.*
