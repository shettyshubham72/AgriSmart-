# AgriSmart-
 AI Crop Recommendation System for Precision Agriculture
# AgriSmart — AI Crop Recommendation & Advisory System for Precision Agriculture

A machine-learning system that recommends the most suitable crop for a farmer's
field from soil and climate inputs, then translates that prediction into
practical advice: fertilizer schedule, irrigation plan, yield estimate and
projected profit.

Built as a final-year B.E. project at **VCET, Mumbai University (2025–26)**.

**Team:** Ved Raut · Shubham Shetty · Balaji Rode
**Supervisor:** Prof. Leena Raut

---

## Table of Contents

- [What it does](#what-it-does)
- [Results](#results)
- [How it works](#how-it-works)
- [Dataset](#dataset)
- [Model architecture](#model-architecture)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Regenerating the results](#regenerating-the-results)
- [Repo layout](#repo-layout)
- [Limitations](#limitations)
- [Credits](#credits)

---

## What it does

The system takes **seven parameters** a farmer can realistically measure —
nitrogen, phosphorus, potassium, soil pH, temperature, humidity and rainfall —
plus the selected district and season. From these it:

1. **Derives 20 model features**, enriching the raw inputs with district-level
   agro-climatic context and physically-grounded derived quantities
   (Penman-Monteith evapotranspiration, growing degree days, NPK ratio, water
   stress index).
2. **Predicts the best crop** using a weighted ensemble of a custom
   Improved Deep Belief Network and a Random Forest, returning a calibrated
   confidence score and top-5 alternatives.
3. **Explains the prediction** two ways — a feature-importance ranking showing
   which inputs drove the decision, and a *counterfactual* engine that
   identifies the smallest change to soil or climate that would flip the
   recommendation.
4. **Generates agronomic advisory** — FAO-56 based irrigation scheduling,
   an ICAR-IIRR site-specific nutrient management (SSNM) fertilizer
   prescription staged across the crop cycle, a yield estimate, and a
   profit/ROI projection.

The web interface is **bilingual (English / Marathi)** and covers all
**35 districts of Maharashtra**, with an offline-capable district map.

---

## Results

Model comparison on a stratified 70/15/15 train/validation/test split
(16 crop classes, 1,500 test samples):

| Model | Accuracy (%) | Macro F1 |
|---|---|---|
| Decision Tree | 87.87 | 0.8786 |
| SVM (RBF) | 88.33 | 0.8832 |
| ANN (MLP, 256-128-64) | 89.67 | 0.8963 |
| **Random Forest** (300 trees) | **91.67** | 0.9167 |
| **IDBN (this work)** | 90.27 | 0.9021 |
| **IDBN + RF Ensemble** ← deployed | **91.47** | 0.9141 |

- The standalone IDBN reaches **90.27%**; the ensemble adds **+1.20 points**.
- Evaluated under **5-fold stratified cross-validation** for robustness.
- The auxiliary yield-regression head is trained jointly, so a single forward
  pass produces both a crop label and a yield estimate.

Evaluation figures — confusion matrix, training curves, cross-validation
spread, feature importance, class distribution and correlation heatmap — are
committed in [`training/outputs/`](training/outputs).

> **A note on the numbers.** The ensemble is not the single highest-scoring
> model here; the plain Random Forest edges it out by 0.20 points. We deployed
> the ensemble anyway because it is markedly more stable across the
> cross-validation folds and degrades more gracefully on out-of-district
> inputs. We would rather report this honestly than tune the split until the
> ensemble won.

---

## How it works

```
                    ┌──────────────────────────────┐
   7 raw inputs ───▶│  Feature derivation          │
   + district       │  · district agro-climata     │
   + season         │  · Penman-Monteith ET        │
                    │  · growing degree days       │
                    │  · water/heat stress index   │
                    └───────────────┬──────────────┘
                                    ▼  20 features
                    ┌──────────────────────────────┐
                    │  StandardScaler              │
                    └───────────────┬──────────────┘
                                    ▼
             ┌──────────────────────┴─────────────────────┐
             ▼                                            ▼
   ┌───────────────────┐                       ┌──────────────────┐
   │  IDBN (NumPy)     │  0.6 ×               │  Random Forest   │ 0.4 ×
   │  60% weight        │──────▶ combined ─────▶│  300 trees       │──────┤
   └───────────────────┘   probability        └──────────────────┘      │
             │                                        │                │
             └───────────────▶ prediction ◀──────────┘────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  Explainability: feature importance           │
        │  Counterfactual: minimal change to flip       │
        │  Advisory: SSNM fertilizer · FAO-56 irrigation│
        │             yield estimate · profit/ROI       │
        └───────────────────────────────────────────────┘
```

### Feature selection

The dataset begins with **42 candidate predictors**, many of which are
collinear (e.g. mean/max/min temperature; field capacity and available water).
We apply a two-stage hybrid:

1. **Correlation filter** — drop any feature with |r| > 0.95 against another.
2. **Backward Elimination Feature Selection (BEFS)** — iteratively remove the
   lowest-importance feature, re-ranking by Random Forest importance, until 20
   remain.

Implemented in [`training/utils/feature_selection.py`](training/utils/feature_selection.py).

---

## Dataset

| Property | Value |
|---|---|
| Samples | 10,000 |
| Raw candidate features | 42 |
| Selected features | 20 |
| Crop classes | 16 |
| Targets | `crop` (classification), `yield_kg_per_ha` (regression) |

Feature groups: soil chemistry (N/P/K, pH, EC, organic carbon, Ca, Mg, S, Zn,
Fe, Mn, Cu, B, CEC), soil physics (moisture, bulk density, porosity, field
capacity, wilting point, available water, texture), weather (temperature,
rainfall, humidity, wind speed, solar radiation, sunshine hours, cloud cover),
irrigation (frequency, amount, drainage, water table depth), and geography
(slope, altitude, agro-ecological zone).

Classes are the principal Maharashtra/Indian field crops — Rice, Wheat, Maize,
Cotton, Sugarcane, Soybean, Groundnut, Barley, Chickpea, Lentil, Mustard,
Sunflower, Jowar, Bajra, Tur and Onion.

### ⚠️ Provenance — please read

**The dataset in this repository is synthetic.** It is generated by
[`training/data/generate_dataset.py`](training/data/generate_dataset.py), which
samples from per-crop ideal ranges drawn from published agronomic sources
(ICAR-IIRR, TNAU Agritech Portal, CRIDA agro-climatic guidelines) and derives
the physical variables using standard equations — Penman-Monteith
evapotranspiration, growing degree days, FAO-56 water-balance terms.

We generated it rather than collecting it because **granularity-level soil
health data for individual Indian districts is not publicly available**. The
simulator is physically grounded and calibrated to published ranges, but the
reported accuracies measure performance on *this simulation*, not on
collected field data.

**This is the main limitation of the project, and we state it plainly rather
than let a reader assume otherwise.** Swapping in real observations requires
only replacing `generate_dataset.py`; the rest of the pipeline is unchanged.

---

## Model architecture

### Improved Deep Belief Network (IDBN)

Implemented from scratch in NumPy — no TensorFlow, PyTorch or Keras — in
[`app/models/idbn.py`](app/models/idbn.py).

```
Input (20)
   │
   ▼
┌──────────────────┐   Gaussian-Bernoulli RBM
│  GRBM  20 → 256  │   continuous visible units,
└──────────────────┘   Bernoulli hidden units
   │
   ▼
┌──────────────────┐
│  RBM  256 → 128  │   Bernoulli-Bernoulli
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  RBM  128 → 64   │   Bernoulli-Bernoulli
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  Dense 64 → 16   │   softmax  (crop class)
│  Yield head 64→1 │   linear    (kg/ha)
└──────────────────┘
```

**Why GRBM as the first layer?** All 20 inputs are continuous agronomic
measurements, so the input layer must model Gaussian visible-unit statistics
rather than Bernoulli ones.

**Training** is two-phase:

1. **Unsupervised pre-training** — layers trained bottom-up with CD-k
   (`pt_epochs=50`).
2. **Supervised fine-tuning** — full network trained end-to-end
   (`epochs=100`, batch size 32).

**Ranger optimizer** (RAdam + Lookahead, k=6) rather than plain Adam —
rectified Adam stabilizes the early pre-training phase, and the Lookahead
wrapper damps the oscillation typical of RBM CD-k updates.

Also applied: **dropout 0.4** and **Gaussian noise augmentation** (σ = 0.05,
training set doubled), both of which measurably reduced overfitting across
the 42→20 feature space.

The yield head is trained jointly with the classifier — one network, two
losses, so a single forward pass serves both the app and the API.

### Ensemble

Soft-voting blend of the IDBN and a Random Forest:

```
P(crop | x) = 0.6 · P_IDBN(x) + 0.4 · P_RF(x)
```

---

## Project structure

```
Agrismart/
├── app/                        Flask application
│   ├── app.py                  Routes, model loading, advisory logic
│   ├── app_standalone.py       Entry point
│   ├── download_map.py         One-off GeoJSON fetcher
│   ├── templates/              index.html, compare.html
│   ├── static/                 district_data.json, maharashtra.geojson
│   └── models/                 idbn.py + loaded artifacts
│
├── training/                   Model development pipeline
│   ├── train.py                7-stage end-to-end training run
│   ├── predict.py              Inference + SSNM advisory (CLI/library)
│   ├── data/
│   │   └── generate_dataset.py Synthetic data generator
│   ├── utils/
│   │   └── feature_selection.py  Correlation filter + BEFS
│   ├── models/
│   │   ├── idbn.py             IDBN reference implementation
│   │   └── baselines.py        DT / SVM / ANN / RF baselines
│   └── outputs/                8 evaluation figures + metrics CSV
│
├── docs/
│   ├── ARCHITECTURE.md         Deeper design detail
│   └── RESULTS.md              Full metrics, CV, ablations
├── download_models.py          Fetches trained artifacts from Releases
└── requirements.txt
```

---

## Installation

**Prerequisites:** Python 3.10 or later.

```bash
git clone https://github.com/<your-username>/Agrismart.git
cd Agrismart

python -m venv .venv
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Fetch the trained models

Trained artifacts are hosted as a GitHub Release rather than committed,
because the Random Forest weights exceed GitHub's 100 MB per-file limit.

```bash
python download_models.py
```

This downloads the artifacts and places them in the correct `models/`
directories. You can also download them manually from
**Releases → `v1.0-models`** if you prefer.

---

## Usage

```bash
python app/app_standalone.py
```

Then open **<http://127.0.0.1:5000>**.

Enter your soil and climate readings, pick your district and season, and press
*Recommend*. You will get the crop, confidence, top-5 alternatives, an
explanation of which factors mattered, the counterfactual ("what would change
the answer"), fertilizer and irrigation schedules, and a profit estimate.

### Run inference directly

```bash
python training/predict.py
```

```python
from predict import predict

result = predict({
    'nitrogen': 90, 'phosphorus': 45, 'potassium': 43,
    'soil_pH': 6.3, 'temperature': 26, 'rainfall': 240, 'humidity': 83,
    'evapotranspiration': 5.5, 'soil_moisture': 65,
    'water_stress_index': 0.80, 'npk_ratio': 59.3,
})

print(result['recommended_crop'])            # 'Rice'
print(result['confidence'])                 # '84.2%'
print(result['estimated_yield_kg_per_ha'])   # 5620.0
print(result['ssnm_advisory'])              # ICAR-based adjusted doses
```

### Optional: local LLM chat

The `/chat` endpoint uses a local [Ollama](https://ollama.com) model when one
is running, and falls back to a built-in rule-based agronomist when it is not.

```bash
ollama pull llama3.2
```

Everything works without it — the fallback is always available.

---

## Regenerating the results

To rebuild the dataset, re-run feature selection, retrain the IDBN, rebuild the
ensemble and regenerate every figure:

```bash
cd training
python train.py
```

This takes a while (roughly 20–40 minutes depending on CPU) and rewrites
`training/outputs/` and `training/models/`. Expect small run-to-run variation
in the reported figures even with `random_state=42` fixed, since NumPy BLAS
reduction order is not deterministic across machines.

---

## Limitations

- **The dataset is synthetic** (see [Provenance](#-provenance--please-read)).
  Reported accuracies reflect the simulation, not field-collected data.
- Yield and ROI figures are **estimates from published per-crop averages**,
  not a validated economic model. Market prices and input costs are
  Maharashtra reference values and go stale.
- Advisory thresholds follow ICAR/TNAU guidelines for general Maharashtra
  conditions; **they are not a substitute for an agronomist's assessment**,
  particularly for local varieties, precision agriculture or unusual soils.
- The model covers 16 crops at the state level. It does not account for
  hybrid/specific varieties, cropping history, or multi-season rotations.
- The Random Forest baseline marginally outperforms the deployed ensemble on
  the held-out test set (see [Results](#results)).

---

## Credits

- **ICAR-IIRR** — site-specific nutrient management guidelines, fertilizer
  dose schedules
- **FAO-56** — Penman-Monteith evapotranspiration, crop coefficients
- **TNAU Agritech Portal** and **CRIDA** — agro-climatic zone ranges
- **datameet/maps** — Maharashtra district boundaries (GeoJSON)
- **[Leaflet](https://leafletjs.com/)**, **[Chart.js](https://www.chartjs.org/)**,
  **[Flask](https://flask.palletsprojects.com/)**,
  **[scikit-learn](https://scikit-learn.org/)**,
  **[NumPy](https://numpy.org/)**, **[Pandas](https://pandas.pydata.org/)**

---

## License

MIT 

The Maharashtra district boundary data is sourced from
[datameet/maps](https://github.com/datameet/maps); please observe its
license terms.
