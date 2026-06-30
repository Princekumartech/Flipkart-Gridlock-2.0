# 🚦 Flipkart Gridlock 2.0 — Traffic Demand Prediction

> **Hackathon:** Flipkart Gridlock 2.0 | **Platform:** HackerEarth  
> **Problem:** Predict traffic demand across Bengaluru road segments  
> **Metric:** `max(0, 100 × R²)` | **Score:** 88.57

---

## 📌 Problem Statement

Bengaluru loses billions of hours annually to traffic congestion. This hackathon challenged participants to build AI models that predict **traffic demand** at specific road locations and time slots — using real-world data from Bengaluru Traffic Police (ASTraM) and MapMyIndia.

**Task:** Given geolocation, road attributes, weather, and timestamp — predict normalized traffic demand (0–1) at that location.

---

## 📂 Dataset

| File | Shape | Description |
|---|---|---|
| `train.csv` | 77,299 × 11 | Training data with demand labels |
| `test.csv` | 41,778 × 10 | Test data for prediction |
| `sample_submission.csv` | 5 × 2 | Submission format |

**Key columns:** `geohash`, `day`, `timestamp`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`, `demand`

---

## 💡 Key Insight

After thorough analysis, the strongest predictor was a **direct lookup table** — not a complex ML model.

- **Test set** = Day 49, hours 02:15–13:45
- **Day 49 in train** only has hours 00:00–02:00
- Therefore, for all test time slots: `full_train geo×slot stats == day48 geo×slot stats`

This means `geo_ts_mean` (average demand per geohash × 15-min time slot) computed from the full training set captures **both location baseline AND time-of-day pattern** perfectly for the test distribution.

> Adding LightGBM on top consistently **hurt** predictions (R² dropped from 0.883 → 0.793) because the model overfit day48's 69k rows and failed to generalise to the test distribution.

---

## 🔧 Approach

### Feature Engineering
- **`geo_ts_mean`** — Mean demand per (geohash × 15-min time slot) from full train → **Primary prediction**
- **Interpolated pivot** — Linear interpolation across time slots per geohash to fill missing slots
- **Fallback chain** for the 11% of test rows with no exact geo×slot match:
  ```
  geo×slot (interpolated) → prefix4×slot → prefix3×slot → geo×hour → geo_mean → global_mean
  ```

### Why No Heavy Model?
| Approach | Val R² (Day49) |
|---|---|
| Direct `geo_ts_mean` | **0.883** ✅ |
| LightGBM (conservative) | 0.793 |
| LightGBM (full features) | 0.754 |
| Blend 70/30 (direct/LGB) | 0.862 |

The direct lookup outperforms all model-based approaches on proper time-split validation.

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation R² (Day48 → Day49) | 0.883 |
| Leaderboard Score | **88.57 / 100** |
| Evaluation Metric | `max(0, 100 × R²)` |

---

## 🗂️ Repository Structure

```
Flipkart-Gridlock-2.0/
│
├── solution_best.ipynb       # Full solution notebook with analysis
├── sub_direct_geo_ts.csv     # Final submission file
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange?logo=numpy)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-red?logo=scikit-learn)

---

## 🚀 How to Reproduce

```bash
# Clone the repo
git clone https://github.com/Princekumartech/Flipkart-Gridlock-2.0.git
cd Flipkart-Gridlock-2.0

# Install dependencies
pip install pandas numpy lightgbm scikit-learn

# Run the notebook
jupyter notebook solution_best.ipynb
```

> Download the dataset from the [HackerEarth hackathon page](https://www.hackerearth.com/) and place `train.csv`, `test.csv`, and `sample_submission.csv` in the working directory.

---

## 👤 Author

**Prince Kumar**  
B.Tech — Metallurgical & Materials Engineering (Minor: CS) | NIT Warangal  
📎 [GitHub](https://github.com/Princekumartech)

---

*Built for Flipkart Gridlock Hackathon 2.0 — an AI/ML challenge to help Bengaluru move smarter.*
