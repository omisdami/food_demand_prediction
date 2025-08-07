
# 🧠 Autoencoder-Based Inventory Anomaly Detection: Final Findings

## 📦 Dataset Overview
- **Total Delivery Windows**: 102
- **Date Range**: Jan 13, 2024 – Dec 30, 2024
- **Average Inventory per Delivery**: 7,510 items

## 🧠 Model Performance
- Trained on **26 inventory features** using an autoencoder.
- **Loss convergence** was smooth; no signs of overfitting.
- **Optuna tuning** achieved low reconstruction loss (~0.04).

## 🚨 Anomaly Detection Results
- **Anomalies Detected**: 18 out of 102 (**17.6%**)
- **Anomaly Threshold**: 0.0419 (balanced)
- **Most Anomalous Delivery**: 2024-12-28 (error: 0.8057)
- **Average Reconstruction Error**: 0.03956

## 📊 Inventory Category Insights
- **Lower inventory during anomalies**: avg 7,213 vs 7,510 overall
- **Most impacted categories**:
  - Tenders: -20%
  - Fries Large: -10%
  - Veggies: -10%
  - Dips: -8%
- **Stable categories**: Wings, Drinks, Flavours (no significant drop)

## 📅 Seasonal Patterns
- **Winter**: Highest anomaly rate (**26.1%**)
- **Fall/Spring/Summer**: Lower rates (11.5%–19.2%)
- Winter had the **lowest average inventory** and **widest variability**.

## 📐 Ratio Anomalies
- Anomalies showed changes in:
  - Wings ratio
  - Protein ratio
  - Sides ratio
- Indicates **distributional shifts**, not just volume anomalies

## ✅ Summary
This autoencoder effectively identified both **volume-based** and **structural** anomalies in inventory patterns, with **clear seasonal risk patterns** and **category-specific insights**. The model and scaler are ready for integration into real-time delivery monitoring.
