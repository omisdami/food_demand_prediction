# 📦 Inventory Forecasting for Restaurant Supply Optimization

This project aims to develop a machine learning system that forecasts inventory needs for a restaurant, optimizing purchase orders and reducing waste. The dataset is based on aggregated sales orders prior to each delivery window, which occurs **twice a week: Monday and Saturday**.

---

## 🧠 Project Objective

The main goal is to **predict how much of each inventory item will be needed before the next scheduled delivery**, using historical data and advanced machine learning techniques. This enables restaurants to:

- Prevent overstocking or understocking
- Automate ordering decisions
- Improve operational efficiency

---

## 🗃️ Dataset Overview

Each row in the dataset corresponds to a **delivery window**, capturing total demand for that period.  
Below are the key columns:

| Column          | Description                                       |
| --------------- | ------------------------------------------------- |
| `delivery_date` | Date of inventory delivery (Monday/Saturday only) |
| `wings`         | Number of wings consumed (target variable)        |
| `tenders`       | Chicken tenders consumed                          |
| `fries_reg`     | Regular fries servings                            |
| `fries_large`   | Large fries servings                              |
| `veggies`       | Veggie stick servings                             |
| `dips`          | Dip cups used                                     |
| `drinks`        | Fountain drinks served                            |
| `flavours`      | Sauce flavor servings                             |

---

## 🧩 Project Structure & Roles

### 👤 Bikash – Dataset Generation

- Generate synthetic data if real data is unavailable
- Simulate realistic patterns for sales, weather, traffic, and events

---

### 👥 Bikash & Callum – Data Collection & Preprocessing

- Collect sales, foot traffic, weather, and event data
- Clean, normalize, and merge datasets
- Tools: `pandas`, `numpy`, `OpenWeather API`, `Google Places API`, Jupyter
- Techniques: Missing value imputation, MinMaxScaler/StandardScaler, feature engineering (e.g., weekday, seasonality)

---

### 📈 Callum – Time Series Forecasting (ARIMA)

- Build and tune ARIMA/SARIMA models for demand prediction
- Tools: `statsmodels`, `pmdarima`, `matplotlib`, `seaborn`
- Metrics: RMSE, MAE

---

### 📊 Friba – Regression Modeling (External Factors)

- Use regression to model impact of weather, traffic, events
- Combine predictions with ARIMA using ensemble/weighted average
- Tools: `scikit-learn`, `GridSearchCV`
- Algorithms: Linear Regression, Ridge/Lasso

---

### 🧠 Gavriel – Autoencoder for Anomaly Detection

- Build a deep autoencoder to learn normal demand patterns
- Detect unusual demand spikes/drops using reconstruction error
- Tools: `TensorFlow/Keras`, `Optuna`, `pandas`, `numpy`

---

### 🔁 Gavriel – Model Integration & Post-Processing

- Combine ARIMA + Regression outputs
- Overlay anomaly alerts from autoencoder
- Translate predictions into purchase orders using reorder logic
- Techniques: Ensemble models, safety stock calculation, rounding logic

---

### 🌐 Bikash & Dami – Backend, Deployment & Dashboard

- Build Flask APIs for model inference
- Visualize outputs with Dash/Streamlit
- Containerize app with Docker, optionally deploy with Gunicorn/Nginx
- Tools: `Flask`, `Plotly Dash` or `Streamlit`, `Docker`

---

## 🛠️ Stack Overview

- **Languages**: Python
- **ML Libraries**: scikit-learn, statsmodels, pmdarima, TensorFlow/Keras
- **Data Tools**: pandas, numpy, Jupyter
- **Hyperparameter Tuning**: Optuna, GridSearchCV
- **Visualization**: matplotlib, seaborn, Plotly, Streamlit
- **APIs**: OpenWeather, Google Places
- **Deployment**: Flask, Docker, Gunicorn, Nginx

---

## 📦 Output

- Forecasted ingredient quantities for upcoming delivery dates
- Anomaly detection alerts for unusual demand patterns
- Purchase order recommendations with safety buffers
- Interactive dashboard for monitoring and control

---
