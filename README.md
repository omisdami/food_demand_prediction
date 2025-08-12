# 🍗 Restaurant Inventory Forecasting System

A comprehensive machine learning system that predicts restaurant inventory needs using advanced regression models and time series analysis. This production-ready system achieved **87.1% accuracy** and provides automated inventory recommendations with safety stock calculations.

---

## 🎯 Project Overview

This system forecasts inventory demand for 8 key restaurant items using historical sales data, enabling:

- **Automated inventory planning** with 87% accuracy
- **Safety stock calculations** with 20% buffer recommendations  
- **Production-ready forecasting tool** with manager-friendly reports
- **Comprehensive model comparison** between regression and ARIMA approaches

### 🏆 Key Achievement
Transformed failing models (R² = -0.189) into production-ready forecasts (R² = 0.871) through systematic machine learning methodology.

---

## 📊 Dataset & Target Variables

The dataset contains **106 records** of delivery windows with 8 inventory items:

| Item | Description | Avg. Demand | Complexity |
|------|-------------|-------------|------------|
| `wings` | Chicken wings | ~5,000 units | High volume |
| `tenders` | Chicken tenders | ~700 units | Medium |
| `fries_reg` | Regular fries | ~140 units | Low |
| `fries_large` | Large fries | ~150 units | Low |
| `veggies` | Veggie sticks | ~145 units | Low |
| `dips` | Dip containers | ~500 units | Medium |
| `drinks` | Fountain drinks | ~220 units | Medium |
| `flavours` | Sauce flavors | ~770 units | Medium |

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd food_demand_prediction

# Install dependencies (using uv)
uv sync
```

### Basic Usage
```bash
# Generate forecast using both models (recommended)
uv run restaurant_forecast_tool.py --dataset data/inventory_delivery_forecast_data.csv --model both --days 7

# Use pre-trained models for quick predictions
uv run restaurant_forecast_tool.py --predict --model regression --days 14

# Export forecasts to CSV
uv run restaurant_forecast_tool.py --dataset data/your_data.csv --model regression --save-csv

# Run with anomaly detection monitoring
uv run restaurant_forecast_tool.py --dataset data/inventory_delivery_forecast_data.csv --anomaly-detection
```

### Training Individual Models
```bash
# Train regression models
uv run inventory_forecasting_regression.py data/inventory_delivery_forecast_data.csv

# Train ARIMA models (requires statsmodels)
uv run arima_forecasting.py data/inventory_delivery_forecast_data.csv

# Train anomaly detection model (requires tensorflow)
uv run autoencoder_anomaly_detection.py --train --dataset data/inventory_delivery_forecast_data.csv
```

---

## 🏗️ System Architecture

```
Production Forecasting System/
├── 🤖 Models/
│   ├── regression/              # Winner: 87.1% accuracy
│   │   ├── lasso_model.pkl     # Best model (α=1.0)
│   │   ├── ridge_model.pkl     # Runner-up
│   │   ├── scaler.pkl          # Feature preprocessing
│   │   └── feature_selector_info.pkl
│   ├── arima/                  # Failed: -4.3% accuracy
│   │   ├── arima_*_model.pkl   # 8 individual models
│   │   └── arima_metadata.pkl
│   └── autoencoder/            # Anomaly detection
│       └── best_inventory_autoencoder.h5
├── 🔍 Autoencoder/             # Anomaly detection system
│   ├── inventory_autoencoder_model.h5  # Trained autoencoder
│   ├── inventory_scaler.pkl            # Feature preprocessing
│   ├── anomaly_threshold.json          # Detection thresholds
│   ├── autoencoder_anomaly_detection.py # Detection engine
│   └── Autoencoder.ipynb              # Development notebook
├── 📊 Results/
│   ├── regression/             # Comprehensive analysis
│   │   ├── plots/             # 4 visualization types
│   │   └── manager_reports/   # Business-ready reports
│   ├── arima/                 # Failed model analysis
│   └── autoencoder/           # Anomaly detection results
├── 🎯 Forecasts/
│   ├── final/                 # Production forecasts
│   │   ├── BEST_MODEL_FORECAST.txt
│   │   └── BEST_MODEL_FORECAST.csv
│   └── *.csv                  # Individual model outputs
├── 📁 Data/
│   ├── inventory_delivery_forecast_data.csv  # Main dataset
│   └── *.csv                              # Additional datasets
│
├── restaurant_forecast_tool.py           # Main interface
├── inventory_forecasting_regression.py   # Regression training
└── arima_forecasting.py                 # ARIMA training
```

---

## 📈 Model Performance

### 🏆 Final Results Comparison

| Model | MAE | R² Score | Status | Performance |
|-------|-----|----------|--------|-------------|
| **Lasso Regression** | **23.36** | **0.871** | ✅ **Winner** | 87.1% accuracy |
| Ridge Regression | 24.76 | 0.856 | ✅ Good | 85.6% accuracy |
| Linear Regression | 24.88 | 0.856 | ✅ Good | 85.6% accuracy |
| ElasticNet | 25.00 | 0.864 | ✅ Good | 86.4% accuracy |
| Random Forest | 74.88 | 0.435 | ❌ Overfitted | 43.5% accuracy |
| **ARIMA Average** | **155.32** | **-0.043** | ❌ **Failed** | Worse than baseline |

### 🎯 Key Performance Metrics
- **Best Model**: Lasso Regression (α=1.0)
- **Accuracy**: 87.1% (R² = 0.871)
- **Error Rate**: ±23.36 units average
- **Generalization**: Excellent (CV: 23.09 vs Test: 23.36)
- **Performance Gap**: 6.6x better than ARIMA models

---

## 🔬 Technical Approach

### Feature Engineering
- **Rolling Averages**: 3-day and 7-day windows for trend capture
- **Calendar Features**: Weekend/weekday patterns, monthly seasonality
- **Business Logic**: Item ratios (wings/tenders), total food demand
- **Trend Components**: Days since start for long-term patterns

### Model Selection Process
1. **Data Preprocessing**: StandardScaler + correlation-based feature selection
2. **Cross-Validation**: TimeSeriesSplit with 3 folds for temporal validation
3. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
4. **Model Comparison**: Comprehensive evaluation across 5 algorithms
5. **Production Selection**: Lasso chosen for best generalization

### Why ARIMA Failed
- **Linear Patterns Dominate**: Business logic more important than temporal patterns
- **Small Dataset**: 106 records insufficient for complex time series models
- **External Factor Dependency**: Calendar and ratios outperform historical values
- **Weak Seasonality**: Daily inventory lacks strong seasonal patterns

---

## 📋 Production Features

### 🎯 Manager-Ready Outputs
- **Daily Forecasts**: 7-day inventory predictions with safety stock
- **Weekly Summaries**: Total demand and recommended stock levels
- **Business Insights**: Weekend patterns, peak days, top items
- **Multiple Formats**: Text reports, CSV exports, console display

### 🛠️ Tool Capabilities
- **Multi-Model Support**: Automatic regression vs ARIMA comparison
- **Flexible Input**: Any dataset with proper column structure
- **Training Modes**: Fresh training or pre-trained model usage
- **Export Options**: CSV, text reports, manager summaries
- **Safety Calculations**: 20% buffer for inventory planning
- **Anomaly Detection**: AI-powered unusual pattern detection

### 📊 Visualization Suite
- **Model Performance**: MAE, R², cross-validation comparisons
- **Time Series Plots**: Actual vs predicted for all items
- **Residual Analysis**: Model assumption validation
- **Error Distribution**: Performance consistency across items
- **Anomaly Analysis**: Unusual inventory pattern identification

---

## 💼 Business Impact

### 🎯 Operational Benefits
- **87% Forecast Accuracy**: Reliable inventory planning
- **±23 Unit Precision**: Actionable prediction errors
- **Automated Recommendations**: Reduce manual planning time
- **Safety Stock Integration**: Prevent stockouts with 20% buffer
- **Weekend Intelligence**: Automatic demand pattern recognition
- **Anomaly Detection**: Early warning system for unusual patterns

### 📈 Cost Savings
- **Reduced Waste**: Prevent overordering with accurate forecasts
- **Stockout Prevention**: Safety stock calculations minimize shortages
- **Labor Efficiency**: Automated planning reduces manual effort
- **Data-Driven Decisions**: Replace intuition with statistical models
- **Quality Assurance**: Detect data errors and supply chain disruptions

---

## 🔧 Advanced Usage

### Custom Dataset Training
```bash
# Train with your own data (must have same column structure)
uv run restaurant_forecast_tool.py --dataset path/to/your/data.csv --model both

# Specify forecast horizon
uv run restaurant_forecast_tool.py --dataset data.csv --days 14 --save-csv
```

### Model Comparison
```bash
# Compare regression vs ARIMA performance
uv run restaurant_forecast_tool.py --dataset data.csv --model both --save-csv

# Individual model outputs saved to:
# - forecasts/regression_forecast.csv
# - forecasts/arima_forecast.csv
# - forecasts/final/RESTAURANT_TOOL_FORECAST.csv
```

### Anomaly Detection
```bash
# Run anomaly detection on historical data
uv run restaurant_forecast_tool.py --dataset data.csv --anomaly-detection

# Train new anomaly detection model
uv run restaurant_forecast_tool.py --dataset data.csv --train-anomaly

# Adjust sensitivity levels
uv run restaurant_forecast_tool.py --dataset data.csv --anomaly-detection --anomaly-threshold sensitive
```

### Production Deployment
```bash
# Use pre-trained models for daily forecasting
uv run restaurant_forecast_tool.py --predict --model regression --days 7

# Daily forecasting with anomaly monitoring
uv run restaurant_forecast_tool.py --predict --model regression --anomaly-detection --days 7

# Automated daily forecasting (cron job example)
0 6 * * * cd /path/to/project && uv run restaurant_forecast_tool.py --predict --model regression --save-csv
```

---

## 📚 Documentation

- **[Model Improvement Journey](model_improvement_journey.md)**: Complete development process from failure to success
- **[Technical Documentation](results/regression/model_performance_detailed.txt)**: Detailed model analysis
- **[Manager Reports](forecasts/final/)**: Production-ready forecasts

---

## 🛠️ Technology Stack

### Core ML Stack
- **Python 3.10+**: Primary language
- **scikit-learn**: Regression models, preprocessing, validation
- **statsmodels**: ARIMA/SARIMA time series models
- **tensorflow/keras**: Autoencoder anomaly detection
- **pandas/numpy**: Data manipulation and analysis
- **matplotlib/seaborn**: Visualization and plotting

### Production Tools
- **joblib**: Model serialization and loading
- **argparse**: Command-line interface
- **datetime**: Time series handling
- **uv**: Dependency management and execution

### Development Tools
- **GridSearchCV**: Hyperparameter optimization
- **TimeSeriesSplit**: Temporal cross-validation
- **StandardScaler**: Feature preprocessing
- **Correlation-based selection**: Feature engineering
- **Optuna**: Autoencoder hyperparameter optimization

---

## 🚀 Future Enhancements

### Short-term Improvements
- **Ensemble Methods**: Combine top 3 regression models
- **Confidence Intervals**: Prediction uncertainty quantification
- **Real-time Updates**: Daily model retraining pipeline
- **Enhanced Anomaly Detection**: Real-time alerts and notifications

### Medium-term Features
- **External Data**: Weather, holidays, promotional events
- **Advanced Features**: Interaction terms, polynomial features
- **Multi-location Support**: Scale to multiple restaurants
- **API Development**: REST API for system integration
- **Anomaly Root Cause Analysis**: Identify why patterns are unusual

### Long-term Vision
- **Deep Learning**: LSTM/GRU for complex patterns (with more data)
- **Real-time Integration**: POS system connectivity
- **Advanced Analytics**: Demand driver analysis
- **AutoML Pipeline**: Automated model selection and tuning
- **Predictive Anomaly Detection**: Forecast unusual patterns before they occur

---

**Key Success Factors:**
- Understanding data characteristics over algorithm complexity
- Proper validation methodology with time series considerations  
- Business logic integration in feature engineering
- Production-ready tooling with manager-friendly outputs

---
