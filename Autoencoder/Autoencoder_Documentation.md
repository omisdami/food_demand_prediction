# Enhanced Inventory Autoencoder: Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Loading & Preprocessing](#data-loading--preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Architecture](#model-architecture)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Training Process](#training-process)
7. [Anomaly Detection](#anomaly-detection)
8. [Visualization & Analysis](#visualization--analysis)
9. [Model Deployment](#model-deployment)
10. [Technical Decisions Explained](#technical-decisions-explained)

---

## Project Overview

### What We're Building
An **autoencoder-based anomaly detection system** for restaurant inventory delivery windows. The system identifies unusual patterns in inventory ordering that could indicate:
- Supply chain disruptions
- Demand forecasting errors
- Operational inefficiencies
- Seasonal anomalies
- Inventory management issues

### Why Autoencoders?
**Autoencoders** are neural networks trained to reconstruct their input. They learn to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding). For anomaly detection:
- **Normal patterns** are reconstructed accurately (low reconstruction error)
- **Anomalous patterns** are reconstructed poorly (high reconstruction error)
- **Unsupervised learning** - no need for labeled anomaly examples

---

## Data Loading & Preprocessing

### Cell 1: Data Loading and Initial Feature Engineering

```python
# Load the new inventory delivery forecast data
df = pd.read_csv('data/inventory_delivery_forecast_data.csv')
```

**Why This Dataset?**
- Contains actual inventory categories: wings, tenders, fries, veggies, dips, drinks, flavours
- Reflects real business delivery schedule (Monday/Saturday)
- Includes temporal patterns crucial for anomaly detection

**Temporal Feature Extraction:**
```python
df['month'] = df['delivery_date'].dt.month
df['day_of_week'] = df['delivery_date'].dt.dayofweek
df['is_monday'] = (df['delivery_date'].dt.dayofweek == 0).astype(int)
df['is_saturday'] = (df['delivery_date'].dt.dayofweek == 5).astype(int)
```

**Why These Features?**
- **Seasonality**: Restaurant demand varies by month/season
- **Weekly Patterns**: Different inventory needs for Monday vs Saturday deliveries
- **Business Logic**: Delivery schedule is fixed to these two days

**Inventory Ratios:**
```python
df['protein_ratio'] = (df['wings'] + df['tenders']) / df['total_inventory']
df['sides_ratio'] = (df['fries_reg'] + df['fries_large'] + df['veggies']) / df['total_inventory']
```

**Why Ratios Matter?**
- **Scale Independence**: Ratios remain consistent regardless of total volume
- **Pattern Recognition**: Unusual proportions indicate operational changes
- **Business Insight**: Helps identify which categories are driving anomalies

---

## Feature Engineering

### Cell 8: Lag and Rolling Features

**Lag Features:**
```python
for item in ['wings', 'tenders', 'total_inventory']:
    df_sorted[f'{item}_lag1'] = df_sorted[item].shift(1)
    df_sorted[f'{item}_lag2'] = df_sorted[item].shift(2)
```

**Why Lag Features?**
- **Temporal Dependencies**: Previous deliveries influence current orders
- **Trend Detection**: Helps identify gradual changes in ordering patterns
- **Business Context**: Inventory managers consider historical data

**Rolling Statistics:**
```python
df_sorted['total_inventory_rolling_mean'] = df_sorted['total_inventory'].shift(1).rolling(window=4).mean()
df_sorted['wings_rolling_std'] = df_sorted['wings'].shift(1).rolling(window=4).std()
```

**Why Rolling Statistics?**
- **Trend Analysis**: Rolling means smooth out noise and show trends
- **Volatility Measurement**: Rolling standard deviation captures variability
- **Window Size (4)**: Represents 2 weeks of deliveries (practical business cycle)

---

## Model Architecture

### Cell 10: Preprocessing Choice - RobustScaler

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

**Why RobustScaler over MinMaxScaler or StandardScaler?**
- **Outlier Resistance**: Uses median and IQR instead of mean and std
- **Inventory Data**: Restaurant inventory often has outliers (special events, holidays)
- **Better Generalization**: Less affected by extreme values during training

### Cell 12: Enhanced Autoencoder Architecture

**Architecture Design:**
```python
def create_improved_autoencoder(trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)
    encoder_units = trial.suggest_int('encoder_units', 32, 128, step=16)
    latent_dim = trial.suggest_int('latent_dim', 8, 24)
```

**Why These Layer Sizes?**
- **2-4 Layers**: Deep enough to learn complex patterns, shallow enough to avoid overfitting
- **32-128 Units**: Sufficient capacity for inventory pattern complexity
- **8-24 Latent Dimensions**: Compressed representation that retains essential information

**Regularization Techniques:**
```python
dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
l2_reg = trial.suggest_float('l2_regularization', 1e-6, 1e-3, log=True)
```

**Why Dropout and L2 Regularization?**
- **Dropout (0.1-0.4)**: Prevents overfitting by randomly setting neurons to zero
- **L2 Regularization**: Penalizes large weights, promotes smoother decision boundaries
- **Combined Effect**: Better generalization to unseen inventory patterns

**Activation Function Choice:**
```python
activation = trial.suggest_categorical('activation', ['relu', 'elu', 'swish'])
```

**Why These Activations?**
- **ReLU**: Fast, simple, works well for most cases
- **ELU**: Smooth, handles negative values better, reduces vanishing gradient
- **Swish**: Self-gated, smooth, often performs better than ReLU

**Loss Function:**
```python
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
```

**Why Huber Loss?**
- **Robust to Outliers**: Less sensitive to extreme values than MSE
- **Smooth**: Differentiable everywhere (unlike MAE)
- **Inventory Context**: Handles occasional extreme orders without skewing training

---

## Hyperparameter Optimization

### Cell 14: Optuna Optimization

```python
from optuna.samplers import TPESampler
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(enhanced_objective, n_trials=75)
```

**Why Optuna?**
- **Tree-structured Parzen Estimator (TPE)**: More efficient than grid search
- **Pruning**: Stops unpromising trials early
- **Parallel Execution**: Can run multiple trials simultaneously

**Why 75 Trials?**
- **Balance**: Enough exploration without excessive computation time
- **Convergence**: Typically sufficient for finding good hyperparameters
- **Practical**: Reasonable training time for production use

**Enhanced Objective Function:**
```python
def enhanced_objective(trial):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
```

**Why These Callbacks?**
- **Early Stopping**: Prevents overfitting, saves training time
- **Learning Rate Reduction**: Allows fine-tuning when progress stagnates
- **Patience Values**: Balanced between premature stopping and excessive training

---

## Training Process

### Cell 17: Enhanced Training Setup

**Advanced Callbacks:**
```python
checkpoint = ModelCheckpoint('best_inventory_autoencoder.keras', 
                           monitor='val_loss', save_best_only=True)
```

**Why Model Checkpointing?**
- **Best Model Recovery**: Saves the best model during training
- **Training Interruption**: Protects against training failures
- **Modern Format (.keras)**: TensorFlow's recommended format

**Optimizer Configuration:**
```python
optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
```

**Why Adam with Gradient Clipping?**
- **Adam**: Adaptive learning rates, handles sparse gradients well
- **Gradient Clipping (1.0)**: Prevents exploding gradients
- **Inventory Context**: Stable training for variable inventory patterns

---

## Anomaly Detection

### Cell 19: Multi-Level Thresholds

```python
thresholds = {
    'conservative': np.percentile(train_reconstruction_errors, 99),
    'balanced': np.percentile(train_reconstruction_errors, 95),
    'sensitive': np.percentile(train_reconstruction_errors, 90)
}
```

**Why Multiple Thresholds?**
- **Business Flexibility**: Different sensitivity levels for different use cases
- **Conservative (99th percentile)**: Critical alerts only, minimal false positives
- **Balanced (95th percentile)**: Standard operations, good precision-recall balance
- **Sensitive (90th percentile)**: Early warning system, catches subtle anomalies

**Reconstruction Error Calculation:**
```python
reconstruction_errors = np.mean(np.square(X_scaled - predictions), axis=1)
```

**Why Mean Squared Error?**
- **Euclidean Distance**: Measures overall pattern deviation
- **Feature Equality**: All inventory features contribute equally
- **Interpretability**: Higher values clearly indicate more anomalous patterns

---

## Visualization & Analysis

### Comprehensive Plotting Strategy

**Time Series Plots:**
- **Business Context**: Show anomalies in temporal business context
- **Seasonal Patterns**: Identify seasonal anomaly trends
- **Delivery Schedule**: Respect Monday/Saturday delivery windows

**Inventory-Specific Analysis:**
- **Category Breakdown**: Which inventory items drive anomalies?
- **Ratio Analysis**: Are proportions unusual or just volumes?
- **Correlation Patterns**: How do inventory categories relate?

**Statistical Summary:**
- **Quantitative Metrics**: Precise anomaly counts and percentages
- **Business Insights**: Actionable information for operations

---

## Model Deployment

### Cell 28: Production-Ready Artifacts

**Model Persistence:**
```python
best_model.save('enhanced_inventory_autoencoder.keras')
joblib.dump(scaler, 'robust_scaler_inventory.pkl')
joblib.dump(enhanced_model_config, 'enhanced_inventory_autoencoder_config.pkl')
```

**Why These Artifacts?**
- **.keras Format**: Modern, efficient, includes architecture and weights
- **Separate Scaler**: Ensures identical preprocessing for new data
- **Configuration File**: Complete metadata for reproducible inference

**Feature Engineering Pipeline:**
```python
def create_inventory_features(data):
    # Complete feature engineering pipeline
```

**Why Packaged Pipeline?**
- **Reproducibility**: Identical feature engineering for training and inference
- **Maintainability**: Single source of truth for data processing
- **Production Safety**: Reduces deployment errors

---

## Technical Decisions Explained

### Library Choices

**TensorFlow/Keras:**
- **Industry Standard**: Mature, well-supported deep learning framework
- **Production Ready**: Excellent deployment tools and serving options
- **Autoencoder Support**: Built-in layers and optimizers for autoencoder architectures

**Optuna:**
- **State-of-the-Art**: Most advanced hyperparameter optimization library
- **Efficiency**: Intelligent search algorithms reduce training time
- **Integration**: Seamless integration with machine learning frameworks

**Scikit-learn (Preprocessing):**
- **Reliability**: Battle-tested preprocessing tools
- **Consistency**: Standard interface across different scalers and transformers
- **Integration**: Works seamlessly with deep learning workflows

**Pandas/NumPy (Data Handling):**
- **Performance**: Optimized for numerical computations
- **Functionality**: Rich set of data manipulation tools
- **Ecosystem**: Standard tools in data science workflows

### Architecture Decisions

**Why Autoencoder over Other Approaches?**

1. **vs. Isolation Forest:**
   - Better at capturing complex, non-linear patterns
   - Learns inventory-specific representations
   - More interpretable reconstruction errors

2. **vs. One-Class SVM:**
   - Handles high-dimensional data better
   - Learns hierarchical features
   - More scalable to larger datasets

3. **vs. Statistical Methods (Z-score, IQR):**
   - Captures multivariate patterns
   - Learns temporal dependencies
   - Handles non-linear relationships

**Why Deep Architecture?**
- **Pattern Complexity**: Inventory patterns involve complex interactions
- **Temporal Dependencies**: Multiple time scales (daily, weekly, seasonal)
- **Multi-Category Learning**: Simultaneous modeling of 8+ inventory categories

### Performance Considerations

**Training Efficiency:**
- **Batch Size Optimization**: Balanced between memory usage and gradient stability
- **Early Stopping**: Prevents unnecessary computation
- **Hyperparameter Search**: Focused on practical ranges

**Inference Speed:**
- **Lightweight Architecture**: Efficient for real-time anomaly detection
- **Preprocessing Pipeline**: Optimized feature engineering
- **Model Format**: Fast loading and execution

**Scalability:**
- **Feature Engineering**: Linear complexity with data size
- **Model Architecture**: Scales well with inventory categories
- **Deployment**: Easy integration with production systems

### Business Alignment

**Operational Requirements:**
- **Interpretability**: Clear anomaly scores and category analysis
- **Actionability**: Specific insights about which inventory items are anomalous
- **Flexibility**: Multiple sensitivity levels for different business contexts

**Integration Considerations:**
- **Data Pipeline**: Works with existing inventory data formats
- **Delivery Schedule**: Respects business constraints (Monday/Saturday)
- **Maintenance**: Self-contained artifacts for easy deployment

This comprehensive documentation provides the technical foundation for understanding, maintaining, and extending the inventory anomaly detection system.