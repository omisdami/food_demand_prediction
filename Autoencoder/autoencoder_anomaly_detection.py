#!/usr/bin/env python3
"""
Autoencoder Anomaly Detection for Inventory Data
===============================================

A script to train and use autoencoders for detecting anomalous inventory patterns.
Converted from Autoencoder/Autoencoder.ipynb for production use.

Usage: 
- Train: python autoencoder_anomaly_detection.py --train --dataset data/inventory_delivery_forecast_data.csv
- Detect: python autoencoder_anomaly_detection.py --detect --dataset data/inventory_delivery_forecast_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
import joblib
import pickle
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Optuna imports
import optuna
from optuna.samplers import TPESampler

def create_directories():
    """Create necessary directories for saving models and results"""
    os.makedirs('Autoencoder', exist_ok=True)
    os.makedirs('models/autoencoder', exist_ok=True)
    os.makedirs('results/autoencoder', exist_ok=True)

def load_and_prepare_data(dataset_path):
    """Load and prepare the inventory data with feature engineering"""
    print(f"📊 Loading data from {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    
    # Extract temporal features
    df['year'] = df['delivery_date'].dt.year
    df['month'] = df['delivery_date'].dt.month
    df['day_of_week'] = df['delivery_date'].dt.dayofweek
    df['is_monday'] = (df['delivery_date'].dt.dayofweek == 0).astype(int)
    df['is_saturday'] = (df['delivery_date'].dt.dayofweek == 5).astype(int)
    df['week_of_year'] = df['delivery_date'].dt.isocalendar().week
    
    # Create seasonality features
    df['season'] = df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                    3: 'spring', 4: 'spring', 5: 'spring',
                                    6: 'summer', 7: 'summer', 8: 'summer',
                                    9: 'fall', 10: 'fall', 11: 'fall'})
    
    # Add inventory totals and ratios
    inventory_features = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
    df['total_inventory'] = df[inventory_features].sum(axis=1)
    df['wings_ratio'] = df['wings'] / df['total_inventory']
    df['protein_ratio'] = (df['wings'] + df['tenders']) / df['total_inventory']
    df['sides_ratio'] = (df['fries_reg'] + df['fries_large'] + df['veggies']) / df['total_inventory']
    df['beverages_ratio'] = df['drinks'] / df['total_inventory']
    
    # Sort by date for lag features
    df_sorted = df.sort_values('delivery_date')
    
    # Create lag features
    for item in ['wings', 'tenders', 'total_inventory']:
        df_sorted[f'{item}_lag1'] = df_sorted[item].shift(1)
        df_sorted[f'{item}_lag2'] = df_sorted[item].shift(2)
    
    # Create rolling statistics
    df_sorted['total_inventory_rolling_mean'] = df_sorted['total_inventory'].shift(1).rolling(window=4).mean()
    df_sorted['wings_rolling_std'] = df_sorted['wings'].shift(1).rolling(window=4).std()
    
    # Remove rows with NaN from lag calculations
    df_clean = df_sorted.dropna()
    
    print(f"✅ Data prepared: {len(df_clean)} records after feature engineering")
    return df_clean

def prepare_features(df):
    """Prepare feature matrix for autoencoder training"""
    inventory_features = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
    ratio_features = ['wings_ratio', 'protein_ratio', 'sides_ratio', 'beverages_ratio']
    temporal_features = ['month', 'day_of_week', 'week_of_year', 'is_monday', 'is_saturday']
    lag_features = [col for col in df.columns if '_lag' in col or '_rolling_' in col]
    
    feature_columns = inventory_features + ratio_features + temporal_features + ['total_inventory'] + lag_features
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    print(f"📊 Selected {len(feature_columns)} features for autoencoder")
    
    X = df[feature_columns].values
    return X, feature_columns

def create_improved_autoencoder(trial, input_dim):
    """Create autoencoder model with hyperparameter optimization"""
    n_layers = trial.suggest_int('n_layers', 2, 4)
    encoder_units = trial.suggest_int('encoder_units', 32, 128, step=16)
    latent_dim = trial.suggest_int('latent_dim', 8, 24)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'swish'])
    l2_reg = trial.suggest_float('l2_regularization', 1e-6, 1e-3, log=True)

    # Input layer
    inputs = Input(shape=(input_dim,))

    # Enhanced encoder with regularization
    x = inputs
    for i in range(n_layers):
        units = encoder_units // (2**i)
        x = Dense(units, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    # Bottleneck layer
    encoded = Dense(latent_dim, activation=activation, kernel_regularizer=l2(l2_reg), name='bottleneck')(x)

    # Enhanced decoder
    x = encoded
    for i in range(n_layers):
        units = encoder_units // (2**(n_layers-i-1))
        x = Dense(units, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(input_dim, activation='linear')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Compile with Huber loss for robustness
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

    return model

def enhanced_objective(trial, X_train, X_val):
    """Objective function for hyperparameter optimization"""
    model = create_improved_autoencoder(trial, X_train.shape[1])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=1e-6
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=0
    )

    # Train
    history = model.fit(
        X_train, X_train,
        epochs=150,
        batch_size=trial.suggest_int('batch_size', 8, 32, step=8),
        validation_data=(X_val, X_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    return min(history.history['val_loss'])

def train_autoencoder(dataset_path, n_trials=75):
    """Train autoencoder with hyperparameter optimization"""
    create_directories()
    
    # Load and prepare data
    df = load_and_prepare_data(dataset_path)
    X, feature_columns = prepare_features(df)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-validation split
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    print(f"🔧 Starting hyperparameter optimization with {n_trials} trials...")
    
    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    def objective(trial):
        return enhanced_objective(trial, X_train, X_val)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✅ Best trial: {study.best_trial.number}")
    print(f"✅ Best value: {study.best_value:.8f}")
    
    # Train final model with best parameters
    print("🔧 Training final model with optimal parameters...")
    best_model = create_improved_autoencoder(study.best_trial, X_train.shape[1])
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=1e-7
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'models/autoencoder/best_inventory_autoencoder.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    optimal_batch_size = study.best_params['batch_size']
    
    history = best_model.fit(
        X_train, X_train,
        epochs=200,
        batch_size=optimal_batch_size,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Calculate reconstruction errors and thresholds
    train_pred = best_model.predict(X_train, verbose=0)
    train_reconstruction_errors = np.mean(np.square(X_train - train_pred), axis=1)
    
    thresholds = {
        'conservative': np.percentile(train_reconstruction_errors, 99),
        'balanced': np.percentile(train_reconstruction_errors, 95),
        'sensitive': np.percentile(train_reconstruction_errors, 90)
    }
    
    # Save model and artifacts
    best_model.save('Autoencoder/inventory_autoencoder_model.h5')
    joblib.dump(scaler, 'Autoencoder/inventory_scaler.pkl')
    joblib.dump(study, 'Autoencoder/optuna_study.pkl')
    
    with open('Autoencoder/anomaly_threshold.json', 'w') as f:
        json.dump(thresholds, f)
    
    with open('Autoencoder/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save feature columns for later use
    with open('Autoencoder/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    print("✅ Autoencoder training completed!")
    print(f"✅ Model saved to: Autoencoder/inventory_autoencoder_model.h5")
    print(f"✅ Thresholds: {thresholds}")
    
    return best_model, scaler, thresholds, feature_columns

def detect_anomalies(dataset_path, threshold_type='balanced'):
    """Detect anomalies using trained autoencoder"""
    # Load trained model and artifacts
    model = load_model('Autoencoder/inventory_autoencoder_model.h5')
    scaler = joblib.load('Autoencoder/inventory_scaler.pkl')
    
    with open('Autoencoder/anomaly_threshold.json', 'r') as f:
        thresholds = json.load(f)
    
    with open('Autoencoder/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    print("✅ Loaded trained autoencoder model")
    
    # Load and prepare data
    df = load_and_prepare_data(dataset_path)
    X, _ = prepare_features(df)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions and calculate reconstruction errors
    predictions = model.predict(X_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(X_scaled - predictions), axis=1)
    
    # Apply threshold
    threshold = thresholds[threshold_type]
    anomalies = reconstruction_errors > threshold
    
    # Add results to dataframe
    df['reconstruction_error'] = reconstruction_errors
    df['is_anomaly'] = anomalies
    
    # Analysis
    total_anomalies = anomalies.sum()
    anomaly_percentage = 100 * total_anomalies / len(df)
    
    print(f"\n🔍 ANOMALY DETECTION RESULTS")
    print("=" * 50)
    print(f"📊 Total anomalies detected: {total_anomalies} out of {len(df)} ({anomaly_percentage:.2f}%)")
    print(f"🎯 Threshold used: {threshold_type} ({threshold:.6f})")
    
    if total_anomalies > 0:
        anomaly_data = df[df['is_anomaly']]
        print(f"\n📈 Anomaly Statistics:")
        print(f"   Average total inventory in anomalies: {anomaly_data['total_inventory'].mean():.0f}")
        print(f"   Average total inventory overall: {df['total_inventory'].mean():.0f}")
        
        # Show most anomalous dates
        top_anomalies = anomaly_data.nlargest(5, 'reconstruction_error')
        print(f"\n🚨 Top 5 Most Anomalous Dates:")
        for _, row in top_anomalies.iterrows():
            print(f"   {row['delivery_date'].strftime('%Y-%m-%d')}: Error {row['reconstruction_error']:.6f}")
    
    # Save results
    results_path = 'results/autoencoder/anomaly_detection_results.csv'
    os.makedirs('results/autoencoder', exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Autoencoder Anomaly Detection for Inventory Data')
    parser.add_argument('--train', action='store_true', help='Train the autoencoder model')
    parser.add_argument('--detect', action='store_true', help='Detect anomalies using trained model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--trials', type=int, default=75, help='Number of optimization trials (default: 75)')
    parser.add_argument('--threshold', choices=['conservative', 'balanced', 'sensitive'], 
                       default='balanced', help='Anomaly detection threshold (default: balanced)')
    
    args = parser.parse_args()
    
    if not args.train and not args.detect:
        print("❌ Please specify either --train or --detect")
        return
    
    if args.train:
        print("🔧 Training autoencoder model...")
        train_autoencoder(args.dataset, args.trials)
    
    if args.detect:
        print("🔍 Detecting anomalies...")
        detect_anomalies(args.dataset, args.threshold)

if __name__ == "__main__":
    main()
