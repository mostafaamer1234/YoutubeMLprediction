#!/usr/bin/env python3
"""
Machine Learning Model for YouTube API Data
Trains Random Forest and XGBoost models on API data to predict engagement rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def load_api_data(file_path):
    """Load and validate API data"""
    print(f"📊 Loading API data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Loaded {len(df)} samples")
    return df

def prepare_api_features(df):
    """Prepare features for API data model"""
    print("🔧 Preparing features for API data...")
    
    # Features available in API data (includes channel_subscribers)
    base_features = [
        "duration_seconds", "publish_age_days", "title_len", "title_caps_ratio", "has_numbers_title",
        "channel_subscribers",  # Key differentiator from scraped data
        "kw_official", "kw_trailer", "kw_live", "kw_remix", "kw_tutorial",
        "kw_news", "kw_review", "kw_shorts", "kw_asmr", "kw_vlog"
    ]
    
    # Check which features are available
    available_features = [f for f in base_features if f in df.columns]
    print(f"   Available features: {len(available_features)}")
    print(f"   Features: {available_features}")
    
    # Prepare feature matrix
    X = df[available_features].fillna(0.0)
    y = df['engagement_rate'].values  # Target: engagement rate
    
    # Remove rows with missing target
    valid_mask = np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"   Valid samples: {len(X)}")
    print(f"   Target variable: engagement_rate")
    print(f"   Target range: {y.min():.4f} - {y.max():.4f}")
    return X, y, available_features

def train_api_models(X, y, features, output_dir):
    """Train Random Forest and XGBoost models on API data"""
    print("🤖 Training models on API data...")
    
    # Split data
    if len(X) < 4:
        print("   Warning: Too few samples, using all data for training")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train Random Forest
    print("   Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Train XGBoost
    print("   Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    # Evaluate models
    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    xgb_pred_train = xgb.predict(X_train)
    xgb_pred_test = xgb.predict(X_test)
    
    # Calculate metrics
    rf_metrics = {
        'train_r2': r2_score(y_train, rf_pred_train),
        'test_r2': r2_score(y_test, rf_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, rf_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred_test))
    }
    
    xgb_metrics = {
        'train_r2': r2_score(y_train, xgb_pred_train),
        'test_r2': r2_score(y_test, xgb_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, xgb_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, xgb_pred_test))
    }
    
    # Print results
    print("\n API Data Model Performance:")
    print("   Random Forest:")
    print(f"     Train R²: {rf_metrics['train_r2']:.4f}")
    print(f"     Test R²: {rf_metrics['test_r2']:.4f}")
    print(f"     Train RMSE: {rf_metrics['train_rmse']:.4f}")
    print(f"     Test RMSE: {rf_metrics['test_rmse']:.4f}")
    
    print("   XGBoost:")
    print(f"     Train R²: {xgb_metrics['train_r2']:.4f}")
    print(f"     Test R²: {xgb_metrics['test_r2']:.4f}")
    print(f"     Train RMSE: {xgb_metrics['train_rmse']:.4f}")
    print(f"     Test RMSE: {xgb_metrics['test_rmse']:.4f}")
    
    # Save models
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/api_random_forest.pkl", 'wb') as f:
        pickle.dump(rf, f)
    
    with open(f"{output_dir}/api_xgboost.pkl", 'wb') as f:
        pickle.dump(xgb, f)
    
    # Save feature info
    feature_info = {
        'feature_names': features,
        'rf_importances': rf.feature_importances_,
        'xgb_importances': xgb.feature_importances_,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics
    }
    
    with open(f"{output_dir}/api_features.pkl", 'wb') as f:
        pickle.dump(feature_info, f)
    
    # Create visualizations
    create_feature_importance_plot(rf.feature_importances_, features, output_dir)
    create_engagement_trends_plot(X, y, features, output_dir)
    
    return rf, xgb, rf_metrics, xgb_metrics

def create_feature_importance_plot(importances, features, output_dir):
    """Create feature importance visualization"""
    print(" Creating feature importance plot...")
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(sorted_features)), sorted_importances)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
    plt.title('Feature Importance - API Data Model (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Color bars by importance
    for i, bar in enumerate(bars):
        if i < 3:  # Top 3 features
            bar.set_color('red')
        elif i < 6:  # Next 3 features
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')
    
    plt.savefig(f"{output_dir}/api_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/api_feature_importance.png")

def create_engagement_trends_plot(X, y, features, output_dir):
    """Create engagement trends visualization"""
    print(" Creating engagement trends plot...")
    
    # Create DataFrame for easier analysis
    df_plot = pd.DataFrame(X, columns=features)
    df_plot['engagement_rate'] = y
    
    # Analyze by duration bins
    if 'duration_seconds' in features:
        durations = df_plot['duration_seconds']
        
        # Create duration bins
        bins = [0, 60, 180, 600, 1800, float('inf')]  # 0-1min, 1-3min, 3-10min, 10-30min, 30+min
        labels = ['<1min', '1-3min', '3-10min', '10-30min', '30+min']
        df_plot['duration_bin'] = pd.cut(durations, bins=bins, labels=labels, right=False)
        
        # Calculate mean engagement by duration bin
        engagement_by_duration = df_plot.groupby('duration_bin')['engagement_rate'].mean()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(engagement_by_duration.index, engagement_by_duration.values)
        plt.title('Engagement Rate by Video Duration (API Data)', fontsize=14, fontweight='bold')
        plt.xlabel('Video Duration', fontsize=12)
        plt.ylabel('Mean Engagement Rate', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Color bars
        for i, bar in enumerate(bars):
            if engagement_by_duration.values[i] > engagement_by_duration.values.mean():
                bar.set_color('green')
            else:
                bar.set_color('lightcoral')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/api_engagement_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {output_dir}/api_engagement_trends.png")

def main():
    parser = argparse.ArgumentParser(description='Train machine learning models on YouTube API data')
    parser.add_argument('--data', required=True, help='Path to API data CSV file')
    parser.add_argument('--output', default='models', help='Output directory for models')
    args = parser.parse_args()
    
    print(" API Data Machine Learning Model Training")
    print("=" * 50)
    
    # Load data
    df = load_api_data(args.data)
    
    # Prepare features
    X, y, features = prepare_api_features(df)
    
    if len(X) == 0:
        print(" No valid data found. Exiting.")
        return
    
    # Train models
    rf, xgb, rf_metrics, xgb_metrics = train_api_models(X, y, features, args.output)
    
    print("\n API data models trained and saved successfully!")
    print(f" Models saved to: {args.output}/")
    print("   - api_random_forest.pkl")
    print("   - api_xgboost.pkl") 
    print("   - api_features.pkl")
    print("   - api_feature_importance.png")
    print("   - api_engagement_trends.png")

if __name__ == "__main__":
    main()
