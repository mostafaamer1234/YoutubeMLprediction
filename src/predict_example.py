#!/usr/bin/env python3


import pandas as pd
import numpy as np
import pickle
import os

def load_model(model_path):
    """Load a saved model from pickle file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_feature_info(features_path):
    """Load feature information"""
    with open(features_path, 'rb') as f:
        feature_info = pickle.load(f)
    return feature_info

def predict_scraped_model(video_features, model_dir='models_scraped'):
    """
    Predict view count using the scraped data model
    
    Args:
        video_features: Dictionary with video features
        model_dir: Directory containing scraped models
    """
    print(" Using Scraped Data Model (View Count Prediction)")
    print("-" * 50)
    
    # Load models
    rf_model = load_model(f"{model_dir}/scraped_random_forest.pkl")
    xgb_model = load_model(f"{model_dir}/scraped_xgboost.pkl")
    feature_info = load_feature_info(f"{model_dir}/scraped_features.pkl")
    
    # Prepare features
    feature_names = feature_info['feature_names']
    feature_vector = []
    
    for feature in feature_names:
        if feature in video_features:
            feature_vector.append(video_features[feature])
        else:
            feature_vector.append(0.0)
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # Make predictions
    rf_pred = rf_model.predict(feature_vector)[0]
    xgb_pred = xgb_model.predict(feature_vector)[0]
    ensemble_pred = (rf_pred + xgb_pred) / 2
    
    print(f"Random Forest Prediction: {rf_pred:,.0f} views")
    print(f"XGBoost Prediction: {xgb_pred:,.0f} views")
    print(f"Ensemble Prediction: {ensemble_pred:,.0f} views")
    
    return {
        'random_forest': rf_pred,
        'xgboost': xgb_pred,
        'ensemble': ensemble_pred
    }

def predict_api_model(video_features, model_dir='models_api'):
    """
    Predict engagement rate using the API data model
    
    Args:
        video_features: Dictionary with video features
        model_dir: Directory containing API models
    """
    print(" Using API Data Model (Engagement Rate Prediction)")
    print("-" * 50)
    
    # Load models
    rf_model = load_model(f"{model_dir}/api_random_forest.pkl")
    xgb_model = load_model(f"{model_dir}/api_xgboost.pkl")
    feature_info = load_feature_info(f"{model_dir}/api_features.pkl")
    
    # Prepare features
    feature_names = feature_info['feature_names']
    feature_vector = []
    
    for feature in feature_names:
        if feature in video_features:
            feature_vector.append(video_features[feature])
        else:
            feature_vector.append(0.0)
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # Make predictions
    rf_pred = rf_model.predict(feature_vector)[0]
    xgb_pred = xgb_model.predict(feature_vector)[0]
    ensemble_pred = (rf_pred + xgb_pred) / 2
    
    print(f"Random Forest Prediction: {rf_pred:.4f} engagement rate")
    print(f"XGBoost Prediction: {xgb_pred:.4f} engagement rate")
    print(f"Ensemble Prediction: {ensemble_pred:.4f} engagement rate")
    
    return {
        'random_forest': rf_pred,
        'xgboost': xgb_pred,
        'ensemble': ensemble_pred
    }

def main():
    print(" YouTube Video Prediction Example")
    print("=" * 60)
    
    # Example 1: Scraped Model Prediction
    print("\n📊 Example 1: Scraped Data Model")
    scraped_video = {
        'duration_seconds': 300,      # 5 minutes
        'title_len': 30,              # Title length
        'title_caps_ratio': 0.15,      # 15% caps
        'has_numbers_title': 1,       # Has numbers
        'kw_official': 0,             # No "official"
        'kw_trailer': 1,              # Has "trailer"
        'kw_live': 0,                 # No "live"
        'kw_remix': 0,                # No "remix"
        'kw_tutorial': 0,             # No "tutorial"
        'kw_news': 0,                 # No "news"
        'kw_review': 0,               # No "review"
        'kw_shorts': 0,               # No "shorts"
        'kw_asmr': 0,                 # No "asmr"
        'kw_vlog': 0                  # No "vlog"
    }
    
    scraped_prediction = predict_scraped_model(scraped_video)
    
    # Example 2: API Model Prediction
    print("\n Example 2: API Data Model")
    api_video = {
        'duration_seconds': 180,      # 3 minutes
        'publish_age_days': 5,        # 5 days old
        'title_len': 25,              # Title length
        'title_caps_ratio': 0.2,       # 20% caps
        'has_numbers_title': 0,       # No numbers
        'channel_subscribers': 1000000, # 1M subscribers
        'kw_official': 1,             # Has "official"
        'kw_trailer': 0,              # No "trailer"
        'kw_live': 0,                 # No "live"
        'kw_remix': 0,                # No "remix"
        'kw_tutorial': 0,             # No "tutorial"
        'kw_news': 0,                 # No "news"
        'kw_review': 0,               # No "review"
        'kw_shorts': 0,               # No "shorts"
        'kw_asmr': 0,                 # No "asmr"
        'kw_vlog': 0                  # No "vlog"
    }
    
    api_prediction = predict_api_model(api_video)
    
    print("\n Predictions completed successfully!")
    print("\n Model files available:")
    print("   models_scraped/ - For view count predictions")
    print("   models_api/ - For engagement rate predictions")

if __name__ == "__main__":
    main()
