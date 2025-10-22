

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

def predict_engagement(model_type, video_features):
    """
    Predict engagement using the specified model
    
    Args:
        model_type: 'scraped' or 'api'
        video_features: Dictionary with video features
    """
    model_dir = 'models'
    
    # Load models
    rf_model = load_model(f"{model_dir}/{model_type}_random_forest.pkl")
    xgb_model = load_model(f"{model_dir}/{model_type}_xgboost.pkl")
    feature_info = load_feature_info(f"{model_dir}/{model_type}_features.pkl")
    
    # Prepare features in correct order
    feature_names = feature_info['feature_names']
    feature_vector = []
    
    for feature in feature_names:
        if feature in video_features:
            feature_vector.append(video_features[feature])
        else:
            feature_vector.append(0.0)  # Default value for missing features
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # Make predictions
    rf_pred = rf_model.predict(feature_vector)[0]
    xgb_pred = xgb_model.predict(feature_vector)[0]
    
    # Average predictions
    ensemble_pred = (rf_pred + xgb_pred) / 2
    
    return {
        'random_forest': rf_pred,
        'xgboost': xgb_pred,
        'ensemble': ensemble_pred,
        'feature_importance': dict(zip(feature_names, feature_info['feature_importances']))
    }

def main():
    print(" Loading Separate Machine Learning Models")
    print("=" * 50)
    
    # Example 1: API Model Prediction
    print("\n API Model Prediction (Engagement Rate):")
    api_video = {
        'duration_seconds': 180,      # 3 minutes
        'title_len': 25,              # Title length
        'title_caps_ratio': 0.2,       # 20% caps
        'has_numbers_title': 0,       # No numbers
        'channel_subscribers': 1000000, # 1M subscribers
        'kw_official': 1,             # Has "official"
        'kw_trailer': 0,              # No "trailer"
        'kw_live': 0,                 # No "live"
        'kw_remix': 0,                # No "remix"
        'kw_tutorial': 0,              # No "tutorial"
        'kw_news': 0,                 # No "news"
        'kw_review': 0,               # No "review"
        'kw_shorts': 0,               # No "shorts"
        'kw_asmr': 0,                 # No "asmr"
        'kw_vlog': 0                  # No "vlog"
    }
    
    api_prediction = predict_engagement('api', api_video)
    print(f"   Random Forest: {api_prediction['random_forest']:.4f}")
    print(f"   XGBoost: {api_prediction['xgboost']:.4f}")
    print(f"   Ensemble: {api_prediction['ensemble']:.4f}")
    
    # Example 2: Scraped Model Prediction
    print("\n📊 Scraped Model Prediction (View Count):")
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
    
    scraped_prediction = predict_engagement('scraped', scraped_video)
    print(f"   Random Forest: {scraped_prediction['random_forest']:,.0f} views")
    print(f"   XGBoost: {scraped_prediction['xgboost']:,.0f} views")
    print(f"   Ensemble: {scraped_prediction['ensemble']:,.0f} views")
    
    # Show feature importance
    print("\n Top 5 Most Important Features (API Model):")
    api_importance = sorted(api_prediction['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
    for feature, importance in api_importance:
        print(f"   {feature}: {importance:.3f}")
    
    print("\n Top 5 Most Important Features (Scraped Model):")
    scraped_importance = sorted(scraped_prediction['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
    for feature, importance in scraped_importance:
        print(f"   {feature}: {importance:.3f}")

if __name__ == "__main__":
    main()
