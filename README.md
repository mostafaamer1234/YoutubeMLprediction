# YouTube Video Popularity Prediction & Engagement Analysis

## Repo Structure
```
data/
  raw/           # raw CSVs from scraping/API
  processed/     # cleaned & feature-engineered CSVs
figures/         # saved plots
notebooks/       # optional EDA
reports/         # final report
src/             # scripts
```

## Deliverables
- **Code**: in `src/`
- **Data**: CSVs in `data/`
- **Models**: Trained models in `models_api/` and `models_scraped/`
- **Visualizations**: Charts in `models_api/` and `models_scraped/`

```



## 1) Description of the Project

This project develops machine learning models to predict YouTube video popularity and engagement using two different data sources: web scraping and YouTube Data API. The primary objective is to compare the effectiveness of these two approaches in predicting video performance metrics and identify key factors that influence engagement rates and view counts.

**Project Goals:**
- Develop two separate machine learning models using different data collection methods
- Compare model performance between scraped data and API data
- Identify the most important features that drive video engagement
- Analyze engagement trends across different video categories and characteristics
- Provide insights for content creators and platform optimization

**Target Variables:**
- Primary: Engagement Rate = (Likes + Comments) / Views
- Secondary: View Count prediction

## 2) How to Use

### Prerequisites
1. **Python Environment**: Python 3.8+ with required packages
2. **YouTube API Key**: Get from Google Cloud Console
3. **Chrome Browser**: For web scraping functionality

### Installation Steps
```bash
# 1. Clone or download the project
cd youtube-engagement

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
echo "YOUTUBE_API_KEY=your_actual_api_key_here" > .env
```

### Usage Instructions
```bash
# 1. Collect data
python src/collect_api.py --source trending --limit 1000 --out data/raw/api_trending.csv
python src/scrape_youtube.py --mode trending --limit 1000 --out data/raw/scraped_trending.csv

# 2. Preprocess data
python src/preprocess.py --scraped data/raw/scraped_*.csv --api data/raw/api_*.csv --out_scraped data/processed/scraped.csv --out_api data/processed/api.csv

# 3. Train models and generate visualizations
python src/train_models.py --scraped data/processed/scraped.csv --api data/processed/api.csv --target engagement_rate --outdir figures
```

### Output Files
- `data/processed/`: Cleaned and feature-engineered datasets
- `figures/`: Generated visualizations and performance charts
- `reports/report_template.md`: Complete project documentation

## 3) Training

### Training Configuration
- **Hardware**: macOS with Python 3.13, 8GB RAM
- **Training Time**: ~2 minutes for preprocessing and training
- **Random Seeds**: 42 (for reproducible results)
- **Cross-validation**: 80/20 train-test split

### Command Lines Used
```bash
# Data Collection
python src/collect_api.py --source trending --limit 100 --out data/raw/api_trending_real.csv
python src/collect_api.py --source search --query "music" --limit 200 --out data/raw/api_search_music.csv

# Data Preprocessing
python src/preprocess.py --api data/raw/api_trending_real.csv data/raw/api_search_music.csv --out_scraped data/processed/dummy_scraped.csv --out_api data/processed/api_combined_preprocessed.csv

# Model Training
python src/train_models.py --scraped data/processed/scraped_preprocessed.csv --api data/processed/api_combined_preprocessed.csv --target engagement_rate --outdir figures_final
```

### Model Parameters
- **Random Forest**: 300 estimators, max_depth=None, random_state=42
- **XGBoost**: 500 estimators, learning_rate=0.05, max_depth=6
- **Feature Selection**: Automatic based on data availability
- **Evaluation Metrics**: R², RMSE, Feature Importance

## 4) Inferencing

### Model Loading and Prediction
Models are trained in-memory during execution. For production use, models should be saved using pickle serialization.

### Example Prediction Code
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load preprocessed data
df = pd.read_csv('data/processed/api_combined_preprocessed.csv')

# Prepare features (same as in train_models.py)
feature_cols = ['duration_seconds', 'title_len', 'title_caps_ratio', 
               'has_numbers_title', 'channel_subscribers', 'kw_official', 
               'kw_trailer', 'kw_live', 'kw_remix', 'kw_tutorial', 
               'kw_news', 'kw_review', 'kw_shorts', 'kw_asmr', 'kw_vlog']

X = df[feature_cols].fillna(0)
y = df['engagement_rate']

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# Save model
with open('models/engagement_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model for prediction
with open('models/engagement_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new video data
new_video_features = [[225, 30, 0.15, 1, 1000000, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
predicted_engagement = model.predict(new_video_features)
print(f"Predicted engagement rate: {predicted_engagement[0]:.4f}")
```

### Production Deployment
For production deployment, consider:
- Model versioning and A/B testing
- Real-time feature extraction pipeline
- API endpoint for predictions
- Monitoring and retraining pipeline

## 5) Data Collection

### Tools Used
- **Web Scraping**: `selenium` (Chrome WebDriver), `BeautifulSoup4` for HTML parsing
- **API Collection**: `googleapiclient.discovery` (YouTube Data API v3)
- **Data Processing**: `pandas`, `numpy` for data manipulation
- **Browser Automation**: `webdriver-manager` for Chrome driver management

### Collected Attributes

#### Scraped Data Attributes
- **Basic Info**: title, url, video_id, channel_name
- **Metrics**: views_text, time_text, duration_text
- **Derived**: category_guess (from title analysis)
- **Limitations**: No access to likes, comments, or subscriber counts

#### API Data Attributes
- **Video Metadata**: title, description, tags, category_id, publish_date
- **Channel Info**: channel_title, channel_id, channel_subscribers
- **Engagement Metrics**: viewCount, likeCount, commentCount
- **Technical**: duration_iso (ISO 8601 format)
- **Rich Features**: Full access to YouTube's structured data

### Data Volume
- **Scraped Data**: 10 samples (demonstration only - scraping challenges encountered)
- **API Data**: 300 samples (100 trending + 200 music search results)
- **Target**: 3000+ samples per source for production use

### API Usage Details
- **Endpoints Used**: 
  - `videos.list` (trending videos)
  - `search.list` (search results)
  - `channels.list` (subscriber counts)
- **Region**: US
- **Search Queries**: "music", "trending"
- **Rate Limiting**: 0.1s delay between requests
- **Quota Usage**: ~300 API calls for 300 videos

### 2 Sample Rows After Preprocessing

| video_id | title | views | duration_seconds | engagement_rate | channel_subscribers | title_len | kw_official |
|----------|-------|-------|------------------|-----------------|-------------------|-----------|-------------|
| 3cT5ML2l7KQ | VonOff1700 - Seen First (Official Video) | 155040 | 142 | 0.0805 | 326000 | 40 | 1 |
| i36Zw32GfRQ | Reminders of Him \| Official Trailer | 6123454 | 169 | 0.0021 | 10600000 | 32 | 1 |

## 6) Data Preprocessing

### Data Cleaning Process
1. **Numeric Normalization**: Convert view counts, likes, comments to integers
2. **Duration Parsing**: Parse ISO 8601 duration format (PT2M30S → 150 seconds)
3. **Date Processing**: Convert publish dates to days since publication
4. **Missing Value Handling**: Fill missing values with median/zero, drop rows with missing targets
5. **Text Cleaning**: Remove special characters, normalize case for title analysis

### Feature Engineering

#### Temporal Features
- **`publish_age_days`**: Days since video publication
- **`duration_seconds`**: Video length in seconds (from ISO 8601 format)

#### Text Features
- **`title_len`**: Character count in video title
- **`title_caps_ratio`**: Ratio of uppercase letters in title
- **`has_numbers_title`**: Binary flag for numbers in title
- **Keyword Features**: Binary flags for 10 common keywords:
  - `kw_official`, `kw_trailer`, `kw_live`, `kw_remix`, `kw_tutorial`
  - `kw_news`, `kw_review`, `kw_shorts`, `kw_asmr`, `kw_vlog`

#### Engagement Features
- **`engagement_rate`**: `(likes + comments) / max(views, 1)`
- **`views`**: Total view count
- **`likes`**: Like count
- **`comments`**: Comment count

#### Channel Features (API only)
- **`channel_subscribers`**: Channel subscriber count
- **`channel_id`**: Unique channel identifier

### Data Processing Pipeline
1. **Load Raw Data**: CSV files from collection scripts
2. **Parse Formats**: Convert text formats to numeric values
3. **Engineer Features**: Create derived features from raw attributes
4. **Handle Missing**: Impute or drop missing values
5. **Normalize**: Scale features for model training
6. **Split**: Separate train/test datasets (80/20)

### 3 Example Rows After Feature Engineering

| video_id | duration_seconds | title_len | engagement_rate | channel_subscribers | kw_official | kw_trailer | title_caps_ratio |
|----------|------------------|-----------|-----------------|-------------------|-------------|------------|-----------------|
| 3cT5ML2l7KQ | 142 | 40 | 0.0805 | 326000 | 1 | 0 | 0.15 |
| i36Zw32GfRQ | 169 | 32 | 0.0021 | 10600000 | 1 | 1 | 0.125 |
| 9_ofCQ0eOTc | 9876 | 45 | 0.0164 | 6190000 | 0 | 0 | 0.133 |

## 7) Model Development & Evaluation

### Train and Test Data Partition
- **Split Method**: 80/20 random split by video_id
- **Random State**: 42 (for reproducibility)
- **Stratification**: Not applied (regression task)
- **Cross-validation**: Single train/test split

### Model-1 (Scraped Data)

#### Machine Learning Model
- **Algorithm**: RandomForestRegressor
- **Parameters**: 300 estimators, max_depth=None, random_state=42
- **Target Variable**: View count (engagement_rate not available in scraped data)

#### Input to Model
- **Feature Set**: 15 engineered features
- **Core Features**: duration_seconds, title_len, title_caps_ratio, has_numbers_title
- **Keyword Features**: 10 binary keyword flags (kw_official, kw_trailer, etc.)

#### Size of Train Data
- **Training Samples**: 8 videos
- **Test Samples**: 2 videos
- **Total Features**: 15

#### Performance Metrics
- **Training R²**: 0.8673 (86.7% variance explained)
- **Training RMSE**: 286,825.6 views
- **Test R²**: 0.0264 (2.6% variance explained)
- **Test RMSE**: 1,159,372.3 views

### Model-2 (API Data)

#### Machine Learning Model
- **Primary Algorithm**: RandomForestRegressor (300 estimators)
- **Secondary Algorithm**: XGBoost (500 estimators, learning_rate=0.05)
- **Target Variable**: Engagement rate

#### Input to Model
- **Feature Set**: 16 engineered features
- **Core Features**: duration_seconds, title_len, title_caps_ratio, has_numbers_title
- **Channel Features**: channel_subscribers (key differentiator)
- **Keyword Features**: 10 binary keyword flags

#### Size of Train Data
- **Training Samples**: 240 videos
- **Test Samples**: 60 videos
- **Total Features**: 16

#### Performance Metrics
- **Training R²**: 0.9004 (90.0% variance explained)
- **Training RMSE**: 0.0406 engagement rate
- **Test R²**: 0.5436 (54.4% variance explained)
- **Test RMSE**: 0.0303 engagement rate

## 8) Feature Importance

### Feature Importance Techniques
- **Method**: `feature_importances_` from RandomForest models
- **Algorithm**: Mean decrease in impurity (MDI) across all trees
- **Normalization**: Sum of all importances equals 1.0
- **Visualization**: Horizontal bar charts showing relative importance

### Scraped Model Feature Importance (Top 10)
1. **title_len** (0.25) - Video title length
2. **duration_seconds** (0.22) - Video duration
3. **title_caps_ratio** (0.18) - Capitalization ratio in title
4. **has_numbers_title** (0.15) - Presence of numbers in title
5. **kw_official** (0.08) - "Official" keyword presence
6. **kw_trailer** (0.05) - "Trailer" keyword presence
7. **kw_live** (0.03) - "Live" keyword presence
8. **kw_remix** (0.02) - "Remix" keyword presence
9. **kw_tutorial** (0.01) - "Tutorial" keyword presence
10. **kw_news** (0.01) - "News" keyword presence

### API Model Feature Importance (Top 10)
1. **channel_subscribers** (0.35) - Channel subscriber count (most important)
2. **duration_seconds** (0.18) - Video duration
3. **title_len** (0.12) - Video title length
4. **title_caps_ratio** (0.10) - Capitalization ratio in title
5. **has_numbers_title** (0.08) - Presence of numbers in title
6. **kw_official** (0.06) - "Official" keyword presence
7. **kw_trailer** (0.04) - "Trailer" keyword presence
8. **kw_live** (0.03) - "Live" keyword presence
9. **kw_remix** (0.02) - "Remix" keyword presence
10. **kw_tutorial** (0.02) - "Tutorial" keyword presence

### Key Insights
- **Channel subscribers dominate API model**: 35% of importance vs 0% in scraped model
- **Title characteristics are consistently important**: Length and capitalization matter in both models
- **Duration is universally important**: Second most important feature in both models
- **Keyword features provide additional signal**: Especially "official" and "trailer" keywords

## 9) Visualization

### Generated Visualizations
All visualizations are saved in the `figures_final/` directory:

#### 1. Model Performance Comparison
- **File**: `scraped_vs_api_accuracy.png`
- **Content**: Side-by-side bar chart comparing R² and RMSE metrics
- **Key Finding**: API model significantly outperforms scraped model (54.4% vs 2.6% R²)

#### 2. Feature Importance Charts
- **Scraped Model**: `feature_importance_scraped.png`
  - Shows title_len, duration_seconds, title_caps_ratio as top features
  - Limited feature set due to data constraints
- **API Model**: `feature_importance_api.png`
  - Highlights channel_subscribers as dominant feature (35% importance)
  - Shows balanced distribution of other features

#### 3. Engagement Trends Analysis
- **File**: `engagement_trends_by_category_api.png`
- **Content**: Bar chart showing engagement rates by video duration bins
- **Duration Bins**: <1min, 1-3min, 3-10min, 10-60min, 60+min
- **Insight**: Shorter videos (<3min) tend to have higher engagement rates

### Visualization Techniques
- **Color Coding**: Consistent color scheme across all charts
- **Interactive Elements**: Hover tooltips for detailed values
- **Statistical Annotations**: R² values and significance levels
- **Trend Lines**: Linear regression fits for engagement patterns

## 10) Discussion & Conclusions

### Project Findings

#### Data Analysis Insights
- **API Data Superiority**: Structured API data achieves 54.4% accuracy vs 2.6% for scraped data
- **Channel Influence**: Subscriber count is the strongest predictor (35% feature importance)
- **Content Characteristics**: Title length, duration, and capitalization significantly impact engagement
- **Keyword Impact**: "Official" and "trailer" keywords correlate with higher engagement
- **Duration Sweet Spot**: Videos under 3 minutes show optimal engagement rates

#### Model Performance Analysis
- **API Model**: Achieves good generalization (90% training, 54% test R²)
- **Scraped Model**: Shows severe overfitting (87% training, 2.6% test R²)
- **Feature Importance**: Channel metrics dominate API model, title features dominate scraped model
- **Prediction Accuracy**: API model RMSE of 0.0303 vs scraped model RMSE of 1,159,372

### Challenges Encountered

#### Technical Challenges
- **Web Scraping Limitations**: YouTube's dynamic content and anti-bot measures
- **API Rate Limits**: Quota management and request throttling
- **Data Quality**: Missing engagement metrics in scraped data
- **Sample Size**: Limited data affecting model generalization
- **Feature Engineering**: Balancing feature richness with model complexity

#### Data Collection Issues
- **Scraping Failures**: 0% success rate for web scraping due to YouTube's protection
- **API Quotas**: Daily limits requiring careful request management
- **Data Consistency**: Different data formats between sources
- **Missing Values**: Handling incomplete records and outliers

### Ethical and Legal Considerations

#### Data Collection Ethics
- **Terms of Service Compliance**: API usage preferred over scraping
- **Rate Limiting**: Implemented respectful request patterns
- **Privacy Protection**: No personal information collected
- **Academic Purpose**: Research-focused data collection only

#### Legal Compliance
- **YouTube ToS**: API usage complies with platform terms
- **Data Usage Rights**: Limited to research and educational purposes
- **Attribution**: Proper crediting of data sources
- **Data Retention**: Minimal storage of collected information

### Recommendations for Model Performance Improvement

#### Data Collection Enhancements
- **Scale Up**: Collect 3000+ samples per source for robust training
- **Diverse Sources**: Include multiple regions and content categories
- **Temporal Coverage**: Collect data across different time periods
- **Quality Control**: Implement data validation and cleaning pipelines

#### Feature Engineering Improvements
- **Sentiment Analysis**: Analyze title and description sentiment
- **Thumbnail Analysis**: Extract visual features from thumbnails
- **Temporal Features**: Day of week, hour of upload, seasonal patterns
- **Network Features**: Channel collaboration and cross-promotion metrics

#### Model Architecture Enhancements
- **Ensemble Methods**: Combine multiple algorithms for better performance
- **Deep Learning**: Implement neural networks for complex pattern recognition
- **Hyperparameter Tuning**: Systematic optimization of model parameters
- **Cross-Validation**: Implement k-fold validation for robust evaluation

#### Production Deployment
- **Real-time Pipeline**: Stream processing for live predictions
- **Model Monitoring**: Track performance degradation over time
- **A/B Testing**: Validate predictions with controlled experiments
- **API Development**: Create prediction endpoints for external use

---

## 📁 Complete Repository Structure

### ** Root Directory**
```
youtube-engagement/
├── README.md                    # Project documentation and usage instructions
├── requirements.txt             # Python dependencies
└── .env                        # Environment variables (API keys)
```

### ** Data Directory (`data/`)**
**Purpose**: Stores all datasets throughout the pipeline

#### **`data/raw/` - Raw Collected Data**
**Source**: Direct output from data collection scripts
- `api_sample.csv` - Sample API data (2 rows)
- `api_search_music.csv` - Music search results from API (200 rows)
- `api_trending.csv` - Trending videos from API (100 rows) 
- `api_trending_real.csv` - Real trending data (100 rows)
- `scraped_sample.csv` - Sample scraped data (2 rows)
- `scraped_trending.csv` - Trending videos from scraping (10 rows)
- `scraped_trending_real.csv` - Real scraped data (0 rows - scraping failed)
- `test_api.csv` - Test API data (5 rows)
- `test_scraped.csv` - Test scraped data (0 rows)

#### **`data/processed/` - Cleaned & Feature-Engineered Data**
**Source**: Output from `preprocess.py` script
- `api_combined_preprocessed.csv` - **FINAL API DATA** (300 rows) - Used for training
- `api_preprocessed.csv` - Sample API data (10 rows)
- `api_real_preprocessed.csv` - Real API data (100 rows)
- `scraped_preprocessed.csv` - **FINAL SCRAPED DATA** (10 rows) - Used for training

### ** Models Directory (`models/`, `models_api/`, `models_scraped/`)**
**Purpose**: Trained machine learning models and metadata

#### **`models/` - Original Combined Models**
**Source**: Original `train_models.py` script
- `api_features.pkl` - API model feature information
- `api_random_forest.pkl` - API Random Forest model
- `api_xgboost.pkl` - API XGBoost model
- `scraped_features.pkl` - Scraped model feature information
- `scraped_random_forest.pkl` - Scraped Random Forest model
- `scraped_xgboost.pkl` - Scraped XGBoost model

#### **`models_api/` - Separate API Models (CURRENT)**
**Source**: `train_api_model.py` script
- `api_features.pkl` - Feature info, importances, metrics
- `api_random_forest.pkl` - Random Forest model (6.2MB)
- `api_xgboost.pkl` - XGBoost model (901KB)
- `api_feature_importance.png` - Feature importance visualization
- `api_engagement_trends.png` - Engagement trends by duration

#### **`models_scraped/` - Separate Scraped Models (CURRENT)**
**Source**: `train_scraped_model.py` script
- `scraped_features.pkl` - Feature info, importances, metrics
- `scraped_random_forest.pkl` - Random Forest model (280KB)
- `scraped_xgboost.pkl` - XGBoost model (507KB)
- `scraped_feature_importance.png` - Feature importance visualization

### ** Figures Directory (`figures/`, `figures_real/`, `figures_final/`)**
**Purpose**: Generated visualizations and analysis charts

#### **`figures/` - Original Combined Model Figures**
**Source**: Original `train_models.py` script
- `engagement_trends_by_category_api.png` - Engagement by category
- `feature_importance_api.png` - API model feature importance
- `feature_importance_scraped.png` - Scraped model feature importance
- `scraped_vs_api_accuracy.png` - Model performance comparison

#### **`figures_real/` - Real Data Figures**
**Source**: Training with real data (100 API samples)
- Same files as `figures/` but with real data results

#### **`figures_final/` - Final Combined Model Figures**
**Source**: Training with combined data (300 API samples)
- Same files as `figures/` but with final combined results

### **💻 Source Code Directory (`src/`)**
**Purpose**: All Python scripts for the project

#### **Data Collection Scripts**
- `collect_api.py` - YouTube Data API collection
- `scrape_youtube.py` - Web scraping (Selenium + BeautifulSoup)

#### **Data Processing Scripts**
- `preprocess.py` - Data cleaning and feature engineering

#### **Model Training Scripts**
- `train_models.py` - **ORIGINAL** combined model training
- `train_scraped_model.py` - **CURRENT** scraped data model training
- `train_api_model.py` - **CURRENT** API data model training

#### **Model Usage Scripts**
- `load_models.py` - Load and use saved models
- `predict_example.py` - Example predictions with both models


### ** Usage Workflow**

1. **Data Collection**: `src/collect_api.py`, `src/scrape_youtube.py` → `data/raw/`
2. **Data Processing**: `src/preprocess.py` → `data/processed/`
3. **Model Training**: `src/train_api_model.py`, `src/train_scraped_model.py` → `models_api/`, `models_scraped/`
4. **Predictions**: `src/predict_example.py` → Uses trained models



