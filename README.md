# YouTube Video Engagement Report

## 1. Description of the Project

This project consists of two machine learning models that predict Youtube video popularity and engagment. The first model uses Youtube data API as the Datasource, and the the second uses webscraped data as the datasource. The goal is to compare these two models effectiveness in predicting the video performance metrics and identifying key factors that influence rate and view counts.

## 2. How to Use

### Training

1. Using Python 3.8 or newer and the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Input your YouTube API key in a `.env` file:

   ```bash
   echo "YOUTUBE_API_KEY=your_api_key_here" > .env
   ```

3. Collect data with either source:

   ```bash
   python src/collect_api.py --source trending --limit 100 --out data/raw/api_trending.csv
   python src/scrape_youtube.py --mode trending --limit 100 --out data/raw/scraped_trending.csv
   ```

4. Clean it up and build features:

   ```bash
   python src/preprocess.py \
     --api data/raw/api_trending.csv data/raw/api_search_music.csv \
     --out_api data/processed/api_combined_preprocessed.csv
   ```

5. Train the models:

   ```bash
   python src/train_api_model.py --data data/processed/api_combined_preprocessed.csv --outdir models_api
   python src/train_scraped_model.py --data data/processed/scraped_preprocessed.csv --outdir models_scraped
   ```

### Inferencing

Run this example to see a prediction:

```bash
python src/predict_example.py
```

## 3. Data Collection

### Used Tools

- `googleapiclient.discovery` for the YouTube Data API
- `selenium`, `BeautifulSoup4`, and `webdriver-manager` for scraping
- `pandas` and `numpy` for tidying up the data

### Collected Attributes

- **API data**: titles, descriptions, tags, publish dates, view/like/comment counts, channel subscriber counts, ISO durations
- **Scraped data**: titles, URLs, video IDs, channel names, view strings, duration strings, a guessed category label

### Number of Data Samples

- API data: 300 videos (100 trending + 200 music search)
- Scraped data: 10 videos

### API Usage

- Endpoints: `videos.list`, `search.list`, `channels.list`
- Roughly one quota unit per video pulled

### 2 Sample Data After Preprocessing

| video_id | title | views | duration_seconds | engagement_rate | channel_subscribers | title_len | kw_official |
|----------|-------|-------|------------------|-----------------|---------------------|-----------|-------------|
| 3cT5ML2l7KQ | VonOff1700 - Seen First (Official Video) | 155,040 | 142 | 0.0805 | 326,000 | 40 | 1 |
| i36Zw32GfRQ | Reminders of Him \| Official Trailer | 6,123,454 | 169 | 0.0021 | 10,600,000 | 32 | 1 |

## 4. Data Preprocessing

### Data Cleaning

- Turn view/like/comment strings into plain integers
- Convert ISO 8601 durations into seconds
- Work out how many days have passed since upload
- Fill empty fields with either zeros or medians
- Lowercase titles and strip odd characters so the text features behave

## 5. Feature Engineering

### How Data Is Processed After Loading

- Load the raw CSVs
- Parse numbers, dates, and durations
- Build extra columns for timing, text, and engagement ratios
- Remove or fill missing values
- Save the cleaned tables back to `data/processed/`

### 3 Collected Data Rows With Features

| video_id | duration_seconds | title_len | engagement_rate | channel_subscribers | kw_official | kw_trailer | title_caps_ratio |
|----------|------------------|-----------|-----------------|---------------------|-------------|------------|-----------------|
| 3cT5ML2l7KQ | 142 | 40 | 0.0805 | 326,000 | 1 | 0 | 0.15 |
| i36Zw32GfRQ | 169 | 32 | 0.0021 | 10,600,000 | 1 | 1 | 0.125 |
| 9_ofCQ0eOTc | 9,876 | 45 | 0.0164 | 6,190,000 | 0 | 0 | 0.133 |

## 6. Model Development and Evaluation

### Train and Test Data Partition

- 80% of each dataset is used for training, 20% for testing
- Random seed is held at 42 so results are repeatable

### Model-1 Based on Scraped Data

#### Machine Learning Model

- RandomForestRegressor with 300 trees and no depth cap

#### Input to Model

- 15 features taken from the scraped table (duration, title stats, keyword flags)

#### Size of Train Data

- 8 training samples, 2 testing samples

#### Attributes Used

- Duration in seconds, title length, share of uppercase letters, number flag, plus ten keyword indicators

#### Performance With Training Data

- R² ≈ 0.87
- RMSE ≈ 286,826 views

####  Performance With Test Data

- R² ≈ 0.03
- RMSE ≈ 1,159,372 views

### Model-2 Based on API Usage

#### Machine Learning Model

- RandomForestRegressor with 300 trees and an XGBoost model with 500 trees (used for comparison)

#### Input to Model

- 16 features, adding channel subscriber count to the set above

####  Size of Train Data

- 240 training samples, 60 testing samples

####  Attributes Used

- Duration, title metrics, keyword flags, subscriber count, basic engagement numbers

#### Performance With Training Data

- R² ≈ 0.90
- RMSE ≈ 0.0406 engagement rate points

####  Performance With Test Data

- R² ≈ 0.54
- RMSE ≈ 0.0303 engagement rate points

### Feature Importance

- Used the built-in `feature_importances_` from each Random Forest model
- The API model leans heavily on channel subscribers, while the scraped model leans on title length and duration
- Importance scores are normalized so they add up to 1.0

## 7. Visualization

- `scraped_vs_api_accuracy.png` compares accuracy and error between the two models
- `feature_importance_scraped.png` and `feature_importance_api.png` show which features mattered most for each model
- `engagement_trends_by_category_api.png` plots engagement versus video duration buckets for the API data

## 8. Discussion and Conclusions

### Project Findings

- The API-based model performs better because it knows the subscriber counts and other engagement numbers
- It's true for both models that shorter titles with moderate capitalization tend to work better
- Videos under about three minutes often pull stronger engagement in this sample

### Challenges Encountered

- Scraping was often blocked by scrape blockers, so the scraped dataset wasn't very big
- Mixing scraped text fields with API metrics required careful cleaning
- Limited samples made it hard to avoid overfitting, especially for the scraped model

### Ethical and Legal Considerations

- Stayed within the YouTube Data API terms and added delays between calls
- Did not collect any personal data
- Only using the information for learning and documentation

### Recommendations for Improving the Model

- Gather a lot more examples, ideally a few thousand for each source
- Add richer text analysis (sentiment, key phrases) and maybe thumbnail cues
- Try stronger validation (like k-fold) and tune the model settings more thoroughly
- Build simple monitoring if the model ever goes live so drifts can be spotted fast, and I plan to revisit the models once I have more reliable data to feed them.<!-- EOF -->

### Conclusion

- The API based pipeline delivers more dependable results because it has direct access to core engagement signals such as subscriber counts, likes, and comments. The limited fields the scraped workflow has captured led to unstable forecasts. The analysis also highlighted how features like video duration, title length, and capitalization style can shape viewer engagement. Shorter videos and concise, well-formatted titles tended to yield higher engagement rates across samples. In contrast, the scraped model suffered from limited data availability and overfitting, underscoring the importance of dataset size and completeness in predictive modeling.

In future work, expanding the dataset and incorporating new variables—such as sentiment analysis of titles, posting time, and thumbnail attributes—could strengthen the model’s generalizability. Overall, this study reinforced the principle that high-quality, structured data is the foundation of meaningful machine learning outcomes and that careful feature engineering remains essential for understanding digital engagement behavior.
