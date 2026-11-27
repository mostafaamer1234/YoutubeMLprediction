"""Cross-validated evaluation with baselines.

A single 80/20 split on a few hundred videos gives a noisy R² (and the scraped
dataset has only 10 rows, so its train/test numbers are not meaningful). This
reports 5-fold cross-validated R² and RMSE for each model next to a baseline that
always predicts the training mean, plus a log-target variant for view counts,
whose heavy tail otherwise dominates the error.

    python src/evaluate_cv.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
TEXT_FEATURES = ["duration_seconds", "publish_age_days", "title_len", "title_caps_ratio", "has_numbers_title"]
KEYWORDS = ["kw_official", "kw_trailer", "kw_live", "kw_remix", "kw_tutorial", "kw_news", "kw_review", "kw_shorts", "kw_asmr", "kw_vlog"]


def cv_report(name: str, X: pd.DataFrame, y: np.ndarray, folds: int = 5) -> list[dict]:
    cv = KFold(folds, shuffle=True, random_state=42)
    models = {
        "Baseline (predict mean)": DummyRegressor(),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
    }
    rows = []
    for label, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=("r2", "neg_root_mean_squared_error"))
        rows.append({
            "task": name, "model": label, "n": len(y),
            "r2": res["test_r2"].mean(), "r2_sd": res["test_r2"].std(),
            "rmse": -res["test_neg_root_mean_squared_error"].mean(),
        })
    return rows


def main() -> None:
    api = pd.read_csv(ROOT / "data/processed/api_combined_preprocessed.csv")
    feats = [f for f in TEXT_FEATURES + ["channel_subscribers"] + KEYWORDS if f in api.columns]
    api = api[np.isfinite(api["engagement_rate"])]
    rows = cv_report("Engagement rate (API data)", api[feats].fillna(0.0), api["engagement_rate"].to_numpy())

    views = api[api["views"] > 0] if "views" in api.columns else None
    if views is not None and len(views) >= 50:
        rows += cv_report("log(views) (API data)", views[feats].fillna(0.0), np.log1p(views["views"].to_numpy()))

    scraped = pd.read_csv(ROOT / "data/processed/scraped_preprocessed.csv")
    print(f"Scraped dataset: {len(scraped)} rows, too few for a meaningful estimate, so it is excluded.\n")

    print("| Task | Model | Videos | CV R² (mean ± sd) | CV RMSE |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['task']} | {r['model']} | {r['n']} | {r['r2']:.2f} ± {r['r2_sd']:.2f} | {r['rmse']:.4f} |")


if __name__ == "__main__":
    main()
