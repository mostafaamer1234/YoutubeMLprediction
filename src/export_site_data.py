"""Export out-of-fold predictions for the static site (site/data.json).

Uses the same features, folds and random forest as src/evaluate_cv.py, so every point on the
site is a prediction for a video the model did not train on.

    python src/export_site_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_cv import KEYWORDS, ROOT, TEXT_FEATURES  # noqa: E402


def oof(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42)
    return cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))


def main() -> None:
    api = pd.read_csv(ROOT / "data/processed/api_combined_preprocessed.csv")
    feats = [f for f in TEXT_FEATURES + ["channel_subscribers"] + KEYWORDS if f in api.columns]
    api = api[np.isfinite(api["engagement_rate"]) & (api["views"] > 0)].reset_index(drop=True)
    X = api[feats].fillna(0.0)
    y_views, y_eng = np.log10(api["views"].to_numpy()), api["engagement_rate"].to_numpy()
    p_views, p_eng = oof(X, np.log1p(api["views"].to_numpy())), oof(X, y_eng)
    p_views = np.log10(np.expm1(p_views))
    out = {
        "n": len(api),
        "views": {"r2": float(r2_score(y_views, p_views)), "actual": np.round(y_views, 3).tolist(), "pred": np.round(p_views, 3).tolist()},
        "engagement": {"r2": float(r2_score(y_eng, p_eng)), "actual": np.round(y_eng, 4).tolist(), "pred": np.round(p_eng, 4).tolist()},
        "title": api["title"].str.slice(0, 70).tolist(),
        "channel": api["channel_title"].tolist(),
    }
    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "data.json").write_text(json.dumps(out))
    print(len(api), "videos; views R2", round(out["views"]["r2"], 3), "engagement R2", round(out["engagement"]["r2"], 3))


if __name__ == "__main__":
    main()
