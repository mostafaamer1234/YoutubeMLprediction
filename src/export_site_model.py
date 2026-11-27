"""Export the view-count model for the browser predictor (site/model.json).

Trains the same random forest as src/evaluate_cv.py (min_samples_leaf=3, log target) on every
video, and writes its trees as flat arrays so site/predict.js can run them. Also writes the
5-fold out-of-fold error band, so the site can show a likely range instead of a single number,
and the videos themselves for the "similar videos" list.

    python src/export_site_model.py
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

TREES = 150


def main() -> None:
    api = pd.read_csv(ROOT / "data/processed/api_combined_preprocessed.csv")
    feats = [f for f in TEXT_FEATURES + ["channel_subscribers"] + KEYWORDS if f in api.columns]
    api = api[np.isfinite(api["engagement_rate"]) & (api["views"] > 0)].reset_index(drop=True)
    X = api[feats].fillna(0.0)
    y = np.log1p(api["views"].to_numpy())

    model = RandomForestRegressor(n_estimators=TREES, min_samples_leaf=3, random_state=42)
    oof = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
    resid = np.log10(api["views"].to_numpy()) - np.log10(np.expm1(oof))
    model.fit(X, y)

    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "f": t.feature.tolist(),
            "t": np.round(t.threshold, 5).tolist(),
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": np.round(t.value[:, 0, 0], 5).tolist(),
        })
    out = {
        "features": feats,
        "trees": trees,
        "importance": dict(zip(feats, np.round(model.feature_importances_, 4).tolist())),
        "cv_r2_log10": round(float(r2_score(np.log10(api["views"]), np.log10(np.expm1(oof)))), 3),
        "band_log10": [round(float(q), 3) for q in np.quantile(resid, [0.1, 0.9])],
        "n": len(api),
        "videos": [
            {"title": str(r.title), "channel": str(r.channel_title), "views": int(r.views), "subs": int(r.channel_subscribers or 0),
             "dur": float(r.duration_seconds if np.isfinite(r.duration_seconds) else 0), "age": float(r.publish_age_days)}
            for r in api.itertuples()
        ],
    }
    (ROOT / "site" / "model.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{TREES} trees, {len(api)} videos, out-of-fold R2 (log10 views) {out['cv_r2_log10']}, 80% band {out['band_log10']}")


if __name__ == "__main__":
    main()
