# src/train_models.py
import argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def pick_features(df, source="api", target="engagement_rate"):
    base = [
        "duration_seconds","publish_age_days","title_len","title_caps_ratio","has_numbers_title",
        "kw_official","kw_trailer","kw_live","kw_remix","kw_tutorial","kw_news","kw_review","kw_shorts","kw_asmr","kw_vlog"
    ]
    if source == "api":
        extra = ["channel_subscribers"]
    else:
        extra = []
    X_cols = [c for c in base+extra if c in df.columns]
    y = df[target].values
    X = df[X_cols].fillna(0.0).values
    return X, y, X_cols

def train_and_eval(df, source_label, target, outdir):
    X, y, cols = pick_features(df, source_label, target)
    train_mask = np.isfinite(y)
    X = X[train_mask]; y = y[train_mask]
    
    if len(X) == 0:
        print(f"Warning: No valid data for {source_label} with target {target}")
        return {"train_r2": 0, "test_r2": 0, "train_rmse": 0, "test_rmse": 0}, None, {"cols": cols, "importances": [], "order": []}
    
    if len(X) < 4:  
        print(f"Warning: Too few samples ({len(X)}) for {source_label}, using all data for training")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_tr = rf.predict(X_train)
    pred_te = rf.predict(X_test)
    rf_metrics = {
        "train_r2": r2_score(y_train, pred_tr),
        "test_r2": r2_score(y_test, pred_te),
        "train_rmse": np.sqrt(mean_squared_error(y_train, pred_tr)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, pred_te))
    }

    # XGBoost (optional)
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_pred_te = xgb.predict(X_test)

    # Feature importances (RF)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar(range(len(cols)), importances[order])
    plt.xticks(range(len(cols)), [cols[i] for i in order], rotation=75, ha="right")
    plt.title(f"Feature Importance ({source_label})")
    plt.tight_layout()
    fp = os.path.join(outdir, f"feature_importance_{source_label}.png")
    plt.savefig(fp); plt.close()

    return rf_metrics, xgb, {"cols": cols, "importances": importances, "order": order}

def plot_accuracy_compare(scraped_metrics, api_metrics, outdir):
    labels = ["Train R2","Test R2","Train RMSE","Test RMSE"]
    s_vals = [scraped_metrics["train_r2"], scraped_metrics["test_r2"], scraped_metrics["train_rmse"], scraped_metrics["test_rmse"]]
    a_vals = [api_metrics["train_r2"], api_metrics["test_r2"], api_metrics["train_rmse"], api_metrics["test_rmse"]]

    x = np.arange(len(labels))
    w = 0.35
    plt.figure()
    plt.bar(x - w/2, s_vals, width=w, label="Scraped")
    plt.bar(x + w/2, a_vals, width=w, label="API")
    plt.xticks(x, labels, rotation=0)
    plt.legend()
    plt.title("Scraped vs API Model Performance")
    plt.tight_layout()
    fp = os.path.join(outdir, "scraped_vs_api_accuracy.png")
    plt.savefig(fp); plt.close()

def plot_engagement_trends_api(api_df, outdir):
    # Example trend: by category_id bins of duration
    d = api_df.copy()
    d["duration_min"] = d["duration_seconds"]/60.0
    d["dur_bin"] = pd.cut(d["duration_min"], bins=[0,1,3,10,60,240], labels=["<1","1-3","3-10","10-60","60+"])
    grp = d.groupby(["category_id","dur_bin"], dropna=False)["engagement_rate"].mean().reset_index()
    # Simple bar: mean engagement by dur_bin (aggregated over categories)
    agg = d.groupby("dur_bin")["engagement_rate"].mean().reset_index()
    plt.figure()
    plt.bar(agg["dur_bin"].astype(str), agg["engagement_rate"])
    plt.title("Engagement Rate by Duration Bin (API)")
    plt.tight_layout()
    fp = os.path.join(outdir, "engagement_trends_by_category_api.png")
    plt.savefig(fp); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scraped", required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--target", choices=["engagement_rate","views"], default="engagement_rate")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    scraped = pd.read_csv(args.scraped)
    api = pd.read_csv(args.api)

    # Use views for scraped data since engagement_rate is not available
    scraped_target = "views" if args.target == "engagement_rate" else args.target
    s_metrics, s_model, s_imp = train_and_eval(scraped, "scraped", scraped_target, args.outdir)
    a_metrics, a_model, a_imp = train_and_eval(api, "api", args.target, args.outdir)

    plot_accuracy_compare(s_metrics, a_metrics, args.outdir)
    plot_engagement_trends_api(api, args.outdir)

    print("=== Scraped Model ===")
    for k,v in s_metrics.items(): print(f"{k}: {v:.4f}")
    print("=== API Model ===")
    for k,v in a_metrics.items(): print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
