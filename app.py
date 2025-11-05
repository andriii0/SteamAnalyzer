# === Flask web app for predicting Steam game price changes ===
import asyncio
import re
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

from Parser.GameIdSearch import GameIdSearch
from Parser.PriceHistoryScraper import PriceHistoryScraper
from Parser.ReviewHistoryScraper import ReviewHistoryScraper


# === Load pretrained XGBoost model ===
MODEL_PATH = "xgbModel6500rows/xgb_model.pkl"
FEATURES_PATH = "xgbModel6500rows/xgb_features.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model file 'xgb_model.pkl' not found! Please train and export it first.")
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Model features file 'xgb_features.pkl' not found! Please export it during training.")

xgb_model = joblib.load(MODEL_PATH)
xgb_model_features = joblib.load(FEATURES_PATH)
print("✅ Loaded model and feature list:", MODEL_PATH)


# === Helper: prepare features for model input ===
def prepare_features(price_df, review_df, model_features):
    """Prepare all features needed for the XGBoost model.

    - Robust numeric parsing (handles commas as thousands separators).
    - Safe parsing of review_change_percent (e.g. "+0,13%" -> 0.13, "-0.5%" -> -0.5).
    - If review_change_percent missing or empty, compute from total_reviews pct_change().
    - One-hot encode publisher and align columns with model_features.
    """
    # Merge price + review on month
    df = pd.merge(price_df, review_df, on="month", how="left", suffixes=("_price", "_review"))

    # --- Helper to parse numeric strings (handles commas, spaces, percent removed elsewhere) ---
    def to_num(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            if s in ["-", "", "nan", "None", "—"]:
                return np.nan
            # replace non-breaking spaces and common thousands separators
            s = s.replace("\xa0", "").replace(" ", "").replace(",", "")
            # keep digits, dot and minus
            s = re.sub(r"[^\d\.\-]", "", s)
            if s == "":
                return np.nan
            return float(s)
        except Exception:
            return np.nan

    # Clean common numeric columns (but skip review_change_percent for special handling)
    numeric_cols = [
        "base_price",
        "lowest_price",
        "price_changes",
        "days_on_sale",
        "positive",
        "negative",
        "total_reviews",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    # --- Handle review_change_percent robustly ---
    if "review_change_percent" in df.columns:
        # normalize: convert to string, replace comma->dot, strip, extract signed number
        s = df["review_change_percent"].astype(str).fillna("").str.replace(",", ".", regex=False)
        # extract first signed number (handles + / - / no sign)
        extracted = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
        df["review_change_percent"] = pd.to_numeric(extracted, errors="coerce")
        # if all NaN or most values missing, we'll compute next (but keep what we have)
    else:
        df["review_change_percent"] = np.nan

    # If total_reviews missing, try calculate from positive/negative
    if "total_reviews" not in df.columns or df["total_reviews"].isna().all():
        df["total_reviews"] = df["positive"].fillna(0) + df["negative"].fillna(0)

    # If review_change_percent is mostly empty (or entirely), compute from total_reviews pct change
    # Condition: compute when >50% of entries are NaN (or all)
    nan_ratio = df["review_change_percent"].isna().mean()
    if nan_ratio > 0.5 or df["review_change_percent"].isna().all():
        # pct_change * 100; fill first with 0
        df["review_change_percent"] = df["total_reviews"].pct_change().fillna(0.0) * 100.0

    # After that, fill any remaining NaN with 0.0
    df["review_change_percent"] = df["review_change_percent"].fillna(0.0)

    # --- Further numeric cleaning for any leftovers (safety) ---
    # ensure all numeric columns are numeric
    all_num_cols = numeric_cols + ["review_change_percent"]
    for col in all_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- SAFETY FILTERS (same as in training) ---
    # require base_price and lowest_price to be positive and lowest <= base
    if "base_price" in df.columns:
        df = df[df["base_price"] > 0]
    if "lowest_price" in df.columns:
        df = df[df["lowest_price"] > 0]
    if "lowest_price" in df.columns and "base_price" in df.columns:
        df = df[df["lowest_price"] <= df["base_price"]]

    # --- CALCULATED FIELDS ---
    # Discount %
    if "lowest_price" in df.columns and "base_price" in df.columns:
        df["discount_percent"] = (1 - df["lowest_price"] / df["base_price"]) * 100
    else:
        df["discount_percent"] = 0.0

    # Release date (use real or fallback)
    if "release_date" in price_df.columns and pd.notna(price_df["release_date"].iloc[0]):
        release_date = pd.to_datetime(price_df["release_date"].iloc[0], errors="coerce")
    else:
        release_date = datetime.now() - pd.Timedelta(days=180)

    # Days since release
    df["days_since_release"] = (pd.to_datetime(df["month"], errors="coerce") - release_date).dt.days

    # Positive / negative ratios (ensure no division by zero)
    df["positive_ratio"] = np.where(df["total_reviews"] > 0,
                                    df["positive"] / df["total_reviews"], 0.0)
    df["negative_ratio"] = np.where(df["total_reviews"] > 0,
                                    df["negative"] / df["total_reviews"], 0.0)

    # Month number
    df["month_num"] = pd.to_datetime(df["month"], errors="coerce").dt.month.fillna(1).astype(int)

    # --- SELECT LATEST ROW ---
    if df.empty:
        raise ValueError("No valid rows after filtering price/review data.")
    latest = df.iloc[-1]

    # --- Publisher (string → cleaned single string) ---
    publisher = None
    if "publisher" in price_df.columns:
        publisher_val = str(price_df["publisher"].iloc[0]).strip()
        if publisher_val and publisher_val.lower() not in ["nan", "none", "-"]:
            # remove commas/semicolons, normalize whitespace
            publisher = re.sub(r"[,;]+", "", publisher_val)
            publisher = re.sub(r"\s+", " ", publisher).strip()
        else:
            publisher = None

    # --- Assemble feature dict (use latest row values) ---
    features = {
        "days_since_release": float(latest.get("days_since_release", 0) or 0),
        "days_on_sale": float(latest.get("days_on_sale", 0) or 0),
        "price_changes": float(latest.get("price_changes", 0) or 0),
        "positive_ratio": float(latest.get("positive_ratio", 0.8) or 0.8),
        "negative_ratio": float(latest.get("negative_ratio", 0.2) or 0.2),
        "total_reviews": float(latest.get("total_reviews", 0) or 0),
        "base_price": float(latest.get("base_price", 0) or 0),
        "review_change_percent": float(latest.get("review_change_percent", 0) or 0),
        "month_num": int(latest.get("month_num", 1) or 1),
        "publisher": publisher,
    }

    # Debug print (optional)
    print("\n==================== DEBUG: MODEL FEATURES ====================")
    for k, v in features.items():
        print(f"{k:<25} {v}")
    print("===============================================================\n")

    # --- Convert to DataFrame and one-hot publisher ---
    X = pd.DataFrame([features])
    X = pd.get_dummies(X, columns=["publisher"], dtype=float)

    # Add missing model features (0-fill)
    missing_cols = [c for c in model_features if c not in X.columns]
    if missing_cols:
        X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing_cols)], axis=1)

    # Ensure column order matches model_features
    X = X[model_features]

    print("\n==== PRICE DATA ====")
    print(price_df.head())
    print("\n==== REVIEW DATA ====")
    print(review_df.head())

    # Warn if mismatch (debug)
    extra_cols = [c for c in X.columns if c not in model_features]
    if missing_cols or extra_cols:
        print("⚠️ Feature mismatch detected:")
        if missing_cols:
            print("  Missing in input:", missing_cols)
        if extra_cols:
            print("  Extra in input:", extra_cols)

    return X


# === Flask app ===
app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for flash messages


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    game_name = request.form.get("games", "").strip()
    if not game_name:
        flash("Please enter a game title.")
        return redirect(url_for("index"))

    try:
        # === Get Steam App ID ===
        app_id = asyncio.run(GameIdSearch.get_app_id(game_name))
        if not app_id:
            flash(f"Could not find Steam app ID for '{game_name}'.")
            return redirect(url_for("index"))

        print(f"[+] Fetching data for app_id={app_id}")

        # === Scrape data ===
        price_scraper = PriceHistoryScraper(app_id)
        review_scraper = ReviewHistoryScraper(app_id)

        price_df = asyncio.run(price_scraper.get_price_history_table())
        review_df = asyncio.run(review_scraper.get_review_history_table())

        if price_df.empty:
            flash(f"No price history found for '{game_name}'.")
            return redirect(url_for("index"))
        if review_df.empty:
            flash("Review data missing, using price-only features.")

        # === Prepare features ===
        X_base = prepare_features(price_df, review_df, xgb_model_features)

        # === Predict 12 months ahead ===
        predictions = []
        current_date = datetime.now()

        for i in range(12):
            future_date = current_date + relativedelta(months=i)
            X = X_base.copy()
            X["month_num"] = future_date.month

            pred_value = float(xgb_model.predict(X)[0])

            predictions.append({
                "game": game_name,
                "app_id": app_id,
                "month_year": future_date.strftime("%B %Y"),
                "price": f"{pred_value:.2f}% expected discount",
            })

        return render_template("results.html", predictions=predictions, original_input=game_name)

    except Exception as e:
        print("Error:", e)
        flash(f"Error during prediction: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
