import asyncio
import os
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from Parser.gameIdSearcher import GameIdSearch
from dateutil.relativedelta import relativedelta
from Parser.DataPrep import get_game_data

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

        # === Call our existing function to get game data (parsing, price, and reviews) ===
        df = asyncio.run(get_game_data(game_name))

        if df.empty:
            flash(f"No data found for '{game_name}'.")
            return redirect(url_for("index"))

        source_frame = df.attrs.get("processed_source")

        latest_idx = None
        if source_frame is not None and not source_frame.empty:
            if "month" in source_frame.columns:
                valid_months = source_frame["month"].dropna()
                if not valid_months.empty:
                    latest_idx = valid_months.idxmax()
            if latest_idx is None:
                latest_idx = source_frame.index[-1]
        else:
            latest_idx = df.index[-1]

        X_base = df.loc[[latest_idx]].copy()

        release_date_value = None
        if source_frame is not None and not source_frame.empty and "release_date" in source_frame.columns:
            release_val = source_frame.loc[latest_idx, "release_date"]
            if release_val is not None:
                if hasattr(release_val, "to_pydatetime"):
                    release_date_value = release_val.to_pydatetime().date()
                elif isinstance(release_val, datetime):
                    release_date_value = release_val.date()
                else:
                    try:
                        release_date_value = datetime.fromisoformat(str(release_val)).date()
                    except ValueError:
                        release_date_value = None

        X_base = X_base[xgb_model_features]

        # === Predict 12 months ahead ===
        predictions = []
        current_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        for i in range(12):
            future_date = current_date + relativedelta(months=i)
            X = X_base.copy()
            idx = X.index[0]
            X.at[idx, "month_num"] = future_date.month

            if release_date_value is not None and "days_since_release" in X.columns:
                days_delta = (future_date.date() - release_date_value).days
                X.at[idx, "days_since_release"] = days_delta

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
