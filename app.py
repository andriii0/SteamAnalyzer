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

        # === Prepare features ===
        X_base = df  # Use the already prepared DataFrame as is (no further manipulation)

        # === Predict 12 months ahead ===
        predictions = []
        current_date = datetime.now()

        for i in range(12):
            future_date = current_date + relativedelta(months=i)
            X = X_base.copy()
            X["month_num"] = future_date.month

            # Prediction using the model
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
