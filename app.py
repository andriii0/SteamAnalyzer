import asyncio
import os
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from dateutil.relativedelta import relativedelta
import pandas as pd

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


# Function to generate fake data for the games
def get_fake_data(game_name):
    if game_name == "PEAK":
        return {
            "days_since_release": 150,  # Fake number of days since release
            "days_on_sale": 1,
            "price_changes": 1,
            "positive_ratio": 152949 / (152949 + 10228),
            "negative_ratio": 10228 / (152949 + 10228),
            "total_reviews": 163177,
            "base_price": 7.99,
            "review_change_percent": -0.01,
            "month_num": 10,  # October
            "publisher_Landfall Aggro Crab": 1.0,
        }
    elif game_name == "Rust":
        return {
            "days_since_release": 2720,  # Fake number of days since release
            "days_on_sale": 4,
            "price_changes": 1,
            "positive_ratio": 1105109 / (1105109 + 164409),
            "negative_ratio": 164409 / (1105109 + 164409),
            "total_reviews": 1269518,
            "base_price": 39.99,
            "review_change_percent": -0.07,
            "month_num": 10,  # October
            "publisher_Facepunch Studios": 1.0,
        }
    elif game_name == "Cyberpunk 2077":
        return {
            "days_since_release": 1800,
            "days_on_sale": 3,
            "price_changes": 1,
            "positive_ratio": 757762 / (757762 + 126765),
            "negative_ratio": 126765 / (757762 + 126765),
            "total_reviews": 884527,
            "base_price": 59.99,
            "review_change_percent": 0.11,
            "month_num": 10,
            "publisher_CD PROJEKT RED": 1.0,
        }
    return None


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
        # === Get fake data for each game ===
        base_features = get_fake_data(game_name)

        if base_features is None:
            flash(f"Game '{game_name}' not found. Please enter a valid game name.")
            return redirect(url_for("index"))

        print(f"[+] Using fake data for '{game_name}'")
        print(base_features)

        # Prepare the fake data as DataFrame
        X_base = pd.DataFrame([base_features])

        # Add missing features
        missing_cols = [col for col in xgb_model_features if col not in X_base.columns]
        for c in missing_cols:
            X_base[c] = 0
        X_base = X_base[xgb_model_features]

        # Predict for the next 12 months
        predictions = []
        current_date = datetime.now()

        for i in range(12):
            future_date = current_date + relativedelta(months=i)
            X = X_base.copy()
            X["month_num"] = future_date.month  # Update month for each prediction

            # Predict with the same model
            pred_value = float(xgb_model.predict(X)[0])

            predictions.append({
                "game": game_name,
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
