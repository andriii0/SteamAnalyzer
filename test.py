import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# === Load model and feature list ===
MODEL_PATH = "xgbModel6500rows/xgb_model.pkl"
FEATURES_PATH = "xgbModel6500rows/xgb_features.pkl"

xgb_model = joblib.load(MODEL_PATH)
xgb_features = joblib.load(FEATURES_PATH)

print("✅ Model loaded:", MODEL_PATH)
print("✅ Feature count:", len(xgb_features))

# === Define baseline feature vector for Cyberpunk 2077 ===
# Используем данные из последней строки CSV (2025-10-01)
release_date = datetime(2020, 12, 9)
current_month = datetime(2025, 10, 1)

base_features = {
    "days_since_release": (current_month - release_date).days,  # количество дней с релиза
    "days_on_sale": 3,                    # из CSV
    "price_changes": 1,                   # из CSV
    "positive_ratio": 757762 / (757762 + 126765),
    "negative_ratio": 126765 / (757762 + 126765),
    "total_reviews": 884527,              # из CSV
    "base_price": 59.99,                  # из CSV
    "review_change_percent": 0.11,        # "+0.11%" → 0.11
    "month_num": 10,                      # октябрь
    "publisher_CD PROJEKT RED": 1.0       # one-hot publisher
}

print (base_features)
# === Create dataframe with all model columns ===
X_base = pd.DataFrame([base_features])
missing_cols = [col for col in xgb_features if col not in X_base.columns]
for c in missing_cols:
    X_base[c] = 0
X_base = X_base[xgb_features]

# === Predict for the next 12 months ===
results = []
current_date = datetime.now()

for i in range(12):
    future_date = current_date + relativedelta(months=i)
    X = X_base.copy()
    X["month_num"] = future_date.month  # обновляем месяц
    pred = float(xgb_model.predict(X)[0])
    results.append({
        "month": future_date.strftime("%B %Y"),
        "predicted_discount": round(pred, 2)
    })

# === Print results ===
print("\n📊 Predicted discount for 'Cyberpunk 2077' for the next 12 months:\n")
for r in results:
    print(f"{r['month']:<20} {r['predicted_discount']}% expected discount")
