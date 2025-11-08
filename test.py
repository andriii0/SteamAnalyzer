import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

MODEL_PATH = "xgbModel6500rows/xgb_date_model.pkl"
DISCOUNT_MODEL_PATH = "xgbModel6500rows/xgb_discount_model.pkl"
FEATURES_PATH = "xgbModel6500rows/xgb_date_features.pkl"
DISCOUNT_FEATURES_PATH = "xgbModel6500rows/xgb_discount_features.pkl"

# Загрузка моделей и признаков
xgb_date_model = joblib.load(MODEL_PATH)
xgb_discount_model = joblib.load(DISCOUNT_MODEL_PATH)

xgb_date_features = joblib.load(FEATURES_PATH)
xgb_discount_features = joblib.load(DISCOUNT_FEATURES_PATH)

# Загрузка тестового датасета
df = pd.read_csv("data/test_dataset.csv", parse_dates=["month", "release_date"])


def percentage_model(df, model):
    # 1. Data preparation
    df = df[df["base_price"] > 0]  # Remove division by zero
    df = df[df["lowest_price"] > 0]  # Remove zero or negative prices
    df = df[df["lowest_price"] <= df["base_price"]]  # Remove negative discounts

    # Discount percentage
    df["discount_percent"] = (1 - df["lowest_price"] / df["base_price"]) * 100

    # Days since release
    df["days_since_release"] = (df["month"] - df["release_date"]).dt.days

    # Review ratios
    df["positive_ratio"] = df["positive"] / df["total_reviews"]
    df["negative_ratio"] = df["negative"] / df["total_reviews"]

    # Convert review_change_percent from string to numeric
    df["review_change_percent"] = (
        df["review_change_percent"]
        .str.replace("-", "0")
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    # Drop weird or missing values
    df = df.dropna(subset=[
        'app_id', 'month', 'base_price', 'lowest_price', 'discount_percent',
        'days_since_release', 'days_on_sale', 'price_changes', 'positive_ratio',
        'negative_ratio', 'total_reviews', 'review_change_percent', 'publisher'
    ])

    # Make review_change_percent numeric
    df["review_change_percent"] = (
        df["review_change_percent"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("-", "0", regex=False)
        .astype(float)
    )

    # Convert month to datetime if it's not
    if not pd.api.types.is_datetime64_any_dtype(df["month"]):
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Just for model convenience
    df["month_num"] = df["month"].dt.month

    df['publisher'] = df['publisher'].astype('category').cat.codes

    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Prepare the input data for the discount percentage model
    # Create future months for prediction
    current_date = df["month"].max()  # Use the latest month from the dataset
    future_months = [current_date + relativedelta(months=i) for i in range(1, 13)]

    results = []

    for future_date in future_months:
        # Обновляем признаки для каждого будущего месяца
        future_data = {
            'price_changes': 1,  # Примерное значение, можно модифицировать
            'base_price': df['base_price'].iloc[0],  # Используем текущую цену
            'days_since_release': (future_date - df['release_date'].iloc[0]).days,
            'month_num': future_date.month,
            'publisher': df['publisher'].iloc[0],  # Используем текущего издателя
            'days_on_sale': 10,  # Пример, можно оставить фиксированным значением
            'positive_ratio': df['positive_ratio'].iloc[0],  # Используем текущий показатель
            'negative_ratio': df['negative_ratio'].iloc[0],  # Используем текущий показатель
            'total_reviews': df['total_reviews'].iloc[0],  # Используем текущее количество отзывов
            'review_change_percent': df['review_change_percent'].iloc[0],  # Используем текущее изменение отзывов
        }

        # Преобразуем в DataFrame
        future_df = pd.DataFrame([future_data])

        # Передаем данные для предсказания процента скидки
        future_df = future_df[xgb_discount_features]  # Убедимся, что признаки соответствуют модели
        discount_pred = xgb_discount_model.predict(future_df)

        # Сохраняем результаты
        results.append({
            "month": future_date.strftime("%B %Y"),
            "predicted_discount_percentage": discount_pred[0]
        })

    # Преобразуем результаты в DataFrame для удобства
    results_df = pd.DataFrame(results)

    # Display the results
    print("\n📊 Predicted discount percentages for the next 12 months:\n")
    print(results_df[["month", "predicted_discount_percentage"]])

    return results_df

def date_model (df, model):
    # Creating a new column for time of discount
    df['discount_occurred'] = (df['lowest_price'] < df['base_price']).astype(int)
    df["days_since_release"] = (df["month"] - df["release_date"]).dt.days
    df["month_num"] = df["month"].dt.month
    df['publisher'] = df['publisher'].astype('category').cat.codes
    features = [
        'price_changes', 'base_price', 'days_since_release',
        'month_num', 'publisher'
    ]

# --- Main execution starts here ---
if __name__ == "__main__":
    # Running the model on the test dataset for the next 12 months
    print("Running discount prediction for the next 12 months...\n")

    # Run the percentage model
    results_df = percentage_model(df, xgb_discount_model)

    # Plotting the results for visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="month", y="predicted_discount_percentage", marker='o')
    plt.title("Predicted Discount Percentage Over Time for Cyberpunk 2077")
    plt.xlabel("Month")
    plt.ylabel("Predicted Discount Percentage")
