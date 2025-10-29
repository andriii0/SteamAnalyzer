import pandas as pd

df = pd.read_csv("data/steam_prices_prepared.csv")

def remove_thousand_separators(text):
    if isinstance(text, str):
        return text.replace(",", "")
    return text

df = df.apply(lambda col: col.map(remove_thousand_separators))

numeric_cols = ["positive", "negative", "total_reviews"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv("data/steam_prices_prepared.csv", index=False, encoding="utf-8-sig")

