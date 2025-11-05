import pandas as pd
from dateutil import parser

prices = pd.read_csv("data/steam_prices_prepared.csv")
sales = pd.read_csv("../data/steam_sales.csv")

sales = sales.rename(columns={
    "name": "game_name",
    "Rating": "rating",
    "#Reviews": "reviews",
    "Release Date": "release_date"
})[["game_name", "rating", "reviews", "release_date"]]

def parse_release_date(date_str):
    try:
        return parser.parse(date_str)
    except Exception:
        return pd.NaT

sales["release_date"] = sales["release_date"].apply(parse_release_date)

sales["reviews_num"] = (
    sales["reviews"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

sales = sales.sort_values("reviews_num", ascending=False).drop_duplicates("game_name")

merged = prices.merge(sales.drop(columns="reviews_num"), on="game_name", how="left")

missing_release_date = merged[merged["release_date"].isna()]
print(missing_release_date.head())

merged.to_csv("steam_prices_merged_fixed.csv", index=False)
