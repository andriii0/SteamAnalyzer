import pandas as pd

df = pd.read_csv("data/steam_prices_prepared.csv")

def remove_nbsp(text):
    if isinstance(text, str):
        return text.replace("\xa0", "").replace("&nbsp;", "")
    return text

df = df.map(remove_nbsp)

df.to_csv("data/steam_prices_prepared.csv", index=False)

