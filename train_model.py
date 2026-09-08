import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

df = pd.read_csv('gurgaon_10k.csv', low_memory=False)


df = df[df['LOCALITY'].str.contains("Sector", na=False)].copy()


df['SECTOR_NUM'] = pd.to_numeric(
    df['LOCALITY'].str.extract(r'Sector\s*(\d+)')[0],
    errors='coerce'
)

df = df.dropna(subset=['BEDROOM_NUM', 'BALCONY_NUM', 'PRICE', 'TOTAL_FLOOR', 'SECTOR_NUM'])


def convert_area_to_sqft(area):
    """
    FIX #1: previous version returned `num` in every branch regardless of
    unit detected, so acres/sq.m/sq.yards were silently treated as sq.ft.
    Now each unit is actually converted to square feet.
    """
    area = str(area).lower().strip()
    area = area.replace(',', '')

    num_match = re.findall(r'\d+\.?\d*', area)
    if not num_match:
        return np.nan

    num = float(num_match[0])

    if 'sq.yard' in area or 'sq yard' in area or 'sq.yards' in area:
        return num * 9.0           
    elif 'sq.m' in area or 'sq. meter' in area or 'sqm' in area:
        return num * 10.7639      
    elif 'acre' in area:
        return num * 43560.0        
    elif 'sq.ft' in area or 'sqft' in area:
        return num
    else:
        return num  


def convert_price(price):
    price = str(price).lower().strip()
    price = price.replace(',', '')

    num = re.findall(r'\d+\.?\d*', price)
    if len(num) == 0:
        return np.nan

    num = float(num[0])

    if 'cr' in price:
        return num * 10000000
    elif 'l' in price:
        return num * 100000
    elif 'k' in price:
        return num * 1000
    else:
        return num


df['PRICE_FLOAT'] = df['PRICE'].apply(convert_price)
df['AREA_FLOAT'] = df['AREA'].apply(convert_area_to_sqft)

df['MIN_PRICE'] = df['MIN_PRICE'].astype(float)
df['MAX_PRICE'] = df['MAX_PRICE'].astype(float)

df['AGE_FLOAT'] = df['AGE'].astype(float)
df['FACING_FLOAT'] = df['FACING'].astype(float)
df['BATHROON_NUM_FLOAT'] = df['BATHROOM_NUM'].astype(float)
df['SECTOR_NUM'] = pd.to_numeric(df['SECTOR_NUM'], errors='coerce').astype(float)

df = df.dropna(subset=['PRICE_FLOAT'])

df['BED_X_BATH'] = df['BEDROOM_NUM'] * df['BATHROON_NUM_FLOAT']
df['AREA_PER_FLOOR'] = df['AREA_FLOAT'] / df['TOTAL_FLOOR'].replace(0, 1)
df.loc[:, 'AVG_AREA_SQFT'] = df[['MIN_AREA_SQFT', 'MAX_AREA_SQFT']].mean(axis=1)


features = [
    "BEDROOM_NUM",
    "BATHROON_NUM_FLOAT",
    "BALCONY_NUM",
    "AVG_AREA_SQFT",
    "AREA_FLOAT",
    "TOTAL_FLOOR",
    "AGE_FLOAT",
    "PROPERTY_TYPE",
    "OWNTYPE",
    "FACING_FLOAT",
    "SECTOR_NUM",
    "BED_X_BATH",
    "AREA_PER_FLOOR",
]
target = "PRICE_FLOAT"

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["AREA_FLOAT", "BED_X_BATH", "AREA_PER_FLOOR"])

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


cat_cols = [
    "PROPERTY_TYPE",
    "OWNTYPE",
    "FACING_FLOAT",
    "SECTOR_NUM"
]
num_cols = [c for c in features if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="passthrough"
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])


model_pipeline.fit(X_train, y_train_log)

y_pred_log = model_pipeline.predict(X_test)


r2 = r2_score(y_test_log, y_pred_log)
rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

print("R2 Score:", r2)
print("RMSE:", rmse)

joblib.dump(model_pipeline, "gurgaon_price_model_pipeline.pkl")
print("Saved model to gurgaon_price_model_pipeline.pkl")
