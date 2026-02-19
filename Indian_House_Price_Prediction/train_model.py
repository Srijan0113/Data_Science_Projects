import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(r"C:\Users\Acer\Desktop\Data_Science_Projects\Indian_House_Price_Prediction\house_price_india.csv")

# Drop ID column
df = df.drop(columns=["ID"])

# ----------------------------
# Target with Log Transform
# ----------------------------
y = np.log1p(df["Price_in_Lakhs"])

# Remove target + leakage feature
X = df.drop(columns=["Price_in_Lakhs", "Price_per_SqFt"])

# ----------------------------
# Column Types
# ----------------------------
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# ----------------------------
# Preprocessing Pipelines
# ----------------------------
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# ----------------------------
# Tuned XGBoost Model
# ----------------------------
model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

# ----------------------------
# Full Pipeline
# ----------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Model
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# Predict (Reverse Log)
# ----------------------------
log_preds = pipeline.predict(X_test)
preds = np.expm1(log_preds)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, preds)

print("Improved Model MAE (Lakhs):", round(mae, 2))

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(pipeline, "model.pkl")
print("Model saved successfully as model.pkl")