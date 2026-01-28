import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

#load data
train_df=pd.read_csv(r"C:\Users\Acer\Desktop\Data_Science_Projects\House_Price_Prediction\Kaggles_data\train.csv")
X=train_df.drop("SalePrice",axis=1)
y=train_df["SalePrice"]

#Identify columns 
cat_cols=X.select_dtypes(include="object").columns
num_cols=X.select_dtypes(exclude="object").columns

#preprocessing
num_transformer=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
])
cat_transformer=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])
preprocessor=ColumnTransformer([
    ("num",num_transformer,num_cols),
    ("cat",cat_transformer,cat_cols)
])

#model
model=XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
pipeline=Pipeline([
    ("preprocessor",preprocessor),
    ("model",model)
])

#train
pipeline.fit(X,y)

#save model
joblib.dump(pipeline,"model/house_price_model.pkl")

print("Model trained and saved successfully ! ")