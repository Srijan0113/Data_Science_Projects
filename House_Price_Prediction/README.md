#  House Price Prediction using Machine Learning  
### Kaggle – Ames Housing Dataset

---

##  Project Overview

This project focuses on building and validating machine learning models to predict residential house prices using the **Ames Housing Dataset** from Kaggle.

The workflow follows structured machine learning practices including:

- Data preprocessing
- Feature selection
- Handling missing values
- Model training
- Cross-validation
- Performance evaluation

---

##  Objectives

- Select relevant features for house price prediction  
- Train machine learning models on structured housing data  
- Validate model performance using appropriate evaluation metrics  
- Compare baseline models with ensemble methods  
- Prevent data leakage using a proper preprocessing pipeline  

---

##  Dataset

- **Dataset Name:** House Prices – Advanced Regression Techniques  
- **Source:** Kaggle  
- **Format:** CSV  
- **Target Variable:** `SalePrice`  
- **Training Samples:** ~1460  
- **Total Features:** 80+  

### Sample Features Used

- Lot Area  
- Overall Quality  
- Year Built  
- Garage Area  
- Living Area (GrLivArea)  
- Total Rooms  
- Neighborhood  
- Basement Features  

The dataset contains both numerical and categorical features.

---

##  Tools & Libraries

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Jupyter Notebook  

---

##  Machine Learning Concepts Applied

- Feature selection  
- Handling missing values (median & most frequent imputation)  
- Train–validation split  
- Log transformation of target variable  
- OneHotEncoding for categorical features  
- ColumnTransformer & Pipeline  
- Cross-validation  
- Ensemble learning  
- Evaluation using RMSLE  

---

##  Models Used

- Linear Regression (Baseline Model)  
- Decision Tree Regressor  
- Random Forest Regressor  
- XGBoost Regressor (Final Tuned Model)  

---

## Model Pipeline Structure

The entire workflow is implemented using a Scikit-learn Pipeline:

- ColumnTransformer  
- Preprocessing (Imputation + Encoding)  
- XGBoost Model  

This ensures:

- No data leakage  
- Clean and reproducible workflow  
- Structured ML development  

---