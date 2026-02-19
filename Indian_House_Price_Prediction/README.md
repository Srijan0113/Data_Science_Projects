#  Indian House Price Prediction (End-to-End ML Project)

##  Project Overview

This project is an end-to-end Machine Learning web application that predicts house prices based on property features such as:

- State
- City
- Property Type
- BHK
- Size (SqFt)
- Year Built
- Furnishing Status
- Amenities
- Nearby Facilities
- And more

The project includes:
- Data analysis
- Feature preprocessing
- Model training (XGBoost)
- Evaluation
- Flask web deployment
- Dark luxury UI frontend

---

##  Problem Statement

Build a regression model to predict House Price using structured housing features and deploy it as a web application.

---

## Dataset

- Structured Indian housing dataset
- Contains categorical and numerical features
- Target variable: `Price_in_Lakhs`

 During analysis, strong data leakage was identified via the feature `Price_per_SqFt` (which is derived from the target).  
This feature was removed to ensure model integrity.

Further correlation analysis showed weak relationships between independent features and price, indicating possible synthetic/noisy data distribution.

---

##  Data Preprocessing

- Dropped ID column
- Removed leakage feature (`Price_per_SqFt`)
- Handled missing values:
  - Numerical → Median Imputation
  - Categorical → Most Frequent Imputation
- OneHotEncoding for categorical variables
- ColumnTransformer for clean preprocessing pipeline
- Log transformation applied to stabilize variance

---

##  Model Used
- **XGBoost Regressor**

---

##  Model Evaluation

- Metric Used: Mean Absolute Error (MAE)
- Final MAE ≈ 120+ Lakhs

Due to weak feature-target correlation in the dataset, model performance is limited by data quality rather than modeling approach.

---

##  Web Application

Built using:

- Flask
- HTML
- CSS (Dark Luxury Theme)

Features:
- User-friendly form input
- Real-time prediction
- Output displayed in:
  - Crores
  - Formatted Indian Rupees
- Clean UI with glassmorphism styling

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML/CSS

---

##  How to Run the Project

###  Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost flask joblib

### Train The Model
python train_model.py

### Run flask app
python app.py

### Then open: http://127.0.0.1:5000