from flask import Flask,request,jsonify
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load("model/house_price_model.pkl")

@app.route("/")
def home():
    return 'House Price Prediction API is running.'

@app.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    input_df=pd.DataFrame([data])
    prediction=model.predict(input_df)
    return jsonify({
        "predicted_price":
        float(prediction[0])
    })

if __name__=="__main__":
    app.run(debug=True)