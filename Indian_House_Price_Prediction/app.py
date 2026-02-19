from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

# Load dataset for dropdown values
df = pd.read_csv(r"C:\Users\Acer\Desktop\Data_Science_Projects\Indian_House_Price_Prediction\house_price_india.csv")
df = df.drop(columns=["ID"])

@app.route("/")
def home():
    return render_template(
        "index.html",
        states=sorted(df["State"].unique()),
        property_types=sorted(df["Property_Type"].unique()),
        furnished_status=sorted(df["Furnished_Status"].unique()),
        facing_options=sorted(df["Facing"].unique()),
        owner_types=sorted(df["Owner_Type"].unique()),
        availability_status=sorted(df["Availability_Status"].unique())
    )

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "State": request.form["State"],
        "City": request.form["City"],
        "Locality": request.form["Locality"],
        "Property_Type": request.form["Property_Type"],
        "BHK": int(request.form["BHK"]),
        "Size_in_SqFt": float(request.form["Size_in_SqFt"]),
        "Year_Built": int(request.form["Year_Built"]),
        "Furnished_Status": request.form["Furnished_Status"],
        "Floor_No": int(request.form["Floor_No"]),
        "Total_Floors": int(request.form["Total_Floors"]),
        "Age_of_Property": int(request.form["Age_of_Property"]),
        "Nearby_Schools": int(request.form["Nearby_Schools"]),
        "Nearby_Hospitals": int(request.form["Nearby_Hospitals"]),
        "Public_Transport_Accessibility": request.form["Public_Transport_Accessibility"],
        "Parking_Space": request.form["Parking_Space"],
        "Security": request.form["Security"],
        "Amenities": request.form["Amenities"],
        "Facing": request.form["Facing"],
        "Owner_Type": request.form["Owner_Type"],
        "Availability_Status": request.form["Availability_Status"]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # Convert Lakhs → Crores & Rupees
    price_crore = round(prediction / 100, 2)
    price_rupees = int(prediction * 100000)
    formatted_rupees = "{:,}".format(price_rupees)

    return render_template(
        "result.html",
        crore=price_crore,
        rupees=formatted_rupees
    )

if __name__ == "__main__":
    app.run(debug=True)