from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline once, when the server starts.
# The .pkl file must be in the same folder as this app.py file
# (or update the path below to point to wherever you saved it).
model = joblib.load("gurgaon_price_model_pipeline.pkl")


def format_price(price):
    """Turn a raw rupee number into a readable string like '1.25 Cr' or '45.00 L'."""
    if price >= 10000000:
        return f"₹ {price / 10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹ {price / 100000:.2f} Lakh"
    else:
        return f"₹ {price:,.0f}"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # 1. Read every field from the submitted form
            bedroom_num = float(request.form["bedroom_num"])
            bathroom_num = float(request.form["bathroom_num"])
            balcony_num = float(request.form["balcony_num"])
            area_sqft = float(request.form["area_sqft"])
            total_floor = float(request.form["total_floor"])
            age = float(request.form["age"])
            property_type = request.form["property_type"]
            owntype = request.form["owntype"]
            facing = float(request.form["facing"])
            sector_num = float(request.form["sector_num"])

            # 2. Re-create the same engineered features used during training
            bed_x_bath = bedroom_num * bathroom_num
            area_per_floor = area_sqft / (total_floor if total_floor != 0 else 1)

            # 3. Build a single-row DataFrame with the exact column names
            #    the model pipeline was trained on
            input_row = pd.DataFrame([{
                "BEDROOM_NUM": bedroom_num,
                "BATHROON_NUM_FLOAT": bathroom_num,
                "BALCONY_NUM": balcony_num,
                "AVG_AREA_SQFT": area_sqft,
                "AREA_FLOAT": area_sqft,
                "TOTAL_FLOOR": total_floor,
                "AGE_FLOAT": age,
                "PROPERTY_TYPE": property_type,
                "OWNTYPE": owntype,
                "FACING_FLOAT": facing,
                "SECTOR_NUM": sector_num,
                "BED_X_BATH": bed_x_bath,
                "AREA_PER_FLOOR": area_per_floor,
            }])

            # 4. Predict (the model was trained on log1p(price), so we
            #    convert the prediction back to a normal rupee value)
            predicted_log_price = model.predict(input_row)[0]
            predicted_price = np.expm1(predicted_log_price)

            prediction = format_price(predicted_price)

        except Exception as e:
            error = f"Something went wrong: {e}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)