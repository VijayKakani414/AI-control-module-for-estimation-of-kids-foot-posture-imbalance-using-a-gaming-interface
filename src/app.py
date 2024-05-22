from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the AI trained models for Left and Right pressure deficiency
model_left = joblib.load('AI_model_left.joblib')
model_right = joblib.load('AI_model_right.joblib')

@app.route('/predict_pressure', methods=['POST'])
def predict_pressure():
    try:
        # Get input from the client
        measured_info = request.json

        # Convert measured_info to DataFrame
        test_case = pd.DataFrame(measured_info)

        # Predict LPressureVal and RPressureVal
        predicted_l_pressure = model_left.predict([[
            test_case['나이(Age)'].values[0],
            test_case['키(Height)'].values[0],
            test_case['몸무게(Weight)'].values[0]
        ]])[0]
        predicted_r_pressure = model_right.predict([[
            test_case['나이(Age)'].values[0],
            test_case['키(Height)'].values[0],
            test_case['몸무게(Weight)'].values[0]
        ]])[0]

        # Calculate Pressure Deficiencies
        lpd = test_case['Measured_LPressureVal'].values[0] - predicted_l_pressure
        rpd = test_case['Measured_RPressureVal'].values[0] - predicted_r_pressure

        # Determine the result
        if lpd < 0 and rpd < 0:
            if lpd > rpd:
                result = "LEFT Pressure Deficiency exists"
            elif lpd < rpd:
                result = "RIGHT Pressure Deficiency exists"
            else:
                result = "Both LEFT Pressure Deficiency and RIGHT Pressure Deficiency exist"
        elif lpd < 0:
            result = "LEFT Pressure Deficiency exists"
        elif rpd < 0:
            result = "RIGHT Pressure Deficiency exists"
        else:
            result = "Candidate doesn't have LEFT Pressure Deficiency or RIGHT Pressure Deficiency"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
