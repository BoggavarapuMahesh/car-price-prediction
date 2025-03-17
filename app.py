from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('car_price_model.pkl')  # Load the trained ANN model
preprocessor = joblib.load('preprocessor.pkl')  # Load the preprocessor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        data = {
            'Make': request.form['Make'],
            'Model': request.form['Model'],
            'Engine Fuel Type': request.form['Engine Fuel Type'],
            'Engine HP': float(request.form['Engine HP']),
            'Engine Cylinders': int(request.form['Engine Cylinders']),
            'Transmission Type': request.form['Transmission Type'],
            'Driven_Wheels': request.form['Driven_Wheels'],
            'Number of Doors': int(request.form['Number of Doors']),
            'Market Category': request.form['Market Category'],
            'Vehicle Size': request.form['Vehicle Size'],
            'Vehicle Style': request.form['Vehicle Style'],
            'Highway MPG': int(request.form['highway MPG']),
            'City MPG': int(request.form['city mpg']),
            'Popularity': int(request.form['Popularity'])
        }

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Preprocess data
        df_prepared = preprocessor.transform(df)

        # Predict price
        prediction = model.predict(df_prepared)
        inr_price = prediction[0] * 87.06  # Convert USD to INR

        # Return response as JSON
        data['Predicted Price'] = f"${prediction[0]:,.2f} | INR {inr_price:,.2f}"
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
