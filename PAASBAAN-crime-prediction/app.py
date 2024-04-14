import os
from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from geopy.geocoders import Nominatim

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/images/<Paasbaan>')
def download_file(Paasbaan):
    return send_from_directory(app.config['images'], Paasbaan)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/work.html')
def work():
    return render_template('work.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/result.html', methods=['POST'])
def predict():
    # Load the trained Random Forest model
    rfc = joblib.load('C:/Users/Yash Waldia/Desktop/crime1/PAASBAAN-crime-prediction/model/rf_model.pkl')

    if request.method == 'POST':
        # Check if 'Timestamp' key exists in form data
        if 'Timestamp' not in request.form:
            return render_template('result.html', prediction='Error: Timestamp key is missing')

        # Get user inputs
        address = request.form['Location']
        Timestamp = request.form['Timestamp']

        # Convert address to Lat and Long
        geolocator = Nominatim(user_agent="crime_prediction_app")
        location = geolocator.geocode(address, timeout=None)
        if location is None:
            return render_template('result.html', prediction='Error: Location not found')

        Lat = location.latitude
        Long = location.longitude

        # Prepare data for prediction
        data = pd.DataFrame({'Lat': [Lat], 'Long': [Long], 'Timestamp': [Timestamp]})
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])

        # Extract features from Timestamp
        data['year'] = data['Timestamp'].dt.year
        data['month'] = data['Timestamp'].dt.month
        data['day'] = data['Timestamp'].dt.day
        data['hour'] = data['Timestamp'].dt.hour

        # Select relevant features for prediction
        X = data[['Lat', 'Long', 'year', 'month', 'day', 'hour']].values

        # Make prediction
        prediction = rfc.predict(X)

        # Map predicted labels to crime types
        crime_mapping = {
            0: 'Accident',
            1: 'Drug Violation',
            2: 'Harassment',
            3: 'Missing Persons',
            4: 'Robbery',
            5: 'Towed'
        }

        # Get the predicted crime type
        if len(prediction) == 1:
            # Convert prediction to int if necessary and retrieve crime label
            predicted_crime = crime_mapping.get(int(prediction[0]), 'Place is safe, no crime expected at that Timestamp.')
        else:
            # Handle the case where prediction is not a single value (e.g., an array)
            if len(prediction) == 0:
                predicted_crime = 'Error: No prediction available'
            else:
                predicted_crime = 'Error: Invalid prediction format'

    return render_template('result.html', prediction=predicted_crime)

if __name__ == '__main__':
    app.run(debug=True)
