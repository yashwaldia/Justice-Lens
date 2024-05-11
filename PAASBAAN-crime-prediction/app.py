# import os
# from flask import Flask, request, render_template, send_from_directory
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from geopy.geocoders import Nominatim

# app = Flask(__name__)

# # Function to load the trained Random Forest model
# def load_model(model_path):
#     return joblib.load(model_path)

# # Load the trained Random Forest model
# rfc_model_path = 'C:/Users/Yash Waldia/Desktop/crime1/PAASBAAN-crime-prediction/model/rf_model.pkl'
# rfc_model = load_model(rfc_model_path)

# @app.route('/')
# def root():
#     return render_template('index.html')

# @app.route('/images/<Paasbaan>')
# def download_file(Paasbaan):
#     return send_from_directory(app.config['images'], Paasbaan)

# @app.route('/index.html')
# def index():
#     return render_template('index.html')

# @app.route('/work.html')
# def work():
#     return render_template('work.html')

# @app.route('/about.html')
# def about():
#     return render_template('about.html')

# @app.route('/contact.html')
# def contact():
#     return render_template('contact.html')

# @app.route('/result.html', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user inputs
#         address = request.form['Location']
#         Timestamp = request.form['Timestamp']

#         # Convert address to Lat and Long
#         geolocator = Nominatim(user_agent="crime_prediction_app")
#         location = geolocator.geocode(address, timeout=None)
#         if location is None:
#             return render_template('result.html', prediction='Error: Location not found')

#         Lat = location.latitude
#         Long = location.longitude

#         # Prepare data for prediction
#         data = pd.DataFrame({'Lat': [Lat], 'Long': [Long], 'Timestamp': [Timestamp]})
#         data['Timestamp'] = pd.to_datetime(data['Timestamp'])

#         # Extract features from Timestamp
#         data['year'] = data['Timestamp'].dt.year
#         data['month'] = data['Timestamp'].dt.month
#         data['day'] = data['Timestamp'].dt.day
#         data['hour'] = data['Timestamp'].dt.hour

#         # Select relevant features for prediction
#         X = data[['Lat', 'Long', 'year', 'month', 'day', 'hour']].values

#         # Make prediction
#         prediction = rfc_model.predict(X)

#         # Map predicted labels to crime types
#         crime_mapping = {
#             0: 'Accident',
#             1: 'Drug Violation',
#             2: 'Harassment',
#             3: 'Robbery'
#         }

#         # Extract the first element from the prediction array for dictionary lookup
#         predicted_crime_class = prediction[0]
#         predicted_crime = crime_mapping.get(predicted_crime_class, 'Place is safe, no crime expected at that Timestamp.')

#         return render_template('result.html', prediction=predicted_crime)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, send_from_directory
# import pandas as pd
# from keras.models import load_model

# app = Flask(__name__)

# # Load the trained model
# model = load_model('C:/Users/Yash Waldia/Desktop/crime1/PAASBAAN-crime-prediction/model/crime_prediction_lstm_model.keras')

# @app.route('/')
# def root():
#     return render_template('index.html')

# @app.route('/images/<Paasbaan>')
# def download_file(Paasbaan):
#     return send_from_directory(app.config['images'], Paasbaan)

# @app.route('/index.html')
# def index():
#     return render_template('index.html')

# @app.route('/work.html')
# def work():
#     return render_template('work.html')

# @app.route('/about.html')
# def about():
#     return render_template('about.html')

# @app.route('/contact.html')
# def contact():
#     return render_template('contact.html')

# # Define the route for prediction
# def predict():
#     # Get the data from the form
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     timestamp = request.form['timestamp']
    
#     # Extract day, hour, month, and year from the timestamp
#     timestamp_datetime = pd.to_datetime(timestamp)
#     day = timestamp_datetime.day
#     hour = timestamp_datetime.hour
#     month = timestamp_datetime.month
#     year = timestamp_datetime.year
#     # week = timestamp_datetime.week
#     # weekday = timestamp_datetime.weekday
#     # dayofyear = timestamp_datetime.dayofyear

    
#     # Create a DataFrame with the input data
#     input_data = pd.DataFrame({
#         'YEAR': [year],
#         'MONTH': [month],
#         'DAY': [day],
#         'HOUR': [hour],
#         'Latitude': [latitude],
#         'Longitude': [longitude]
#         })

#     # Make predictions
#     prediction = model.predict(input_data)
    
#     # Render the result template with the prediction
#     return render_template('result.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request
# import joblib
# import pandas as pd

# # Define the model directory
# model_dir = 'C:/Users/Yash Waldia/Desktop/crime1/PAASBAAN-crime-prediction/model/'

# # Define the list of crime types (replace with your actual crime types)
# crime_types = ['crime1', 'crime2', 'crime3', 'crime4']

# # Load models
# models = {}
# for crime_type in crime_types:
#     model_path = f"{model_dir}{crime_type}_model.pkl"
#     models[crime_type] = joblib.load(model_path)
#     print(f"Loaded model for crime type: {crime_type}")

# app = Flask(__name__)  # Create the application instance

# # Define the route for the index page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Define the route for the predict page
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the selected crime type from the form
#     crime_type = request.form['crime_type']
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     timestamp = request.form['timestamp']

#     # Extract day, hour, month, and year from the timestamp
#     timestamp_datetime = pd.to_datetime(timestamp)
#     day = timestamp_datetime.day
#     hour = timestamp_datetime.hour
#     month = timestamp_datetime.month
#     year = timestamp_datetime.year
#     minute = timestamp_datetime.minute

#     # Create a DataFrame with the input data
#     input_data = pd.DataFrame({
#         'YEAR': [year],
#         'MONTH': [month],
#         'DAY': [day],
#         'HOUR': [hour],
#         'MINUTE': [minute],
#         'Latitude': [latitude],
#         'Longitude': [longitude]
#     })

#     # Make predictions for each crime type
#     predictions = {}
#     for crime_type, model in models.items():
#         prediction = model.predict(input_data)[0]  # Assuming the model predicts a single class
#         predictions[crime_type] = prediction

#     # Render the prediction results
#     return render_template('result.html', predictions=predictions)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Load the trained model (assuming the path is correct)
model = load_model('C:/Users/Yash Waldia/Desktop/crime1/PAASBAAN-crime-prediction/model/crime_prediction_lstm_model.keras')

# Define the optimizer (assuming Adam)
optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed

# Compile the model with the loaded optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    timestamp = request.form['timestamp']

    # Extract features from timestamp
    timestamp_datetime = pd.to_datetime(timestamp)
    day = timestamp_datetime.day
    hour = timestamp_datetime.hour
    month = timestamp_datetime.month
    year = timestamp_datetime.year
    minute = timestamp_datetime.minute

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'YEAR': [year],
        'MONTH': [month],
        'DAY': [day],
        'HOUR': [hour],
        'MINUTE': [minute],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Handle missing values (if applicable)
    # input_data.fillna(method='ffill', inplace=True)  # Example: fill with previous value

    # Reshape input data (adjust number of features based on your model)
    reshaped_data = input_data.values.reshape((1, -1))  # Assuming all columns are features

    # Make prediction
    prediction = model.predict(reshaped_data)

    # Render the result template
    return render_template('result.html', prediction=prediction.tolist())  # Convert to list for template

if __name__ == '__main__':
  app.run(debug=True)
