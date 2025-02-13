from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    # Convert data into DataFrame
    df = pd.DataFrame([data])
    # Make prediction
    prediction = model.predict(df)
    # Return prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

# Instructions:
# - Save your trained model as 'model.pkl' using pickle.
# - Ensure Flask is installed: pip install flask
# - Run this script to start the Flask server.
# - Send a POST request with data to http://127.0.0.1:5000/predict to get predictions.
