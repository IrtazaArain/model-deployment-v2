from flask import Flask, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model_cat_4.joblib')

# Define routes
@app.route('/', methods=['GET'])
def predict():
    # Replace 'input.csv' with the path to your CSV file
    file_path = 'input.csv'
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Convert the features DataFrame to a 2D array for model prediction
    data_2d_array = df.values
    # Make predictions
    predictions = model.predict(data_2d_array)
    # Combine the predictions with longitude and latitude
    results = []
    for i in range(len(predictions)):
        result = [float(predictions[i]), float(df.iloc[i]['Latitude']), float(df.iloc[i]['Longitude'])]
        results.append(result)
    # Return the combined data as a JSON response
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
