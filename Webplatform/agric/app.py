from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('decision_tree_model.pkl', 'rb'))

# Updated dictionary to map prediction integers to crop names
crop_mapping = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

# Function to assign level based on predicted crop
def get_crop_level(crop_name):
    if crop_name in ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes", "jute"]:
        return 1  # Level 1
    elif crop_name in ["kidneybeans", "lentil", "maize", "mango", "mothbeans", "mungbean", "muskmelon", "orange"]:
        return 2  # Level 2
    elif crop_name in ["papaya", "pigeonpeas", "pomegranate", "rice", "watermelon"]:
        return 3  # Level 3
    else:
        return 0  # Default level if no match

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Prepare the data for prediction
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction_idx = model.predict(data)[0]  # Get the predicted crop index
        
        # Map the prediction index to the crop name
        prediction = crop_mapping[prediction_idx]
        
        # Get the level for the predicted crop
        level = get_crop_level(prediction)
        
        # Send the result and level back to the template
        return render_template('index.html', prediction=prediction.capitalize(), level=level)

if __name__ == "__main__":
    app.run(debug=True)