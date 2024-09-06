from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('linear_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define route for the home page (serves HTML file)
@app.route('/')
def home():
    return render_template('index.html')  # This will serve index.html

# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    weight = float(request.form['weight'])  # Get user input from form

    # Scale the input weight and make prediction
    scaled_weight = scaler.transform(np.array([[weight]]))
    height_pred = model.predict(scaled_weight)

    return jsonify({'predicted_height': height_pred[0]})

if __name__ == "__main__":
    app.run(debug=True)
