from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from models import load_model, predict

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests

# Load your trained model
model = load_model('model.pkl')  # Path to your saved model file

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get JSON data from the request
        data = request.get_json()
        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = predict(model, df)
        
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
