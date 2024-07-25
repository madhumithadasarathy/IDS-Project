import joblib

def load_model(model_path):
    """Load the trained model from a file."""
    return joblib.load(model_path)

def predict(model, data):
    """Make predictions using the trained model."""
    return model.predict(data)

