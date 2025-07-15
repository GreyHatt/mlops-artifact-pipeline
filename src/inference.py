import pickle
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

def load_model():
    """Load the trained model from file."""
    model_path = Path("models/model_train.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Please run training first.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_digits(model, digits):
    """Generate predictions on the digit classification dataset."""
    X = digits.data
    y = digits.target
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Inference accuracy: {accuracy:.4f}")
    return y_pred, accuracy

if __name__ == "__main__":
    model = load_model()
    digits = load_digits()
    predictions, accuracy = predict_digits(model, digits)
    print(f"Predictions generated with accuracy: {accuracy:.4f}")