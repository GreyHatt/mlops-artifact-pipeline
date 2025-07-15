import json
import pickle
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_config():
    """Load configuration from JSON file."""
    config_path = Path("config/config.json")
    if not config_path.exists():
        raise FileNotFoundError("Config file not found")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_model(config, digits):
    """Train a Logistic Regression model on the digits dataset."""
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['random_state']
    )
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=config['random_state']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    return model

def save_model(model, path):
    """Save the trained model to a file."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    config = load_config()
    digits = load_digits()
    model = train_model(config, digits)
    model_path = Path("models/model_train.pkl")
    save_model(model, model_path)
