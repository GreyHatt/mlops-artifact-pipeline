import json
import os
import pytest
import pickle
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.train import load_config, train_model, save_model

def test_config_loading():
    """Test configuration file loading."""
    config = load_config()
    assert isinstance(config, dict)
    assert all(key in config for key in ['C', 'solver', 'max_iter', 'random_state'])
    assert isinstance(config['C'], (float))
    assert isinstance(config['solver'], str)
    assert isinstance(config['max_iter'], int)
    assert isinstance(config['random_state'], int)

def test_model_creation():
    """Test model creation with Training Method created in previous branch."""
    config = load_config()
    digits = load_digits()
    model = train_model(config, digits)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'classes_')

def test_model_accuracy():
    """Test model training and accuracy."""
    config = load_config()
    digits = load_digits()
    X = digits.data
    y = digits.target
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=config['random_state']
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.80, f"Model accuracy {accuracy} is below threshold"