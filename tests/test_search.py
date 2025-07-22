import numpy as np
import os
import pytest
from cynsearch import LearnedSearch

def generate_dummy_data(n=1000):
    data = np.arange(n)
    np.save("_dummy_data.npy", data)
    return data

def test_model_training_and_search():
    data = generate_dummy_data()
    model_path = "_dummy_model.pkl"
    
    # Ensure no leftover model
    if os.path.exists(model_path):
        os.remove(model_path)

    searcher = LearnedSearch(
        npyfile="_dummy_data.npy",
        model_path=model_path,
        num_bins=32,
        epochs=100
    )

    for val in [0, 100, 500, 999]:
        idx = searcher.search(val)
        assert idx != -1, f"Value {val} not found"
        assert searcher.data[idx] == val

    os.remove("_dummy_data.npy")
    os.remove(model_path)

def test_model_reload():
    data = generate_dummy_data()
    model_path = "_dummy_model_reload.pkl"

    # First pass: train and save model
    searcher1 = LearnedSearch(
        npyfile="_dummy_data.npy",
        model_path=model_path,
        num_bins=32,
        epochs=50
    )

    # Second pass: reload model
    searcher2 = LearnedSearch(
        npyfile="_dummy_data.npy",
        model_path=model_path
    )

    for val in [123, 456, 789]:
        idx = searcher2.search(val)
        assert idx != -1
        assert searcher2.data[idx] == val

    os.remove("_dummy_data.npy")
    os.remove(model_path)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
