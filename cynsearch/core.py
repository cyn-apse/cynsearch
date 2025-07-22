import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import os

class LearnedSearch:
    def __init__(self, npyfile, num_bins=1024, epochs=1000, model_path=None):
        self.npyfile = npyfile
        self.num_bins = num_bins
        self.epochs = epochs
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler()
        self.bin_bounds = []

        if not os.path.exists(npyfile):
            raise FileNotFoundError(f"[ERROR] Data file '{npyfile}' not found.")
        self.data = np.load(npyfile)
        self.n = len(self.data)

        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        elif self.model_path:
            self.model = HistGradientBoostingRegressor(max_iter=self.epochs, max_depth=10)
            self.fit(self.data)
        else:
            print("[INFO] No model will be used. Only raw data is available.")

    def fit(self, data):
        print("[INFO] Training new model...")
        X = data.reshape(-1, 1).astype(np.float32)
        X_log = np.log1p(X)  # Handle exponential data better
        X_scaled = self.scaler.fit_transform(X_log)

        bin_size = self.n / self.num_bins
        y = (np.arange(self.n) // bin_size).clip(0, self.num_bins - 1).astype(int)

        self.model.fit(X_scaled, y)

        if self.model_path:
            dump((self.model, self.scaler, self.num_bins), self.model_path)
            print(f"[INFO] Model saved to '{self.model_path}'")

        self._compute_bin_bounds()

    def load_model(self):
        print(f"[INFO] Loading model from '{self.model_path}'...")
        self.model, self.scaler, self.num_bins = load(self.model_path)
        self._compute_bin_bounds()

    def _compute_bin_bounds(self):
        bin_size = self.n // self.num_bins
        self.bin_bounds = [(i * bin_size, (i + 1) * bin_size) for i in range(self.num_bins)]
        self.bin_bounds[-1] = (self.bin_bounds[-1][0], self.n)

    def search(self, target, fallback=True):
        if self.model is None:
            raise RuntimeError("No trained model available for search.")

        z = self.scaler.transform(np.log1p(np.array([[target]], dtype=np.float32)))
        predicted_bin = int(round(self.model.predict(z)[0]))
        predicted_bin = max(0, min(predicted_bin, self.num_bins - 1))

        start, end = self.bin_bounds[predicted_bin]
        subarray = self.data[start:end]
        matches = np.where(subarray == target)[0]
        if matches.size > 0:
            return start + matches[0]

        if fallback:
            for offset in range(1, 3):
                for neighbor in [predicted_bin - offset, predicted_bin + offset]:
                    if 0 <= neighbor < self.num_bins:
                        s, e = self.bin_bounds[neighbor]
                        sub = self.data[s:e]
                        match = np.where(sub == target)[0]
                        if match.size > 0:
                            return s + match[0]
        return -1
