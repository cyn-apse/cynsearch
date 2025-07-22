import numpy as np

def generate_chaotic_sorted_data(size=1_000_000, max_step=1_000, seed=42, output_file="chaotic_sorted_data.npy"):
    """
    Generates a chaotic but strictly increasing sorted array using log-normal step sizes.
    """
    np.random.seed(seed)
    step_sizes = np.random.lognormal(mean=2.0, sigma=1.5, size=size)
    step_sizes = np.maximum(np.round(step_sizes).astype(np.int64), 1)
    chaotic_sorted_array = np.cumsum(step_sizes)
    np.save(output_file, chaotic_sorted_array)
    print(f"[INFO] Saved chaotic sorted dataset of size {size} to '{output_file}'")
    return chaotic_sorted_array


def generate_locally_chaotic_sorted_array(size=1_000_000, window_size=5, seed=42, output_file="locally_chaotic_sorted_data.npy"):
    """
    Generates a mostly sorted array where small windows are locally shuffled.
    """
    np.random.seed(seed)
    data = np.arange(size)
    for i in range(0, size, window_size):
        np.random.shuffle(data[i:i + window_size])
    np.save(output_file, data)
    print(f"[INFO] Saved locally chaotic sorted data to '{output_file}'")
    return data


if __name__ == "__main__":
    generate_chaotic_sorted_data()
    generate_locally_chaotic_sorted_array()
