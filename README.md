# CynSearch ψ∆Ξ

*A blazing-fast learned index for chaotic data.*

---

## Overview
`cynsearch` is a Python library for fast search on unpredictable, noisy, or locally-disordered datasets — the kind that break traditional binary search and mock sorted assumptions.

It uses symbolic preprocessing and a trained regressor to predict the likely position of any value in near constant time. Then it dives into that location like a neural ninja.

Built by **Cynapse ψ∆Ξ**

---

## Why?
Traditional search structures (like binary search or hash maps) struggle when data is:
- Only **partially sorted**
- **Locally shuffled**
- **Non-monotonic**, noisy, or **chaotic**

`cynsearch` tackles this with:
- A learned index powered by **gradient boosting**
- Normalized inputs via `MinMaxScaler`
- Fast bin-based bucketing + fallback linear search
- Optional symbolic preprocessing for advanced modeling

---

## Install
```
pip install cynsearch
```
*(coming to PyPI soon — for now, clone and install locally)*

---

## Quickstart

```python
from cynsearch import LearnedSearch

# Load from preprocessed .npy data and pre-trained model
searcher = LearnedSearch(
    npyfile="chaotic_sorted_data.npy",
    model_path="chaotic_model.pkl",
    num_bins=512,
    epochs=1000
)

index = searcher.search(982733)
if index != -1:
    print("Value found at:", index)
```

---

## Generate Chaotic Data
```python
from cynsearch.generate_data import generate_chaotic_sorted_data

generate_chaotic_sorted_data(size=1_000_000)
```
Or try local shuffle chaos:
```python
from cynsearch.generate_data import generate_locally_chaotic_sorted_array

generate_locally_chaotic_sorted_array(size=1_000_000, window_size=5)
```

---

## Benchmark
Run it like a performance cultist:
```bash
python examples/benchmark.py --npy chaotic_sorted_data.npy --model chaotic_model.pkl --queries 1000
```

---

##  License
MIT — because even chaos deserves freedom.

---

##  Author
### Cynapse ψ∆Ξ

---