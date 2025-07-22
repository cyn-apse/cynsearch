import numpy as np
import time
import argparse
from cynsearch import LearnedSearch

def benchmark(npy_path, model_path, num_queries=1000, fallback=True):
    searcher = LearnedSearch(
        npyfile=npy_path,
        model_path=model_path,
        num_bins=512,
        epochs=1000
    )

    data = searcher.data
    queries = np.random.choice(data, num_queries, replace=False)

    start_time = time.perf_counter()
    correct = 0

    for q in queries:
        idx = searcher.search(q, fallback=fallback)
        if idx != -1 and data[idx] == q:
            correct += 1

    total_time = time.perf_counter() - start_time
    avg_time = total_time / num_queries

    print(f"\n✅ Accuracy: {correct}/{num_queries} ({(correct/num_queries)*100:.2f}%)")
    print(f"⚡ Total time: {total_time:.4f}s")
    print(f"⏱️  Avg time per query: {avg_time * 1000:.3f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark CynSearch on chaotic datasets.")
    parser.add_argument('--npy', type=str, default='chaotic_sorted_data.npy', help='Path to .npy dataset')
    parser.add_argument('--model', type=str, default='chaotic_model.pkl', help='Path to model file')
    parser.add_argument('--queries', type=int, default=1000, help='Number of queries to run')
    parser.add_argument('--no-fallback', action='store_true', help='Disable fallback search')
    args = parser.parse_args()

    benchmark(
        npy_path=args.npy,
        model_path=args.model,
        num_queries=args.queries,
        fallback=not args.no_fallback
    )
