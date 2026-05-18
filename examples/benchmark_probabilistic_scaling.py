import time
import numpy as np
import pandas as pd
import properscoring as ps
import psutil
import os
from sklearn.metrics import mean_squared_error

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_scaling():
    n_rows = 200000
    n_samples = 100
    
    print(f"Benchmark Configuration: N_rows={n_rows}, N_samples={n_samples}")
    
    y_true = np.random.rand(n_rows)
    y_pred = np.random.rand(n_rows, n_samples)
    
    print("")
    print("--- CRPS Benchmark ---")
    
    # Legacy CRPS
    df_pred = pd.DataFrame({'pred': [list(row) for row in y_pred]})
    df_actual = pd.DataFrame({'true': [[v] for v in y_true]})
    
    start = time.time()
    np.mean([
        ps.crps_ensemble(actual[0], np.array(pred))
        for actual, pred in zip(df_actual['true'], df_pred['pred'])
    ])
    legacy_duration = time.time() - start
    print(f"Legacy CRPS Time: {legacy_duration:.4f}s")
    
    # Native CRPS
    start = time.time()
    np.mean(ps.crps_ensemble(y_true, y_pred, axis=1))
    native_duration = time.time() - start
    print(f"Native CRPS Time: {native_duration:.4f}s")
    print(f"CRPS Speedup: {legacy_duration / native_duration:.1f}x")

    print("")
    print("--- MSE Benchmark (The 'Expansion' Pattern) ---")
    
    # Legacy MSE
    start = time.time()
    # This mimics calculate_mse in metric_calculators.py
    actual_values = np.concatenate(df_actual['true'].values)
    pred_values = np.concatenate(df_pred['pred'].values)
    actual_expanded = np.repeat(actual_values, [len(x) for x in df_pred['pred']])
    legacy_mse = mean_squared_error(actual_expanded, pred_values)
    legacy_duration = time.time() - start
    print(f"Legacy MSE Time: {legacy_duration:.4f}s")
    
    # Native MSE (Vectorized Broadcasting)
    start = time.time()
    # (N, 1) - (N, S) -> (N, S)**2 -> mean
    native_mse = np.mean((y_true[:, np.newaxis] - y_pred)**2)
    native_duration = time.time() - start
    print(f"Native MSE Time: {native_duration:.4f}s")
    print(f"MSE Speedup: {legacy_duration / native_duration:.1f}x")
    
    print("")
    print(f"MSE Parity: {abs(legacy_mse - native_mse) < 1e-10}")

if __name__ == "__main__":
    benchmark_scaling()
