from typing import Sequence
import numpy as np
import torch
from utils.helpers_processing import (
    shifted_matrix
)

def _run_case_subprocess(
    features: np.ndarray,
    delays: Sequence[int],
    use_gpu: bool,
    warmup: int,
    repeats: int,
    opt_flag: bool,
    out_queue
) -> None:
    import time
    import gc

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False

    # Warmup
    for _ in range(warmup):
        _ = shifted_matrix(
            features=features,
            delays=delays,
            use_gpu=use_gpu,
            optimized_shifted=opt_flag,
            output_torch=False
        )
        if use_gpu:
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        if use_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = shifted_matrix(
            features=features,
            delays=delays,
            use_gpu=use_gpu,
            optimized_shifted=opt_flag,
            output_torch=False
        )
        if use_gpu:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    # Cleanup
    del features
    del delays
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_queue.put(float(np.median(times)))

def _benchmark_shifted_matrix(
    sizes=(2000, 5000, 10000, 250000),
    n_features_list=(1, 8, 32, 64),
    n_delays_list=(16, 64, 128),
    use_gpu: bool = True,
    warmup: int = 2,
    repeats: int = 5
) -> None:
    """
    Grid search; cada caso corre en su propio proceso (secuencial).
    """
    import multiprocessing as mp

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False

    ctx = mp.get_context("spawn")

    print(f"Benchmarking shifted_matrix | use_gpu={use_gpu}")
    for n_samples in sizes:
        for n_features in n_features_list:
            for n_delays in n_delays_list:
                features = np.random.randn(n_samples, n_features).astype(np.float32)
                delays = list(range(n_delays))

                est_bytes = n_samples * n_delays * n_features * 4
                est_mb = est_bytes / (1024 ** 2)

                def _run_in_proc(opt_flag: bool) -> float:
                    q = ctx.Queue()
                    p = ctx.Process(
                        target=_run_case_subprocess,
                        args=(features, delays, use_gpu, warmup, repeats, opt_flag, q)
                    )
                    p.start()
                    result = q.get()
                    p.join()
                    return result

                t_plain = _run_in_proc(False)
                t_opt = _run_in_proc(True)
                winner = "optimized" if t_opt < t_plain else "plain"
                print(
                    f"samples={n_samples:6d} feat={n_features:3d} delays={n_delays:3d} "
                    f"| out≈{est_mb:8.2f} MB | plain={t_plain:8.2f} ms | opt={t_opt:8.2f} ms | best={winner}"
                )
if __name__ == "__main__":
    _benchmark_shifted_matrix()