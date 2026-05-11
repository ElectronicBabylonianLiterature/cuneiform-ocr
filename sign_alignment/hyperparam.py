"""
Hyperparameter search for the PointSetRegistrationOptimizer.

Provides a coordinate-wise (one-parameter-at-a-time) sweep over PSR
hyperparameters.  The evaluation function is injected by the caller so
that this module remains independent of the evaluation pipeline.
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional


# Search axes for the coordinate-wise sweep
SEARCH_AXES: Dict[str, List] = {
    'lambda_data':     [0.5, 1.0, 2.0, 5.0, 10.0],
    'lambda_anchor':   [0.005, 0.01, 0.05, 0.1],
    'lambda_seq':      [0.01, 0.05, 0.1, 0.5],
    'lambda_height':   [0.0, 0.005, 0.01, 0.05],
    'lambda_rows':     [1.0, 2.0, 5.0, 10.0],
    'lambda_boundary': [0.0, 0.5, 1.0, 5.0],
    'sigma_factor':    [1.0, 1.5, 2.0, 2.5],
    'w_noise':         [0.05, 0.1, 0.2],
}


def hyperparameter_search(
    context,
    fragment_ids: List[str],
    eval_fn: Callable,
    base_params: dict,
    output_dir: str = "evaluation_results",
    full_num_iterations: Optional[int] = None,
) -> Dict:
    """
    Fast coordinate-wise (one-parameter-at-a-time) hyperparameter sweep
    for the PointSetRegistrationOptimizer.

    Sweeps PSR-specific parameters: lambda_data, lambda_anchor, lambda_seq,
    lambda_height, lambda_rows, lambda_boundary, sigma_factor, w_noise.
    Runs two rounds to allow parameters to adapt to each other.

    Args:
        context: Pipeline context (CropContext).
        fragment_ids: Fragment IDs to evaluate on during the sweep.
        eval_fn: Callable ``(context, fragment_ids, params) -> float`` that
            returns mAP (higher is better).  A return value of -1.0 signals
            an error.
        base_params: Starting PSR parameter dict (typically with a reduced
            ``num_iterations`` for speed).
        output_dir: Directory to save ``hyperparam_search.json``.
        full_num_iterations: If provided, overrides ``base_params['num_iterations']``
            in the returned best_params so callers get the full-iteration config.

    Returns:
        Dict with keys ``best_params``, ``best_mAP``, ``all_results``.
    """
    best_params = dict(base_params)

    total_evals = 2 * sum(len(v) for v in SEARCH_AXES.values())
    print(f"Coordinate-wise sweep: {total_evals} evaluations "
          f"(2 rounds × {sum(len(v) for v in SEARCH_AXES.values())} candidates)")

    best_score = eval_fn(context, fragment_ids, best_params)
    print(f"  Baseline mAP = {best_score:.4f}")

    all_search_results = []
    eval_count = 0

    for round_idx in range(2):
        print(f"\n--- Round {round_idx + 1} ---")
        for key, candidates in SEARCH_AXES.items():
            old_val = best_params[key]
            round_best_val = old_val
            round_best_score = best_score

            for val in candidates:
                if val == old_val:
                    continue  # already evaluated
                trial = dict(best_params)
                trial[key] = val
                eval_count += 1

                t0 = time.time()
                score = eval_fn(context, fragment_ids, trial)
                elapsed = time.time() - t0

                entry = {
                    'round': round_idx + 1,
                    'param': key,
                    'value': val,
                    'mAP': score,
                    'elapsed_s': elapsed,
                    'full_params': {k: v for k, v in trial.items()},
                }
                all_search_results.append(entry)

                tag = ""
                if score > round_best_score:
                    round_best_score = score
                    round_best_val = val
                    tag = " *"

                print(f"  [{eval_count:3d}] {key}={val:<10}  mAP={score:.4f} "
                      f"({elapsed:.1f}s){tag}")

            # Update best for this axis
            if round_best_score > best_score:
                best_params[key] = round_best_val
                best_score = round_best_score
                print(f"  >> {key} updated to {round_best_val} (mAP={best_score:.4f})")
            else:
                print(f"  >> {key} stays at {old_val}")

    # Restore full iteration count for the returned params
    if full_num_iterations is not None:
        best_params['num_iterations'] = full_num_iterations

    # Sort results by mAP descending
    all_search_results.sort(key=lambda x: -x['mAP'])

    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS (top 10)")
    print(f"{'='*60}")
    for entry in all_search_results[:10]:
        print(f"  {entry['param']:15s}={entry['value']:<10}  mAP={entry['mAP']:.4f}")

    print(f"\nBest params: {best_params}")
    print(f"Best mAP:    {best_score:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    search_save = {
        'best_params': best_params,
        'best_mAP': best_score,
        'all_results': all_search_results,
    }
    with open(os.path.join(output_dir, "hyperparam_search.json"), 'w') as f:
        json.dump(search_save, f, indent=2)
    print(f"Saved to {output_dir}/hyperparam_search.json")

    return search_save
