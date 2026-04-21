"""
MC-Dropout + bootstrap + permutation-test analysis for the brilliance classifier.

Produces the numbers used in Tables 3-5 and Figure 2 of the report:

  (a) MC-Dropout uncertainty on the 5-move test set:
      for each position, keep dropout *on* at inference time and do N
      stochastic forward passes through AggReduce. Report mean and
      standard deviation of the sigmoid "Brilliant" probability.

  (b) Bootstrap 95% confidence interval on 5-move accuracy, resampling
      moves with replacement.

  (c) Paired permutation test on Table 4 (Bug 3): does the post-fix
      scaler produce a different score than the pre-fix batch-fit
      scaler, beyond chance?

All three are standard, reviewer-defensible tools.
"""

from __future__ import annotations
import json
import os
import pickle
import random
import argparse
import numpy as np
import torch

from inference_from_trees import (
    NeuralNetworkDropout,
    parse_trees,
    apply_scaler,
)


# ---------------------------------------------------------------------------
# MC-Dropout: turn dropout on, sample N times, summarise.
# ---------------------------------------------------------------------------

def mc_dropout_predict(model, X, n_samples=500, device="cuda"):
    """Return (n_moves, n_samples) tensor of sigmoid 'Brilliant' probs."""
    model.train()  # enable dropout at inference; this is the key line
    for m in model.modules():
        # keep BN (if any) in eval mode; we only want dropout stochastic
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    probs = np.empty((X.shape[0], n_samples), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_samples):
            logits = model(X_t)
            # match the inference script: "Brilliant" = sigmoid(-logit)
            probs[:, i] = torch.sigmoid(-logits).squeeze(-1).cpu().numpy()

    model.eval()
    return probs


# ---------------------------------------------------------------------------
# Bootstrap CI on 5-move accuracy.
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(correct, n_boot=10000, alpha=0.05, seed=0):
    """95% CI on mean of a 0/1 vector via non-parametric bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    accs = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[i] = correct[idx].mean()
    lo = np.quantile(accs, alpha / 2)
    hi = np.quantile(accs, 1 - alpha / 2)
    return float(correct.mean()), float(lo), float(hi), accs


# ---------------------------------------------------------------------------
# Paired permutation test on Bug 3 pre/post scores.
# ---------------------------------------------------------------------------

def paired_permutation_test(before, after, n_perm=100000, seed=0):
    """
    Two-sided paired permutation test on the mean difference.
    H0: post-fix score has the same distribution as pre-fix score.
    """
    rng = np.random.default_rng(seed)
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    diff = after - before
    obs = diff.mean()

    # sample random sign flips of each paired difference
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        stat = (signs * diff).mean()
        if abs(stat) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(obs), float(p)


# ---------------------------------------------------------------------------
# Pretty print.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--moves_dir", default="moves")
    ap.add_argument("-s", "--state_dict",
                    default=os.path.join("models", "model-7936-2.pth"))
    ap.add_argument("--scaler",
                    default=os.path.join("models", "scaler.pkl"))
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default="mc_dropout_results.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # -- features -----------------------------------------------------------
    print("[1/3] parsing trees ...")
    X = parse_trees(moves_dir=args.moves_dir)
    X = apply_scaler(X, scaler_path=args.scaler)

    # map parse_trees' walk order to move names (same walk inference uses)
    dirs = [d for d, _, _ in os.walk(args.moves_dir)][1:]
    move_names = [os.path.basename(d) for d in dirs]

    # -- MC-Dropout forward passes -----------------------------------------
    print(f"[2/3] MC-Dropout: {args.n_samples} stochastic forward passes ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetworkDropout(25, 400, 50, dropout=0.2).to(device)
    model.load_state_dict(torch.load(args.state_dict, map_location=device))

    probs = mc_dropout_predict(model, X, n_samples=args.n_samples, device=device)
    mean = probs.mean(axis=1)
    std  = probs.std(axis=1, ddof=1)

    # expected labels for the 5 fixed moves in the report (1 = brilliant)
    expected_map = {
        "game_of_the_century": 1,
        "morphy_qb8_sac":       1,
        "opera_game_1e4":       0,
        "kasparov_rxd4":        1,
        "Vranesic_Stein":       0,
    }
    predictions = (mean > 0.5).astype(int)
    expected    = np.array([expected_map.get(n, -1) for n in move_names])
    correct     = (predictions == expected).astype(int)

    print()
    print(f"{'move':28s}  mean   std    95% CI         pred  exp  hit")
    print("-" * 74)
    per_move = []
    for name, m, s, p, e, c in zip(move_names, mean, std, predictions,
                                   expected, correct):
        ci_lo = max(0.0, m - 1.96 * s)
        ci_hi = min(1.0, m + 1.96 * s)
        per_move.append({"move": name, "mean": float(m), "std": float(s),
                         "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                         "prediction": int(p), "expected": int(e),
                         "correct": int(c)})
        print(f"{name:28s}  {m:.3f}  {s:.3f}  [{ci_lo:.3f},{ci_hi:.3f}]  "
              f"{p:>4d}  {e:>3d}  {c:>3d}")

    # -- bootstrap CI on accuracy ------------------------------------------
    print()
    print(f"[3/3] bootstrap 95% CI on accuracy ({args.n_boot} resamples) ...")
    acc, lo, hi, _ = bootstrap_accuracy_ci(correct, n_boot=args.n_boot)
    print(f"    accuracy = {acc:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]")

    # -- permutation test on Bug 3 -----------------------------------------
    # data copied from REPORT.tex Table 4 (Bug 3 before/after).
    before = [0.88, 0.93, 0.91]
    after  = [0.97, 0.97, 0.97]
    d_obs, p_perm = paired_permutation_test(before, after, n_perm=100000)
    print()
    print(f"Paired permutation test on Bug 3 (n={len(before)}):")
    print(f"    mean difference (after - before) = {d_obs:+.3f}")
    print(f"    two-sided p-value               = {p_perm:.5f}")

    # -- save ---------------------------------------------------------------
    summary = {
        "per_move": per_move,
        "accuracy": {"value": acc, "ci_95": [lo, hi], "n_boot": args.n_boot},
        "bug3_permutation": {
            "before": before, "after": after,
            "mean_diff": d_obs, "p_value": p_perm, "n_perm": 100000,
        },
        "n_mc_samples": args.n_samples,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
