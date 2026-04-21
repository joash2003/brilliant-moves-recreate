"""
train_classifier.py -- training script for the AggReduce brilliance classifier.

This script was NOT released with the paper (Zaidi, Guerzhoy, 2024).
It reproduces the training loop described in the paper:
  * AggReduce neural network architecture (from inference_from_trees.py)
  * Binary cross-entropy with sigmoid output (label 0 = brilliant, 1 = not)
  * Adam optimizer
  * Random 80/10/10 train/val/test split
  * Early stopping on validation loss
  * Persists a StandardScaler fit on the training set only (fixes Bug 3).

Input expected:
    --moves_dir  : directory containing labelled move folders, each with a
                   `fen.txt`, `uci.txt` and `label.txt` file. `label.txt`
                   must contain a single token: "brilliant" or "not_brilliant".
    --trees_dir  : directory containing the .gml trees produced by
                   generate_trees.py (expected layout: trees/<weight>/<move>/).

Output artifacts (written to `models/`):
    model_<run-id>.pth  -- PyTorch state dict
    scaler_<run-id>.pkl -- pickled sklearn StandardScaler
    train_log_<run-id>.json -- metrics per epoch + final test metrics

Usage:
    python train_classifier.py --moves_dir moves --trees_dir trees \
        --epochs 200 --lr 1e-3 --batch_size 32

See REPORT.md for a full discussion of why we had to write this from scratch.
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from inference_from_trees import NeuralNetworkDropout, parse_trees


LABEL_MAP = {
    "brilliant": 0,      # paper convention: 0 = brilliant
    "not_brilliant": 1,  # paper convention: 1 = not brilliant
}


def load_labels(moves_dir):
    """Reads label.txt in every move folder and returns {move_name: int_label}."""
    labels = {}
    for entry in sorted(os.listdir(moves_dir)):
        path = os.path.join(moves_dir, entry)
        if not os.path.isdir(path):
            continue
        label_file = os.path.join(path, "label.txt")
        if not os.path.exists(label_file):
            raise FileNotFoundError(
                f"Missing label.txt for {entry}; each training sample must be labelled."
            )
        with open(label_file, "r") as f:
            token = f.read().strip().lower()
        if token not in LABEL_MAP:
            raise ValueError(
                f"{label_file}: expected 'brilliant' or 'not_brilliant', got {token!r}"
            )
        labels[entry] = LABEL_MAP[token]
    return labels


def stratified_split(names, labels, rng, frac=(0.8, 0.1, 0.1)):
    """Splits move names into train/val/test preserving the class balance."""
    by_class = {0: [], 1: []}
    for n in names:
        by_class[labels[n]].append(n)
    train, val, test = [], [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_total = len(items)
        n_train = int(round(n_total * frac[0]))
        n_val = int(round(n_total * frac[1]))
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def select_rows(X, all_names, subset_names):
    """Returns the rows of X corresponding to subset_names, in that order."""
    index = {name: i for i, name in enumerate(all_names)}
    rows = [index[n] for n in subset_names]
    return X[rows]


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    losses, all_logits, all_y = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item() * xb.size(0))
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())
    n = sum(len(y) for y in all_y)
    avg_loss = sum(losses) / max(n, 1)
    logits = np.concatenate(all_logits) if all_logits else np.array([])
    y = np.concatenate(all_y) if all_y else np.array([])
    return avg_loss, logits, y


def compute_metrics(logits, y):
    if len(y) == 0:
        return {"acc": float("nan"), "f1": float("nan"), "auc": float("nan")}
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = float("nan")  # single class in y (tiny demo set)
    return {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the AggReduce brilliance classifier.")
    parser.add_argument("--moves_dir", default="moves", help="Labelled move folders with fen/uci/label.txt.")
    parser.add_argument("--trees_dir", default="trees", help="Pre-generated .gml trees (unused; inference_from_trees.parse_trees uses a fixed 'trees' layout).")
    parser.add_argument("--models_dir", default="models", help="Where to write the model and scaler.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--h1", type=int, default=25)
    parser.add_argument("--h2", type=int, default=400)
    parser.add_argument("--h3", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (val loss).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # -- 1. Load features and labels -----------------------------------------
    print("[train] parsing trees and building feature matrix")
    X = parse_trees(moves_dir=args.moves_dir)  # (N, 3980)
    names = [os.path.basename(d[0]) for d in os.walk(args.moves_dir)][1:]
    labels = load_labels(args.moves_dir)

    if len(names) != X.shape[0]:
        raise RuntimeError(
            f"Mismatch: parse_trees returned {X.shape[0]} rows but found {len(names)} move folders."
        )
    n_brilliant = sum(1 for n in names if labels[n] == 0)
    print(f"[train] loaded {len(names)} samples ({n_brilliant} brilliant, {len(names)-n_brilliant} not).")

    # -- 2. Stratified split --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    import random
    py_rng = random.Random(args.seed)
    # Use python's random for in-place shuffle of lists (numpy's default_rng doesn't shuffle lists)
    def _split():
        by_class = {0: [], 1: []}
        for n in names:
            by_class[labels[n]].append(n)
        train, val, test = [], [], []
        for cls, items in by_class.items():
            py_rng.shuffle(items)
            n_total = len(items)
            n_train = max(int(round(n_total * 0.8)), 1 if n_total > 0 else 0)
            n_val = max(int(round(n_total * 0.1)), 1 if n_total >= 3 else 0)
            train.extend(items[:n_train])
            val.extend(items[n_train : n_train + n_val])
            test.extend(items[n_train + n_val :])
        py_rng.shuffle(train)
        py_rng.shuffle(val)
        py_rng.shuffle(test)
        return train, val, test

    train_names, val_names, test_names = _split()
    print(f"[train] split: train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")

    # -- 3. Fit scaler on TRAIN only (Bug 3 fix) ------------------------------
    X_train = select_rows(X, names, train_names)
    X_val = select_rows(X, names, val_names) if val_names else np.zeros((0, X.shape[1]))
    X_test = select_rows(X, names, test_names) if test_names else np.zeros((0, X.shape[1]))

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    if X_val.size:
        X_val = scaler.transform(X_val)
    if X_test.size:
        X_test = scaler.transform(X_test)

    scaler_path = os.path.join(args.models_dir, f"scaler_{run_id}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    # Also write canonical name so inference picks it up by default
    with open(os.path.join(args.models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"[train] wrote scaler to {scaler_path} (and models/scaler.pkl)")

    # -- 4. Build tensors + loaders ------------------------------------------
    y_train = np.array([labels[n] for n in train_names], dtype=np.float32)
    y_val = np.array([labels[n] for n in val_names], dtype=np.float32)
    y_test = np.array([labels[n] for n in test_names], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    torch.manual_seed(args.seed)

    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if len(test_ds) > 0 else None

    # -- 5. Model + optimiser -------------------------------------------------
    model = NeuralNetworkDropout(args.h1, args.h2, args.h3, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []

    print(f"[train] starting training for up to {args.epochs} epochs")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_logits, tr_y = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        if val_loader is not None:
            vl_loss, vl_logits, vl_y = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        else:
            vl_loss, vl_logits, vl_y = float("nan"), np.array([]), np.array([])
        tr_m = compute_metrics(tr_logits, tr_y)
        vl_m = compute_metrics(vl_logits, vl_y)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss, "train": tr_m, "val": vl_m})
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:4d} | train loss={tr_loss:.4f} acc={tr_m['acc']:.3f}"
                f" | val loss={vl_loss:.4f} acc={vl_m['acc']:.3f} auc={vl_m['auc']:.3f}"
            )
        # Early stopping on val loss
        if val_loader is not None and vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if val_loader is not None and patience_counter >= args.patience:
                print(f"[train] early stopping at epoch {epoch} (best val loss @ epoch {best_epoch}: {best_val_loss:.4f})")
                break
    elapsed = time.time() - t0
    print(f"[train] finished in {elapsed:.1f}s (best epoch={best_epoch})")

    # -- 6. Evaluate on test with best weights --------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if test_loader is not None:
        ts_loss, ts_logits, ts_y = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        ts_m = compute_metrics(ts_logits, ts_y)
        print(f"[train] TEST: loss={ts_loss:.4f} acc={ts_m['acc']:.3f} f1={ts_m['f1']:.3f} auc={ts_m['auc']:.3f}")
    else:
        ts_loss, ts_m = float("nan"), {"acc": float("nan"), "f1": float("nan"), "auc": float("nan")}

    # -- 7. Save artefacts ----------------------------------------------------
    model_path = os.path.join(args.models_dir, f"model_{run_id}.pth")
    torch.save(model.state_dict(), model_path)
    log_path = os.path.join(args.models_dir, f"train_log_{run_id}.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "args": vars(args),
                "n_train": len(train_names),
                "n_val": len(val_names),
                "n_test": len(test_names),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test": {"loss": ts_loss, **ts_m},
                "history": history,
                "train_names": train_names,
                "val_names": val_names,
                "test_names": test_names,
            },
            f,
            indent=2,
        )
    print(f"[train] wrote model   -> {model_path}")
    print(f"[train] wrote scaler  -> {scaler_path}")
    print(f"[train] wrote log     -> {log_path}")


if __name__ == "__main__":
    main()
