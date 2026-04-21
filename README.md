# Brilliant-Moves Recreate

Bachelor's practical work (IML, JKU) reproducing *Predicting User
Perception of Move Brilliance in Chess* (Zaidi & Guerzhoy,
arXiv:2406.11895).

- **Author:** Joash Johnson Samuel, K12340310
- **Report:** [`REPORT.tex`](REPORT.tex) (compile with pdflatex; see
  also [`REPORT.md`](REPORT.md) for the Markdown source)
- **Everything the grader needs:** this README + the `REPORT.pdf` +
  the code in this repo

## TL;DR

```powershell
# (0) Clone vanilla lc0 at the pinned SHA and apply the one-line patch
git clone https://github.com/LeelaChessZero/lc0.git lc0-upstream
cd lc0-upstream
git checkout ee6866911663485d94c1e7ff99e607c15f2110be
git apply ..\patches\lc0_logging_h_chrono.patch
cd ..

# (1) Override toolchain paths if your CUDA / cuDNN live elsewhere (optional)
$env:CUDA_PATH  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CUDNN_ROOT = "C:\tools\cudnn-windows-x86_64-9.21.0.82_cuda12-archive"
$env:CC_CUDA    = "86"   # Ampere (30-series). Use 89 for Ada, 75 for Turing.

# (2) Build patched lc0 with CUDA + cuDNN + GML tree-dump patch
.\build_lc0.ps1

# (3) Sanity check with the author's demo trees (reproduces paper scores)
cd brilliant-moves-clf\brilliant_moves_clf
python inference_from_trees.py --moves_dir moves --scaler models\scaler.pkl

# Expected (two canonical demo moves from the paper):
#   game_of_the_century: 0.88, Brilliant.
#   Vranesic_Stein:      0.01, Not Brilliant.
```

## Requirements

* Windows 10/11 x64 (the build scripts and paths are Windows-specific;
  the Python code itself is OS-agnostic).
* NVIDIA GPU with CUDA 12.x toolkit installed.
* cuDNN 9.x (we used 9.21.0) extracted next to the CUDA installation.
* Visual Studio 2022 Build Tools (`vcvarsall.bat amd64`).
* Meson + Ninja on `PATH`.
* Python 3.12 with a venv at `.venv/`.  Deps: `torch` (CUDA wheel),
  `networkx`, `scikit-learn`, `python-chess`, `numpy`, `requests`.

## Layout at a glance

| Path | Purpose |
|---|---|
| `REPORT.tex` / `REPORT.md` | Full write-up (methodology, results, 3 bugs, training pipeline, MC-dropout analysis). |
| `build_lc0.ps1` / `build_lc0_incremental.ps1` | Build patched lc0 (expects `lc0-upstream/` next to this folder after step 0 in TL;DR). |
| `patches/lc0_logging_h_chrono.patch` | Our one-line `<chrono>` fix to upstream lc0 (Bug 1). |
| `brilliant-moves-clf/brilliant_moves_clf/` | Main codebase (forked from the authors). |
| `...inference_from_trees.py` | Feature extraction + classification (patched, Bug 3). |
| `...train_classifier.py` | Training pipeline (new, not in paper repo). |
| `...scrape_lichess.py` | Public Lichess Study scraper (new). |
| `...analysis_mc_dropout.py` | MC-Dropout + bootstrap + permutation-test analysis (new). |
| `...generate_trees.py` | lc0-based tree generation (patched, Bug 2). |
| `...lc0/src/mcts/search.cc` | Authors' GML tree-dump patch for lc0. |
| `...models/model-7936-2.pth` | Authors' pretrained AggReduce weights. |
| `...models/scaler.pkl` | Training-time StandardScaler (our Bug 3 fix). |
| `...moves/` | 5 labelled moves (with `label.txt`) usable for train/infer. |
| `...trees/` | 50 pregenerated `.gml` trees for those moves. |
| `...mc_dropout_results.json` | Raw output of the MC-dropout analysis. |
| `email-to-author-DRAFT.md` | Draft data request to the paper authors. |

## Common runs

All `python` commands assume the venv at `..\..\.venv\Scripts\python.exe`
(or whichever Python you have installed with torch+cuda+networkx+
scikit-learn).  Run them from `brilliant-moves-clf\brilliant_moves_clf\`
because `parse_trees` uses relative paths.

### A. Inference (paper reproduction)

```
python inference_from_trees.py --moves_dir moves --scaler models\scaler.pkl
```

### B. Regenerate trees for the labelled moves (slow, requires built lc0)

```
python generate_trees.py
```

This will consume `moves/*/fen.txt`, invoke the patched `lc0.exe`, and
rewrite every `trees/{lc0,maia}/*/tree_*_*.gml`.  Budget ≈15 s per
(weight × depth), so ≈60 s per move on an RTX 3060 Ti.

### C. Parse a custom PGN

```
python pgn_parser.py ..\..\my_game.pgn
```

### D. Train the classifier on the labelled moves

```
python train_classifier.py --moves_dir moves --models_dir models --epochs 60 --batch_size 4
```

Produces `models/model_<ts>.pth`, `models/scaler_<ts>.pkl`, and
`models/train_log_<ts>.json`.  Also overwrites `models/scaler.pkl` so
that subsequent inference runs are batch-invariant.

### E. Scrape Lichess studies for a larger training run

```
python scrape_lichess.py --ids_file studies.txt --pgn_out raw_pgns --merge
python pgn_parser.py raw_pgns\merged.pgn
# hand-label moves/*/label.txt
python generate_trees.py
python train_classifier.py
```

### F. MC-Dropout uncertainty + bootstrap CI + Bug 3 permutation test

```
python analysis_mc_dropout.py --n_samples 500 --n_boot 10000
```

Runs 500 stochastic forward passes per move with dropout enabled at
inference (Gal & Ghahramani 2016), a 10 000-resample bootstrap CI on
accuracy, and a 100 000-permutation paired test on the Bug 3 pre/post
scores. Output is written to `mc_dropout_results.json` and feeds
Tables 3 and 4 of the report.

## Bugs fixed (summary, see `REPORT.md` §5 for details)

| # | Where | Symptom | Fix |
|---|---|---|---|
| 1 | `lc0-upstream/src/utils/logging.h` | MSVC 14.44: `'system_clock' is not a member of 'std::chrono'` | add `#include <chrono>` |
| 2 | `brilliant_moves_clf/generate_trees.py` | deadlock on 10⁴/10⁵-node searches | `stdout=DEVNULL, stderr=DEVNULL`, timeout 30→600 s |
| 3 | `brilliant_moves_clf/inference_from_trees.py` | confidence scores depend on batch composition | split parse/scale; load train-time scaler from `models/scaler.pkl` |

## Known limitations

* Full numerical reproduction of the paper's 78 % accuracy / 0.83 AUC
  requires the authors' 10 120-sample training set, which was not
  released.  We implement the pipeline end-to-end and demonstrate it
  runs; scaling to paper-size is purely a data-and-compute issue
  (≈1 week of GPU time for tree generation on our hardware).
* Paths in `parse_trees()` are hardcoded relative to CWD; run from
  `brilliant-moves-clf\brilliant_moves_clf\`.

## Packaging a submission

```powershell
.\make_submission.ps1
```

Produces `submission.zip` (~28 MB) by staging a clean copy without `.venv`,
`lc0-upstream/build/`, downloaded network weights, or per-run training
artefacts. The pretrained `model-7936-2.pth` and the training-fit
`scaler.pkl` are kept. To include the compiled report, build
`REPORT.tex` to `REPORT.pdf` (e.g. via Overleaf) *before* running the
script.

## Reproducibility pins

| Component | Pin |
|---|---|
| Upstream `lc0` | commit `ee68669116` (apply `patches/lc0_logging_h_chrono.patch` on top) |
| Authors' `brilliant-moves-clf` | commit `b992e4ca61` (shipped as a subfolder here) |
| Leela T82 weights | `768x15x24h-t82-swa-7464000.pb.gz` (fetch separately, see REPORT) |
| Maia weights | ratings 1100 / 1500 / 1900 from `CSSLab/maia-chess` |
| Python | 3.12, PyTorch 2.6.0 + cu124 |
| Toolchain | CUDA 12.x, cuDNN 9.21, MSVC 2022 |

## License & credits

* Code in `brilliant-moves-clf/` originates with Zaidi & Guerzhoy (2024);
  only the files noted in `REPORT.md` / `REPORT.tex` have been modified.
* The lc0 patch (`patches/lc0_logging_h_chrono.patch`) is a one-line
  `#include <chrono>` fix; apply it on top of vanilla Leela Chess Zero
  at the pinned SHA.
* Additions (`train_classifier.py`, `scrape_lichess.py`,
  `analysis_mc_dropout.py`, `build_lc0.ps1`, `REPORT.tex`, `REPORT.md`,
  this `README.md`) are original work for this practical.
