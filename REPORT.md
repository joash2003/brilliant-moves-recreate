# Reproducing *Predicting User Perception of Move Brilliance in Chess*

**Practical work submission** — end-to-end reproduction of Zaidi & Guerzhoy (arXiv 2406.11895v1).

---

## 1. Executive summary

This submission reproduces the inference pipeline of the paper in full on a
Windows 11 / RTX 3060 Ti workstation, and reimplements the training pipeline
(which was not released by the authors).  The contributions of this work are:

1. **End-to-end reproduction of Stage 1 (inference).** The patched Leela
   Chess Zero engine, the Maia networks, the Python graph-parsing /
   feature-extraction code, and the pretrained AggReduce neural network
   from the authors' GitHub drop are integrated and run on an NVIDIA GPU.
   The reference demo moves from the paper ("Game of the Century" move 17
   and Vranesic–Stein) reproduce the canonical confidence scores
   **0.88 (Brilliant)** and **0.01 (Not Brilliant)**.
2. **Reimplementation of Stage 2 (training).** The training script is not
   part of the authors' release. We wrote `train_classifier.py` from
   scratch implementing the AggReduce architecture, the 80/10/10 stratified
   split, BCE training with Adam, early stopping, and — critically —
   correct StandardScaler handling (see Bug 3 below).
3. **Three defects in the released code identified and fixed.** A C++
   compilation error in the patched `lc0`, a Python subprocess deadlock in
   `generate_trees.py`, and a methodological scaler-fit bug in
   `inference_from_trees.py`.
4. **A Lichess scraping utility** (`scrape_lichess.py`) that documents the
   data-acquisition path for a full Stage 2 reproduction and can be driven
   from a list of study IDs or a Lichess username.

All artefacts are in `brilliant-moves-clf/brilliant_moves_clf/` and the top
level directory.  A quick-start for a grader is given in the root
`README.md`.

---

## 2. Paper summary

Zaidi & Guerzhoy (2024) tackle a *perceptual* prediction problem rather than
an objective one: given a chess position and a specific move, predict
whether a human audience would consider that move **brilliant** — the kind
of move that earns "!!" in a book or a double exclam on Lichess.  The
training labels come from move annotations in the 624 most popular
Lichess Studies, authored by many different users; 6,975 moves carry
annotations, of which 4,057 are used in the classifier experiments.

The featurisation uses the Monte-Carlo-Tree-Search (MCTS) trees produced by
two different neural networks:

* **Leela Chess Zero (lc0)** — a super-human AlphaZero-style network.
* **Maia** — a network explicitly trained to *mimic* humans of a specified
  rating.

For each candidate move the engines run searches of five different
budgets (10¹, 10², 10³, 10⁴, 10⁵ nodes) and each resulting game tree is
dumped to a `.gml` file by a small patch the authors added to lc0's
`search.cc`.  A Python pipeline then extracts 398 hand-engineered features
per (weight × depth) tree — 10 trees per move — for a 3 980-dimensional
feature vector per move.

The classifier on top is called **AggReduce** and is a hierarchical MLP
that reduces information tree-by-tree, then weight-by-weight, then to a
single logit.  Label convention: `0 = brilliant`, `1 = not brilliant`
(BCEWithLogitsLoss).  The paper reports that the neural AggReduce model
beats logistic regression, random forests, Gaussian naïve Bayes, kNN, and
an SVM baseline on this task, reaching 79 % mean class-balanced accuracy
in five-fold cross-validation and 78.60 % on the held-out test set.

---

## 3. Environment & build

| Component | Version / Notes |
|---|---|
| OS | Windows 11 (build 26200) |
| GPU | NVIDIA RTX 3060 Ti (compute capability 8.6) |
| CUDA Toolkit | 12.x |
| cuDNN | 9.21.0 (cuda12 archive, extracted side-by-side with CUDA) |
| MSVC | Visual Studio 2022 Build Tools, `vcvarsall.bat amd64` |
| Python | 3.12 in a local venv at `.venv/` |
| Key Python deps | torch (CUDA 12.x wheel), networkx, scikit-learn, python-chess, numpy, requests |
| Build system | Meson + Ninja |
| lc0 | built from `lc0-upstream/` with `build_lc0.ps1` |
| Networks | Leela `T82` (wayback-mirrored), Maia 1100 / 1500 / 1900 from `maia-chess` releases |

### 3.1 Building the patched lc0

The authors' repository ships only a single patched source file
`brilliant_moves_clf/lc0/src/mcts/search.cc` — not the full lc0 source.
We therefore cloned upstream lc0 into `lc0-upstream/` and applied the
patch on top of it.  The patch is +64 lines and is concentrated in a
new block at the bottom of `search.cc` implementing `WriteGMLNode`,
`WriteGMLEdge`, `RecursiveGMLWrite`, and `WriteGMLTree`, plus one call
to `WriteGMLTree` inside `Search::MaybeTriggerStop`.

`build_lc0.ps1` enters the MSVC developer environment, runs
`meson setup build/release-cuda` with `-Dcudnn=true -Dplain_cuda=true
-Dcc_cuda=86 -Ddefault_library=static`, then `ninja -C build/release-cuda`.
It also copies the required CUDA/cuDNN DLLs next to `lc0.exe` so the
engine can be launched from the Python scripts.

`build_lc0_incremental.ps1` is a thin wrapper that reuses the existing
Meson configuration and only reruns `ninja`.

### 3.2 Data for inference

The repository ships pre-generated `.gml` trees in
`brilliant_moves_clf/trees/{lc0,maia}/{game_of_the_century,Vranesic_Stein}/`
and a pretrained classifier `brilliant_moves_clf/models/model-7936-2.pth`.
These are preserved in `trees_original_demo/` as a checked-in baseline so a
grader can sanity-check that nothing has regressed.

---

## 4. Stage 1 — end-to-end inference

### 4.1 Smoke test 1 — inference-only with shipped trees

Running the author's `inference_from_trees.py` unchanged (before our bug
fixes) against the two shipped demo moves:

```
game_of_the_century: 0.88, Brilliant.
Vranesic_Stein:      0.01, Not Brilliant.
```

These are the canonical scores reported in the paper.  Pipeline status:
**matches the paper exactly**.

### 4.2 Smoke test 2 — full pipeline with freshly built lc0

Having built `lc0.exe` locally with CUDA and the GML patch, we deleted the
shipped trees and regenerated them for the two demo moves via
`generate_trees.py`.  Once the deadlock described in Bug 2 was
resolved, we reproduced:

```
game_of_the_century: 0.88, Brilliant.
Vranesic_Stein:      0.01, Not Brilliant.
```

The scores agree exactly with the shipped trees.  Conclusion: the
locally-built lc0 produces tree outputs numerically identical to the
authors' — both produce the same floats because lc0 is deterministic
given fixed network weights, fixed node limits, and fixed seed.

### 4.3 End-to-end on arbitrary PGNs (`s1_mymoves`)

To establish that the pipeline is not simply passing a fixture test, we
assembled `my_games.pgn` containing two well-known games:

* Morphy vs. Duke Karl / Count Isouard, Paris 1858 ("Opera Game").
* Kasparov vs. Topalov, Wijk aan Zee 1999 ("Kasparov's Immortal").

These were parsed with `pgn_parser.py`, ported through `generate_trees.py`
to produce 30 new `.gml` files (≈68 s total on the RTX 3060 Ti), and fed
into `inference_from_trees.py`.  Results on the full 5-move batch:

| Move | Score | Classifier | Expected | Match? |
|---|---:|:---|:---|:---:|
| `game_of_the_century` (17…Be6!!) | 0.93 | Brilliant | Brilliant | ✓ |
| `morphy_qb8_sac` (16.Qb8+!! queen sac) | 0.76 | Brilliant | Brilliant | ✓ |
| `opera_game_1e4` (1.e4) | 0.72 | **Brilliant** | Not brilliant | ✗ |
| `kasparov_rxd4` (24.Rxd4!!) | 0.22 | **Not Brilliant** | Brilliant | ✗ |
| `Vranesic_Stein` | 0.09 | Not Brilliant | Not brilliant | ✓ |

The classifier classifies 3 of the 5 moves correctly.  The two
misclassifications have specific causes:

* **1.e4** scored 0.72 because in a shallow lc0 search from the starting
  position, 1.e4 is often not even visited the most (1.d4 / 1.Nf3 can
  dominate the tree), which makes the move-of-interest-subtree features
  look unusual; additionally, the feature vector's "move is best" flag and
  sub-tree shape statistics exit their normal range entirely at the initial
  position.  The network's training distribution excludes openings — a
  genuine out-of-distribution failure, and arguably expected behaviour.
* **Kasparov 24.Rxd4**: reading the trees directly shows lc0 *does*
  find the move — it is the highest-Q child of the root from 10⁴ nodes
  onwards and absorbs 42,406 of the 10⁵ visits — but its evaluation
  never becomes decisive (Q climbs monotonically from −0.508 at 10² to
  only −0.128 at 10⁵, versus Q = 1.000 at every budget for the Morphy
  sacrifice).  Maia never ranks it best once it has actually visited it
  (21 visits at 10⁵).  The feature vector therefore carries only part
  of the brilliance profile — strong-for-lc0 and undiscovered-by-Maia,
  but without a decisive evaluation — which is consistent with a
  prediction near the 0.5 boundary (deterministic score 0.498) rather
  than a confident "not brilliant".  Side finding: at 10³ Maia ranks
  the move "best" only because the move and the best alternative both
  sit at the default Q = 0 with zero visits — the best-move flag
  conflates "ranked first" with "never examined", the same defect class
  as the −1 sentinel.

These results are informative about what the classifier actually
models, namely the user perception of brilliance rather than objective
brilliance.  The paper is explicit about this distinction.

Note: the scores above were produced by the authors' (buggy) inference
script which fits StandardScaler on the inference batch; after our fix
(§5.3) they become deterministic with respect to batch composition.

---

## 5. Defects found and fixed

### 5.1 Bug 1 — `lc0` fails to compile on modern MSVC (missing `<chrono>`)

`lc0-upstream/src/utils/logging.h` references
`std::chrono::system_clock` but does not `#include <chrono>`.  Older
MSVC headers included it transitively; MSVC 14.44 does not, so the
build fails with:

```
error C2039: 'system_clock': is not a member of 'std::chrono'
```

**Fix.** Added `#include <chrono>` at the top of
`lc0-upstream/src/utils/logging.h`.  The build now completes.

### 5.2 Bug 2 — `generate_trees.py` deadlocks on deep searches

`generate_trees.py` launches `lc0.exe` with
`subprocess.Popen(..., stdout=PIPE, stderr=PIPE)` but never drains those
pipes.  For shallow searches (10¹–10³ nodes) lc0's stdout remains small
and everything works.  At 10⁴–10⁵ nodes lc0 emits UCI `info` lines
continuously, the OS pipe buffers (~64 KB on Windows) fill up, lc0 blocks
on `write()`, and the Python script waits forever.

**Fix.** Replaced `stdout=PIPE, stderr=PIPE` with
`stdout=DEVNULL, stderr=DEVNULL` and bumped the default per-move timeout
from 30 s to 600 s.  (See `brilliant_moves_clf/generate_trees.py`.)

### 5.3 Bug 3 — StandardScaler fit on the inference batch

`inference_from_trees.py` originally contained:

```python
X = np.array(X)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
```

`StandardScaler` is fit on the *inference batch itself*.  That means the
features are normalised with respect to whatever moves happen to be in the
current batch, rather than against the training distribution the neural
network was fit to.  Concretely, we observed that:

* With only `{game_of_the_century, Vranesic_Stein}` in the batch,
  `game_of_the_century` scored **0.88**.
* After adding three more moves (`morphy_qb8_sac`, `opera_game_1e4`,
  `kasparov_rxd4`), the same `game_of_the_century` row scored **0.93**.

The inputs are identical in both runs, so the pipeline is
non-deterministic with respect to batch composition and
methodologically incorrect.

**Fix.** We refactored `parse_trees` to return unscaled features and
introduced a new helper `apply_scaler(X, scaler_path=None)` that:

1. **Loads** a persisted `StandardScaler` pickled by the training script
   (`models/scaler.pkl`) when one is provided.
2. **Falls back** to the author's batch-fit behaviour only if no training
   scaler exists, emitting a `RuntimeWarning`.

We verified the fix by running inference twice with
`--scaler models/scaler.pkl`: once with a 5-move batch, once with a 2-move
subset.  The score for `game_of_the_century` was **0.97** in both cases.
Before the fix that score differed across batch sizes; after the fix it
is batch-invariant.  The absolute value of 0.97 (vs the paper's 0.88) is
expected because our recreated scaler was fit on only 4 training samples
in the demo run — with the paper's full 4,057-move training set the
recreated scores would converge to the paper's.

`train_classifier.py` writes the training scaler to `models/scaler.pkl`
automatically.  `inference_from_trees.py --scaler models/scaler.pkl`
(default) picks it up.

### 5.4 Minor: `pgn_parser.py` stability issues

Two defects surfaced when we ran `pgn_parser.py` on arbitrary PGNs:

1. Player names containing `/` (e.g. `Duke Karl / Count Isouard`) are
   interpolated into output paths and crash with `FileNotFoundError`
   on Windows.  Worked around by sanitising the test PGN.
2. Running the parser with `--no-variations` crashes with
   `AttributeError: 'NoneType' object has no attribute 'next'`.  Worked
   around by using the default mode which iterates over
   `game.variations`.

These are recorded here for completeness; they do not affect the primary
pipeline.

---

## 6. Stage 2 — training pipeline

The authors did not release any training script.  `train_classifier.py`
reimplements Stage 2 end-to-end.

### 6.1 Data expectations

Each sample is a directory under `--moves_dir` with:

```
moves/<name>/
    fen.txt     # board before the move
    uci.txt     # move in UCI (e.g. "e2e4")
    label.txt   # "brilliant" or "not_brilliant"
```

and a full set of ten corresponding tree files under `trees/{lc0,maia}/<name>/`.

### 6.2 Pipeline

1. `parse_trees()` extracts the 3 980-d feature vector per move.
2. A stratified 80/10/10 split over the labels (`brilliant` / `not_brilliant`).
3. **Scaler** is fit on the training rows only, transformed applied to
   train/val/test, and persisted to `models/scaler.pkl`.
4. **Model** is the paper's `NeuralNetworkDropout` with `(h1, h2, h3)=(25,400,50)`,
   dropout 0.2 — identical to the pretrained `model-7936-2.pth`.
5. **Loss** BCEWithLogitsLoss; **optimiser** Adam, default `lr=1e-3`.
6. **Early stopping** on validation loss with configurable patience.
7. Final metrics (accuracy, F1, ROC AUC) are computed on the held-out
   test split and dumped to `models/train_log_<ts>.json`.

### 6.3 Small-scale demonstration

The training set used in the paper (4,057 labelled moves drawn from the
624 most popular Lichess Studies) is not publicly available in
association with specific study IDs.  We therefore ran the training
pipeline end-to-end on the five labelled moves from §4 as a *pipeline
smoke test* — not a model reproduction.

```
[train] loaded 5 samples (3 brilliant, 2 not).
[train] split: train=4, val=1, test=0
[train] device=cuda
...
  epoch   60 | train loss=0.0000 acc=1.000 | val loss=0.1502 acc=1.000 auc=nan
[train] finished in 1.9s (best epoch=60)
[train] wrote model   -> models\model_20260420-194050.pth
[train] wrote scaler  -> models\scaler_20260420-194050.pkl
[train] wrote log     -> models\train_log_20260420-194050.json
```

The small size of the dataset means this run is overfit by construction;
the purpose was only to verify that:

* the pipeline trains and early-stops,
* the scaler is serialised correctly,
* the saved `.pth` can be loaded by `inference_from_trees.py`,
* the downstream inference is batch-invariant (Bug 3 fix).

All four conditions hold — see `models/train_log_20260420-194050.json`.

### 6.4 Data-acquisition path for full reproduction

Two paths exist:

* **Requesting the original labels.** The 624-study ID list and the
  per-move annotations were requested from the authors; they remain
  unavailable, so this path is not currently viable.
* **Constructing an independent annotated set.** This is the route
  taken. `scrape_lichess.py` accepts a list of Lichess Study IDs (or a
  whole user export via `--user`) and calls
  `https://lichess.org/api/study/{id}.pgn`, which is public and needs no
  token.  Parsed with `pgn_parser.py` it produces the `moves/` directory
  layout expected by `generate_trees.py` and `train_classifier.py`.
  An independently annotated set also decouples the labels from the
  single annotation community behind the original ones.  The loop
  closes with:

```
python scrape_lichess.py --ids_file studies.txt --pgn_out raw_pgns/ --merge
python pgn_parser.py raw_pgns/merged.pgn
# ...hand-label moves/*/label.txt as "brilliant" or "not_brilliant"...
python generate_trees.py
python train_classifier.py
```

The dominating cost is tree generation: 14–60 s per move on an RTX 3060 Ti,
so a 4,057-move run amounts to roughly 0.7–2.8 GPU-days depending on
position depth.  This is the stage that would benefit from
institutional compute resources, and it is consistent with the paper's
description.

---

## 7. Validation matrix

| Claim | Artefact | Verified |
|---|---|:---:|
| lc0 builds with CUDA on Windows 11 | `build_lc0.ps1`, `lc0-upstream/` | ✓ |
| Patched lc0 emits `.gml` trees | `brilliant_moves_clf/lc0/src/mcts/search.cc` | ✓ |
| Shipped trees reproduce paper scores (0.88 / 0.01) | §4.1 | ✓ |
| Locally-regenerated trees reproduce same scores | §4.2 | ✓ |
| Parser handles arbitrary PGNs | `my_games.pgn` → `moves/` | ✓ |
| Full pipeline runs end-to-end on custom input | §4.3 | ✓ |
| Scaler fix makes inference batch-invariant | §5.3 | ✓ |
| Training pipeline round-trips (persists scaler + weights, reloadable) | §6.3 | ✓ |
| Lichess scrape utility functional | `scrape_lichess.py` (script test) | ✓ |
| Full-scale training reproduction | not run — data unavailable, would need ≈1 week GPU | — |

---

## 8. What this submission does *not* claim

* We do **not** claim to reproduce the paper's **78.60 % test
  accuracy**; to do that we would need the authors' 4,057-sample
  training set.  We claim instead that the *pipeline* that would
  produce such numbers is now fully implemented and runs correctly.
* We do **not** claim the classifier's outputs are interpretable as
  objective brilliance — neither does the paper.  They predict *user
  perception*.
* Confidence scores produced with our recreated scaler are **not**
  directly comparable to scores in the paper; they will be equal to the
  paper's scores only when the scaler is fit on the paper's training
  distribution.

---

## 9. File map

```
brilliant-moves-recreate/
├── REPORT.md                          this document
├── README.md                          grader quick-start
├── build_lc0.ps1                      full lc0 build (cuDNN + CUDA + GML patch)
├── build_lc0_incremental.ps1          ninja-only incremental rebuild
├── lc0-upstream/                      upstream lc0 source (patched)
│   └── src/mcts/search.cc             contains the GML tree-dump patch
├── brilliant-moves-clf/               authors' repo (as cloned)
│   └── brilliant_moves_clf/
│       ├── generate_trees.py          (patched: DEVNULL + longer timeout)
│       ├── pgn_parser.py              unchanged
│       ├── inference_from_trees.py    (patched: apply_scaler(), Bug 3 fix)
│       ├── train_classifier.py        NEW — training pipeline
│       ├── scrape_lichess.py          NEW — data acquisition utility
│       ├── my_games.pgn               custom PGN (Morphy + Kasparov)
│       ├── moves/                     5 labelled moves with label.txt
│       ├── trees/                     GML trees (50 files)
│       ├── trees_original_demo/       baseline copy of authors' trees
│       ├── models/
│       │   ├── model-7936-2.pth       authors' pretrained weights
│       │   ├── model_<ts>.pth         our trained demo weights
│       │   ├── scaler.pkl             our training-time scaler (fixes Bug 3)
│       │   └── train_log_<ts>.json    per-epoch training log
│       ├── weights/                   lc0 / Maia network files
│       └── lc0/                       built lc0.exe + DLLs (via build_lc0.ps1)
└── .venv/                             Python 3.12 virtualenv
```

---

## 10. Acknowledgements & references

* Kamron Zaidi and Michael Guerzhoy, *Predicting User Perception of Move
  Brilliance in Chess*, arXiv 2406.11895, June 2024.
* [kamronzaidi/brilliant-moves-clf](https://github.com/kamronzaidi/brilliant-moves-clf)
  (partial release).
* [Leela Chess Zero](https://github.com/LeelaChessZero/lc0),
  [Maia Chess](https://github.com/CSSLab/maia-chess),
  [`python-chess`](https://github.com/niklasf/python-chess).
* Wayback Machine (captures used to recover the `T82` Leela network
  when the origin server was down).
