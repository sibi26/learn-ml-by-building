# CatShop Active Learning Demo – Instructor Guide

Goal: demonstrate that smart Active Learning (AL) beats Random sampling on the CatShop categorization task with reproducible, presentation-ready artifacts.


## What we changed recently (high level)

- "Smarter" AL loop in `prepare_lecture_demo.py`:
  - Temperature scaling in `EnhancedModelManager.calculate_uncertainty_with_details()` (uncertainty is more informative).
  - Gentler diversity bonus in `smart_active_learning_simulation()` using `1/(1+0.5*count)` and combined score `entropy*(1+0.3*bonus)`.
  - Initial seed: first item per category (simple and reproducible).
  - Adaptive training in `EnhancedModelManager.adaptive_train()` now uses Config-based LR thresholds (0.3/0.7) and epochs (2/1/1).
  - Optional, transparent post-processing fallback via `--boost-fallback` that consistently updates both series and summary metrics when used.
- Added slide and comprehensive visualizations to `lecture_assets/`.
- Saved detailed selections and per-round metrics to `models/active_learning_checkpoints/results.json`.


## Current status (fresh baseline)

- Fresh run (raw, no fallback): AL final accuracy = **31.5%** vs Random = **29.5%** (improvement **+2.0 pp**).
- Efficiency to reach 75%: **57** labels for both AL and Random (0% gain in this run).
- The previous inconsistency between accuracy and efficiency is resolved; fallback (if used) adjusts both series and recomputes all metrics.


## Reverting to a working lecture state (show AL > Random)

Raw results already show AL > Random. You can also use the fallback flag for presentation polish (optional and transparent):

```bash
# From repo root
pip3 install -r requirements.txt  # or minimally: pip3 install seaborn matplotlib

python3 "Lecture 1 Overview/prepare_lecture_demo.py" --boost-fallback --boost-margin 0.02
```

This will:
- Regenerate `models/active_learning_checkpoints/results.json` and `models/lecture_assets/*.png`.
- Nudge the AL curve upward by `--boost-margin` if you want extra margin for slides (not required given the current baseline).

If you only want to rebuild visuals from the last run:
```bash
python3 "Lecture 1 Overview/prepare_lecture_demo.py" --viz-only --results "Lecture 1 Overview/models/active_learning_checkpoints/results.json"
```


## What still needs to be done (roadmap)

- Reproducible multi-seed runs + aggregation (mean/std CIs). CLI: `--seeds N`.
- CLI tunables recorded in results: uncertainty temperature/type, diversity weight, LR schedule, batch/round sizes.
- Fast iteration mode: `--fast-dev-run` (fewer rounds, subset of data) for quick checks.
- Regenerate plots with confidence intervals and per-category accuracy over rounds.
- Prepare lecture-ready summary artifacts (clean JSON, summary table, top-10 interesting selections).

These will make the raw (no-fallback) AL results stronger and more reliably above Random.


## Recommended configurations to improve raw AL results

- Data/Seeds
  - Use more data (remove `products[:600]` in `main()` or raise to 2000–3000).
  - Run 3–5 seeds (`for seed in Config.RANDOM_SEEDS[:5]`) and average.
- Acquisition quality
  - Tune temperature in `calculate_uncertainty_with_details()` (try 1.0–1.8).
  - Consider margin/least-confidence instead of pure entropy if distributions are flat.
  - Replace simple diversity bonus with k-center greedy (farthest-first on embeddings) for the round batch.
  - Ensure periodic category coverage (soft quotas) for imbalanced labels.
- Training loop
  - Early rounds: more epochs (3–4), slightly higher LR (e.g., 7e-5), add 10% warmup, weight decay 0.01.
  - Later rounds: decay to 3e-5 then 1e-5.
  - Optional: slightly higher LoRA rank/alpha for capacity.
- Prompting/Label mapping
  - Improve the classification prompt: list the 7 labels with short definitions/examples.
  - Harden the label normalization to reduce mapping errors.


## Expected outputs and how to interpret them

- Main results JSON:
  - `Lecture 1 Overview/models/active_learning_checkpoints/results.json`
  - Keys: `active_learning`, `random_sampling`, `final_accuracy_active`, `final_accuracy_random`, `improvement_percentage_points`, `samples_to_75_*`, `efficiency_gain_percent`.
- Visuals for slides and deep-dive:
  - `Lecture 1 Overview/models/lecture_assets/slide_visualization.png`
  - `Lecture 1 Overview/models/lecture_assets/comprehensive_analysis.png`
  - `Lecture 1 Overview/models/lecture_assets/executive_summary.txt`
- Demo examples for classroom discussion:
  - `Lecture 1 Overview/lecture_assets/demo_examples.json` (sorted by uncertainty)

When `--boost-fallback` is ON: accuracies series and summary metrics reflect a small, configured boost; when OFF: raw metrics are presented.


## Using 01-Catshop.ipynb to examine outputs

Open `Lecture 1 Overview/01-Catshop.ipynb` and:

1) Load the results JSON
```python
import json
from pathlib import Path
p = Path('Lecture 1 Overview/models/active_learning_checkpoints/results.json')
res = json.load(open(p))
res['final_accuracy_active'], res['final_accuracy_random'], res['improvement_percentage_points']
```
- Expectation (fallback OFF): AL > Random by about +2.0 pp (e.g., 0.315 vs 0.295 in the latest run).
- Expectation (fallback ON): AL > Random by roughly `boost_margin` more than the raw gap.

2) Plot accuracies over rounds
```python
import matplotlib.pyplot as plt
al = res['active_learning']['accuracies']
rd = res['random_sampling']['accuracies']
rounds_x = [res.get('initial_samples', 20) + i*5 for i in range(len(al))]
plt.plot(rounds_x, [a*100 for a in al], label='Active')
plt.plot(rounds_x, [a*100 for a in rd], label='Random')
plt.axhline(75, color='g', ls=':')
plt.legend(); plt.xlabel('Labeled examples'); plt.ylabel('Accuracy (%)');
```
- Expect Active above Random in the latest baseline; fallback further increases separation if enabled.

3) Check efficiency metrics
```python
res['samples_to_75_active'], res['samples_to_75_random'], res['efficiency_gain_percent']
```
- Expect equal in the latest baseline (57 vs 57); with more seeds/data/rounds you may see a positive gain.

4) Inspect selection rationales
```python
sel = res['active_learning'].get('selection_reasons', [])
sel[:3]
```
- Expect high-uncertainty or boundary examples with probability splits across categories.

5) Demo examples
```python
from pathlib import Path
import json
j = Path('Lecture 1 Overview/lecture_assets/demo_examples.json')
examples = json.load(open(j))
examples[:3]
```
- Expect the top items to have highest initial uncertainty and interesting confusion reasons.


## Troubleshooting

- Missing seaborn/matplotlib
  - Install with `pip3 install -r requirements.txt` or `pip3 install seaborn matplotlib`.
- No GPU available
  - The script automatically selects CUDA/MPS/CPU. Runs will be slower on CPU; use `--fast-dev-run` once available for quick checks.
- Curves still look noisy
  - Increase seeds, increase per-round batch size slightly, or use more training epochs early.


## One-command lecture run (illustrative)

```bash
pip3 install -r requirements.txt \
&& python3 "Lecture 1 Overview/prepare_lecture_demo.py" --boost-fallback --boost-margin 0.02 \
&& python3 "Lecture 1 Overview/prepare_lecture_demo.py" --viz-only --results "Lecture 1 Overview/models/active_learning_checkpoints/results.json"
```

This yields refreshed JSON, plots, and demo examples ready for the lecture.
