# SPAR: Saved-answer Evaluation

This repository evaluates the risk of a fixed focal VLM answer using saved focal and peer responses. It extends the midterm pipeline with model-scale, peer-capacity, ablation, robustness, calibration, bootstrap, and subsample analyses. The scorer ranks focal answers for selective prediction and never replaces them.

## Input and scope

Each JSONL row represents one focal answer:

```json
{"sample_id":"example-1","dataset":"vqav2","model":"focal-vlm","model_size_b":4,"answer":"two people","confidence":0.8,"correct":1,"peer_answers":["two people","2 people","three people"],"peer_confidences":[0.8,0.7,0.6]}
```

The scripts consume saved answers and compute selective-prediction metrics. They do not run VLM inference, download datasets, or reproduce training of the paper's full graph-feature L1 model. The `correct` field is used for evaluation and the separate L2 transfer fit, but not by the fixed SPAR scorer.

The implemented fixed score is

```text
SPAR = 0.40 * (1 - focal confidence)
     + 0.35 * confidence-weighted focal/peer disagreement
     + 0.25 * normalized peer-answer entropy
```

These supplied weights are fixed. Their original training and parameter export are not included. The `semantic_entropy` output key is retained for compatibility and denotes entropy over normalized answer strings, without semantic clustering.

## Experiments

| Script | Output |
| --- | --- |
| `main_experiment.py` | Overall, dataset, and model risk-ranking metrics |
| `baseline_experiment.py` | Confidence, disagreement, answer-count entropy, and fixed SPAR comparisons |
| `transfer_experiment.py` | Source-to-target L2 transfer on saved features |
| `efficiency_experiment.py` | CPU scoring time on saved answers; excludes VLM inference |
| `scale_experiment.py` | Metrics grouped by `model_size_b` |
| `control_experiment.py` | Peer-count sweep and selective-coverage accuracy |
| `ablation_experiment.py` | Fixed-score component comparisons |
| `robustness_experiment.py` | Confidence and peer-answer perturbations |
| `calibration_experiment.py` | Brier score and expected calibration error |
| `significance_experiment.py` | Row-bootstrap AURC difference versus focal confidence |
| `subsample_stability_experiment.py` | Variation across seeded synthetic runs or row subsamples |

The added experiments are diagnostics on saved predictions. They do not by themselves reproduce the paper's benchmark tables or measured generation-time comparison. In particular, the row-bootstrap script does not account for image-level dependence.

## Run

Python 3.10 or newer and the packages in `requirements.txt` are required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash run.sh
```

With no input, every script uses deterministic synthetic data and marks its output with `"demo": true`. These outputs verify execution and are not experimental results.

To evaluate a saved-answer export:

```bash
bash run.sh outputs /path/to/answers.jsonl
```

The combined runner executes all eleven experiments. Its input must include both source and target datasets for transfer evaluation and `model_size_b` in every row for the scale analysis. Run applicable scripts individually when an export lacks those fields:

```bash
python3 main_experiment.py --input /path/to/answers.jsonl
python3 ablation_experiment.py --input /path/to/answers.jsonl
```

The transfer defaults use `vqav2,aokvqa` as source datasets and `textvqa,vizwiz` as targets. Source and target names must be disjoint, both subsets must be nonempty, and the source subset must contain correct and incorrect answers.

`examples/demo_answers.jsonl` and `examples/demo_results.json` provide synthetic saved input and reference metrics for the main experiment.

## Reproducibility boundary

The full study uses an all-pairs answer graph, percentile-rank alignment, and an L1 risk model. This repository contains a compact fixed-weight score and a separate L2 transfer fit. It does not include the complete 50-feature extraction, raw VLM prediction exports, generation logs, or the training artifacts behind the report tables. Scoring-time measurements exclude VLM generation.

The retained upstream record is in [UPSTREAM.md](UPSTREAM.md). The original source history and full paper pipeline are not bundled here.
