# SPAR: Midterm Progress

Risk estimation for a fixed focal VLM answer using peer answers. The scorer ranks answers for selective prediction; it never replaces the focal answer.

This repository is a midterm submission assembled from the current release. It contains the baseline evaluation pipeline, the exported SPAR score, a preliminary transfer experiment, and scoring-time measurements. It is not a historical checkout of the project at the presentation date.

## Included experiments

| Script | Purpose |
| --- | --- |
| `main_experiment.py` | Overall, per-dataset, and per-model selective prediction metrics |
| `baseline_experiment.py` | Confidence, disagreement, answer entropy, variation ratio, and SPAR comparisons |
| `transfer_experiment.py` | Preliminary source-to-target feature transfer comparison |
| `efficiency_experiment.py` | CPU scoring throughput on existing answers |
| `common.py` | JSONL loading, answer features, metrics, and synthetic data generation |
| `run.sh` | Run the four experiment entry points |
| `requirements.txt` | Python dependencies: NumPy and scikit-learn |
| `examples/` | Synthetic JSONL input, matching sample results, and data notes |
| `UPSTREAM.md` | Retained upstream provenance information |

Metrics include AURC (lower is better), error AUROC, and accuracy at 80% coverage. Scale studies, peer-capacity controls, deployment sweeps, and full ablations are outside this submission.

## Run

Install dependencies with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash run.sh
```

With no input, scripts generate deterministic synthetic examples and mark outputs with `demo: true`. These examples check the pipeline and are not VQAv2, A-OKVQA, TextVQA, or VizWiz evaluation results.

Outputs are written to `outputs/main.json`, `outputs/baseline.json`, `outputs/transfer.json`, and `outputs/efficiency.json`. Override the interpreter with `PYTHON=/path/to/python bash run.sh`.

A small saved synthetic example is also included:

```bash
python3 main_experiment.py --input examples/demo_answers.jsonl --output outputs/example.json
```

`examples/demo_results.json` contains the matching metrics. For this comparison, ignore the `demo` field: the CLI uses `demo: false` whenever `--input` is supplied, even if the supplied file is synthetic. The input file remains synthetic regardless of that flag.

Evaluate saved model answers instead:

```bash
bash run.sh outputs /path/to/answers.jsonl
python3 main_experiment.py --input /path/to/answers.jsonl
python3 transfer_experiment.py --input /path/to/answers.jsonl --source vqav2,aokvqa --target textvqa,vizwiz
```

The combined runner expects data for both source and target datasets. For a VQAv2-only export, run the main and baseline scripts individually:

```bash
python3 baseline_experiment.py --input /path/to/vqav2.jsonl
```

The transfer experiment requires disjoint source/target dataset names, nonempty subsets, and both correct and incorrect source answers. It saves learned coefficients and the intercept alongside the target metrics. Use `--help` on any experiment for its arguments. Efficiency repetitions can be changed with `--repeats` and must be positive.

Each JSONL line is one focal answer with its peer answers:

```json
{"sample_id":"example-1","dataset":"vqav2","model":"focal-vlm","answer":"two people","confidence":0.8,"correct":1,"peer_answers":["two people","2 people","three people"],"peer_confidences":[0.8,0.7,0.6]}
```

`correct` is a binary evaluation label, not an input to the fixed risk scorer. Confidence values must lie in [0, 1]. Peer confidences are optional; when provided, their count must match the peer answers. The scripts consume saved answers and do not query VLMs.

## Method implemented here

`common.py` computes token F1 between the focal answer and each peer, confidence-weighted disagreement, and normalized answer-count entropy. The fixed SPAR score uses the supplied trained weights:

```text
SPAR = 0.40 * (1 - focal confidence)
     + 0.35 * weighted peer disagreement
     + 0.25 * peer answer entropy
```

The original L1 training run and its parameter export are not included. The transfer script fits a separate L2 logistic regression; it does not reproduce that L1 training run. The `semantic_entropy` field is an existing name for normalized text-count entropy, without semantic clustering.

## Relation to the midterm presentation

The presentation describes the broader SPAR method: an all-pairs answer graph, isolation/ambiguity decomposition `D = I + A`, answer-pattern features, rank alignment, and an L1 risk model. This submission contains the reduced scoring and evaluation implementation available in the release. The full graph decomposition, rank alignment, and L1 training pipeline are not included here.

The presentation reports a 7.05% macro-AURC reduction under joint model and dataset shift and approximately 10x speedup over SCoOP. Real prediction exports, the SCoOP implementation, inference timings, and corresponding experiment logs are not included, so the demo does not reproduce those numbers. The efficiency script measures scoring only, excluding VLM calls. The transfer split is by dataset and does not enforce held-out models.

## Data preparation

For actual evaluation, obtain the datasets from their official sources:

- [VQAv2](https://visualqa.org/download.html): download the questions, annotations, and corresponding COCO images for the chosen split.
- [A-OKVQA](https://github.com/allenai/aokvqa): follow the dataset download instructions and obtain the matching COCO images.
- [TextVQA](https://textvqa.org/download/): obtain the image and annotation files for the chosen split.
- [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/): obtain the VQA images and annotations for the chosen split.

Generate focal and peer answers externally, retain the model checkpoints and decoding settings, and export one record per focal answer using the schema above. Compute binary `correct` labels from the reference annotations under an explicitly recorded correctness rule. The scripts do not convert raw datasets, perform model inference, or compute labels from references. To reproduce a benchmark, prediction files must include the actual evaluated sample IDs and model outputs.

## Provenance

The supplied release identifies [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) as its upstream source. The retained details are in [UPSTREAM.md](UPSTREAM.md); the original source history is not bundled here. This submission consists of standalone saved-answer scoring and evaluation scripts rather than the upstream task/model evaluation framework. The compact SPAR features, fixed scoring rule, and experiment entry points describe the local implementation; the retained provenance record does not establish a file-by-file derivation.

## Remaining work for the midterm scope

- Attach real baseline and transfer evaluation exports with their run configurations.
- Include the structured feature extraction and L1 parameter export used for the reported method.
- Compare scalar and structured features through ablation and error analysis.
- Validate joint model/dataset transfer and end-to-end runtime against the reported baseline.
