# Compound Figure Panel Detection

End-to-end pipeline for detecting and labelling sub-panels (A, B, C, …) in compound scientific figures. Covers panel detection, multi-model benchmarking, and cross-modal retrieval.


---

## Repository Layout

```
Compoun_img_01/
├── MatDetect/          # DAB-DETR panel detector — training, eval, inference
├── ModelBench/         # Multi-model benchmark (YOLO v8/9/10/11/12 + DAB-DETR)
├── Testing/            # Unified test-split comparison across all models
├── retrieval/          # Cross-modal dual-encoder retrieval module
└── main_data/          # Source data, panel cropping, multimodal dataset building
```

---

## Environment Setup

All deep-learning scripts run inside a single conda environment (Python 3.12, CUDA 12.4).

**Create from YAML (first time):**
```bash
conda env create -f environment.yaml
conda activate rfdetr_env
```

**Or activate existing env:**
```bash
conda activate /mnt/d/Subham/Compoun_img_01/ModelBench/rfdetr_env
```

---

## 1. MatDetect — Panel Detector (DAB-DETR)

Fine-tunes `IDEA-Research/dab-detr-resnet-50` on YOLO-format labelled data to detect compound figure panels.

**Data format** — YOLO layout with 21 reported classes (A–T, single):
```
all_mydata/
├── train/images/   &   train/labels/
├── val/images/     &   val/labels/
└── test/images/    &   test/labels/
```

**Train**
```bash
cd MatDetect
python train.py \
    --train-dir ./all_mydata/train \
    --val-dir   ./all_mydata/val \
    --output    ./checkpoints_mydata
```

**Evaluate on test split**
```bash
python test.py \
    --checkpoint ./checkpoints_mydata/best \
    --data-dir   ./all_mydata/test
```

**Batch inference**
```bash
python infer.py \
    --image-dir  ./all_mydata/test/images \
    --checkpoint ./checkpoints_mydata/best
```

Best checkpoint: `MatDetect/checkpoints/epoch046` (original data) / `checkpoints_mydata/` (all_mydata).

---

## 2. ModelBench — Multi-Model Benchmark

Trains and evaluates all YOLO variants and DAB-DETR on the same dataset. Results are saved as JSON per model.

### Registry files

| File | Purpose |
|---|---|
| `models_registry.py` | Benchmark registry — YOLOv8m, YOLOv9c, YOLOv10m, YOLO11m, YOLO12m + DAB-DETR |
| `models_registry_mydata.py` | Mydata retraining — same 5 YOLO + DAB-DETR |

### Train all models (original data)

```bash
cd ModelBench
python benchmark.py                        # train everything
python benchmark.py --only yolo12m yolo11m # specific models
python benchmark.py --skip-done            # skip already finished
python benchmark.py --dry-run              # preview without running
```

Results saved to `ModelBench/results/`.

### Retrain on all_mydata

```bash
python benchmark_mydata.py
python benchmark_mydata.py --only yolo12m yolo11m
python benchmark_mydata.py --skip-done
```

Results saved to `ModelBench/results_mydata/`. Requires `data_mydata.yaml` (already created).

### Compute APr / APc / APf (rare / common / frequent class split)

Patches existing result JSONs in-place with frequency-stratified AP metrics.

```bash
python eval_extended.py \
    --val-dir     ../MatDetect/all_mydata/test \
    --results-dir ./results \
    --force

# process only specific models
python eval_extended.py \
    --val-dir     ../MatDetect/all_mydata/test \
    --results-dir ./results \
    --models      yolov8m,yolov9c,yolov10m,yolo11m,yolo12m \
    --force
```

### View benchmark table

```bash
python compare.py
```

---

## 3. Testing — Full Test-Split Comparison

Runs all models on the held-out test split with a unified evaluation protocol (standard NMS, same conf/IoU for all).

```bash
cd Testing
python compare_all.py
python compare_all.py --conf 0.3 --iou 0.5
python compare_all.py --max-samples 50
python compare_all.py --bootstrap 1000   # confidence intervals
```

Outputs per model:
- `results_comparison/<model>/results.json` — full numeric record
- `results_comparison/<model>/per_class.csv` — per-class P/R/F1/AP
- `results_comparison/<model>/confusion_matrix.csv`
- `results_comparison/comparison_summary.csv` — one row per model

Models compared: all YOLO variants from `ModelBench/runs/detect/runs_mydata/` + DAB-DETR checkpoint.

---

## 4. main_data — Multimodal Dataset Building

Builds the multimodal image-caption dataset from raw compound figures across four material domains (alloy, ceramics, composite, nickel alloy).

**Pipeline:**

**Step 1 — Run panel detector on raw images**
```bash
cd main_data
python infer.py --image-dir ./alloy_elsevier_cc_by_contents --checkpoint ../MatDetect/checkpoints/epoch046
```
Saves per-image detection JSON to `inference_results/`.

**Step 2 — Crop detected panels**
```bash
python crop_panels.py \
    --image-dir ./alloy_elsevier_cc_by_contents \
    --json-dir  ./inference_results \
    --output-dir ./alloy_prod_crops
```
Saves individual panel crops as `imgXXXX_A.jpg`, `imgXXXX_B.jpg`, etc.

**Step 3 — Link crops to generated subcaptions**
```bash
python build_dataset.py
```
Pairs each crop (`alloy_prod_crops/`) with its JSON subcaption (`generated_subcaptions_alloy_prod/`) and outputs `alloy_linked_dataset.csv`.

Repeat Steps 1–3 for each domain (ceramics, composite, ni_alloy) — each has its own crops, JSON, and linked CSV.

**Step 4 — Upload to HuggingFace**
```bash
python upload_dataset.py
python upload_dataset.py --shard-size 1GB
```
Streams all crops + metadata to HuggingFace as Parquet shards. Columns: `image`, `image_id`, `panel_suffix`, `visualization_category`, `visualization_subtype`, `subcaption`, `summary`.

**Output CSVs per domain:**
- `alloy_linked_dataset.csv`
- `ceramics_linked_dataset.csv`
- `composite_linked_dataset.csv`
- `ni_alloy_linked_dataset.csv`

---

## 5. Retrieval — Cross-Modal Dual-Encoder

Dual-encoder architecture (vision + text) trained with InfoNCE loss for image↔caption retrieval at scale.

```bash
cd retrieval

# train
python -m retrieval.main \
    --output_dir ./outputs_proper \
    --batch_size 64 \
    --num_epochs 20

# evaluate only (skip training)
python -m retrieval.main --eval_only

# resume from checkpoint
python -m retrieval.main --resume ./outputs_proper/checkpoints/best.pt
```

Production model is in `retrieval/outputs_proper/`.

---

## Data Overview

| Path | Contents |
|---|---|
| `MatDetect/all_mydata/` | Full annotated dataset — train / val / test splits, YOLO format, 21 classes |
| `MatDetect/data/` | Original benchmark data — train + val only |
| `MatDetect/checkpoints/epoch046` | Best DAB-DETR (original data) |
| `MatDetect/checkpoints_mydata/` | Best DAB-DETR (all_mydata) |
| `ModelBench/runs/detect/runs_mydata/` | Fine-tuned YOLO weights per model |
| `ModelBench/results/` | Per-model benchmark JSONs (val metrics + APr/APc/APf) |
| `retrieval/outputs_proper/` | Production dual-encoder weights |

---

## Class Labels

21 classes reported: A–T (20 letter labels) + `single`. The `common` class (ID 21) has no training examples and is excluded from all evaluation metrics.

| IDs | Labels |
|---|---|
| 0–19 | A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T |
| 20 | single |
