# Experiment 3 — Panel Subcaption Generation

Fine-tune a VLM on (panel image → 20-30 word subcaption) and compare against
zero-shot and few-shot baselines.

## Project structure

```
exp3_captioning/
├── data/
│   ├── prepare_data.py       # build HF dataset + stratified splits
│   └── dataset.py            # PyTorch Dataset + collator
├── train.py                  # QLoRA fine-tuning
├── baselines/
│   └── zero_shot.py          # zero-shot and 3-shot baselines
├── evaluation/
│   ├── metrics.py            # BLEU / METEOR / ROUGE-L / BERTScore
│   ├── llm_judge.py          # Claude-as-judge domain scoring
│   ├── run_finetuned.py      # inference with fine-tuned adapter
│   └── compare_results.py    # cross-condition table + LaTeX
└── requirements.txt
```

---

## Step 0 — Install

```bash
pip install -r requirements.txt
```

---

## Step 1 — Prepare data

Your annotation JSONs + panel images should be in one folder.
See `data/prepare_data.py` for the two supported layout conventions.

```bash
python data/prepare_data.py \
    --data_root /path/to/your/data \
    --out_dir   ./datasets \
    --val_frac  0.10 \
    --test_frac 0.10
# Output: ./datasets/matfig_captioning/{train, val, test}
```

---

## Step 2 — Zero-shot + 3-shot baseline

```bash
python -m baselines.zero_shot \
    --dataset_dir ./datasets/matfig_captioning \
    --model_name  Qwen/Qwen2-VL-2B-Instruct \
    --output_dir  ./results/zero_shot \
    --mode        both
```

Compute metrics:
```bash
python -m evaluation.metrics \
    --predictions_json ./results/zero_shot/zero_shot_predictions.json \
    --output_dir       ./results/zero_shot

python -m evaluation.metrics \
    --predictions_json ./results/zero_shot/few_shot_predictions.json \
    --output_dir       ./results/few_shot
```

---

## Step 3 — Fine-tune (QLoRA)

**Recommended hardware:** 1× A100 40GB or 2× A40 (use `accelerate launch`).

Single GPU:
```bash
python train.py \
    --dataset_dir  ./datasets/matfig_captioning \
    --output_dir   ./checkpoints/qwen2vl_subcap \
    --model_name   Qwen/Qwen2-VL-2B-Instruct \
    --epochs       5 \
    --batch_size   8 \
    --grad_accum   4 \
    --lr           2e-4
```

Multi-GPU (2 GPUs):
```bash
accelerate launch --num_processes 2 train.py \
    --dataset_dir  ./datasets/matfig_captioning \
    --output_dir   ./checkpoints/qwen2vl_subcap \
    --batch_size   4
```

Monitor with TensorBoard:
```bash
tensorboard --logdir ./checkpoints/qwen2vl_subcap
```

---

## Step 4 — Fine-tuned inference + metrics

```bash
python -m evaluation.run_finetuned \
    --adapter_dir  ./checkpoints/qwen2vl_subcap/final_adapter \
    --base_model   Qwen/Qwen2-VL-2B-Instruct \
    --dataset_dir  ./datasets/matfig_captioning \
    --output_dir   ./results/finetuned

python -m evaluation.metrics \
    --predictions_json ./results/finetuned/finetuned_predictions.json \
    --output_dir       ./results/finetuned
```

---

## Step 5 — LLM-as-judge (optional, needs ANTHROPIC_API_KEY)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m evaluation.llm_judge \
    --predictions_json ./results/zero_shot/zero_shot_predictions.json \
    --dataset_dir      ./datasets/matfig_captioning \
    --output_dir       ./results/zero_shot \
    --n_sample         200

python -m evaluation.llm_judge \
    --predictions_json ./results/finetuned/finetuned_predictions.json \
    --dataset_dir      ./datasets/matfig_captioning \
    --output_dir       ./results/finetuned \
    --n_sample         200
```

---

## Step 6 — Final comparison table

```bash
python -m evaluation.compare_results \
    --results_dirs \
        ./results/zero_shot \
        ./results/few_shot \
        ./results/finetuned \
    --output_dir ./results/final
# Outputs: combined_metrics.json  +  table_captioning.tex
```

---

## Expected result structure (for the paper)

| Model                  | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BERTScore-F1 |
|------------------------|--------|--------|--------|---------|--------------|
| Qwen2-VL-2B zero-shot  |        |        |        |         |              |
| Qwen2-VL-2B 3-shot     |        |        |        |         |              |
| Qwen2-VL-2B fine-tuned |        |        |        |         |              |

Plus per-subtype breakdown (SEM / TEM / Line Graph / etc.) and
LLM-as-judge scores (domain_accuracy / visual_grounding / completeness).

---

## Swapping the base model

Change `--model_name` in all scripts.
Tested-compatible models:
- `Qwen/Qwen2-VL-2B-Instruct`  (default, fits on 1× A40)
- `Qwen/Qwen2-VL-7B-Instruct`  (needs 1× A100 80GB)
- `llava-hf/llava-1.5-7b-hf`
- `google/paligemma2-3b-pt-224`
