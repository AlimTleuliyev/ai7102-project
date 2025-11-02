# How to Run with Your Own Models

This repository is for **inference and analysis only** - you bring your own pretrained language models!

## Quick Start: Use Any HuggingFace Model

### 1. ✅ You already have preprocessed data!

**VERIFIED**: Your `data/DC/` folder contains **31 context-modified JSON files** ready to use:
- 7 n-gram sizes: `2, 3, 5, 7, 10, 20, 1000`
- 5-6 context functions: `delete`, `lossy-0.5`, `lossy-0.25`, `lossy-0.125`, `lossy-0.0625`
- Eye-tracking data: `all.txt.annotation.filtered.csv` (515,010 observations)
- Full context baseline: `pieces.json`

**You can skip preprocessing and jump straight to running your model!**

<details>
<summary>🔧 If you need to regenerate (click to expand)</summary>

```bash
# Only if you modified preprocessing scripts or want different parameters
cd preprocess/DC
bash modify_context.sh
```
</details>

### 2. Run surprisal calculation with YOUR model:

```bash
# Example with GPT-2 (2-gram context)
python experiments/calc_surprisal_hf.py \
  -m gpt2 \
  -o surprisals/my-test/arch_gpt2-ngram_2-contextfunc_delete \
  --batchsize 10 \
  -d data/DC/ngram_2-contextfunc_delete.json

# Example with Llama-2 (7-gram context)
python experiments/calc_surprisal_hf.py \
  -m meta-llama/Llama-2-7b-hf \
  -o surprisals/my-test/arch_llama2-ngram_7-contextfunc_delete \
  --batchsize 4 \
  -d data/DC/ngram_7-contextfunc_delete.json

# Example with YOUR custom model
python experiments/calc_surprisal_hf.py \
  -m your-username/your-model-name \
  -o surprisals/my-test/arch_mymodel-ngram_2-contextfunc_delete \
  --batchsize 8 \
  -d data/DC/ngram_2-contextfunc_delete.json
```

**⚠️ Important**: Output directory name should follow pattern: `arch_MODELNAME-ngram_N-contextfunc_FUNC`
- This helps scripts parse experimental conditions automatically

**Output**: Creates `scores.json` and `eval.txt` in the specified directory

### 3. Run statistical analysis:

```bash
# Convert scores to regression format
python experiments/convert_scores.py --dir surprisals/my-test

# Run mixed-effects regression (calculates PPP)
python experiments/dundee.py surprisals/my-test/
```

**Output**: Creates `scores.csv` and `likelihood.txt` in each subdirectory

### 4. Compare multiple models:

```bash
# Run for different context lengths
for ngram in 2 3 5 7 10 20 1000; do
  python experiments/calc_surprisal_hf.py \
    -m gpt2 \
    -o surprisals/gpt2-test/arch_gpt2-ngram_${ngram}-contextfunc_delete \
    --batchsize 10 \
    -d data/DC/ngram_${ngram}-contextfunc_delete.json
done

# Analyze all at once
python experiments/convert_scores.py --dir surprisals/gpt2-test
python experiments/dundee.py surprisals/gpt2-test/

# Aggregate results into CSV
python experiments/aggregate.py --dir surprisals/gpt2-test --file likelihood.txt
```

## What Models Can You Use?

### ✅ Any HuggingFace causal LM:
- **GPT family**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **Llama family**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-3-8B`
- **Mistral**: `mistralai/Mistral-7B-v0.1`
- **Pythia**: `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-6.9b`
- **Your own model**: Upload to HuggingFace Hub and use the model name

### Requirements:
- Must be a **causal language model** (left-to-right)
- Must work with `AutoModelForCausalLM.from_pretrained()`
- That's it!

## Understanding the Output

### `scores.json`
```json
{
  "article_1": [0.234, 1.456, 0.789, ...],  // surprisal per word
  "article_2": [2.345, 0.123, ...]
}
```

### `eval.txt`
```json
{"PPL": 156.3}  // Perplexity (lower = better prediction)
```

### `likelihood.txt`
```
linear_fit_logLik: -0.00234
delta_linear_fit_logLik: 0.00056  ← This is PPP!
delta_linear_fit_chi_p: 0.0001
```

**PPP (Psychometric Predictive Power)** = How well surprisal predicts human reading times
- Higher PPP = more human-like
- Paper found: 2-gram models have higher PPP than full-context!

## Full Pipeline Example

```bash
#!/bin/bash
# Complete workflow for your model

MODEL="gpt2"  # Change to your model
OUTPUT_DIR="surprisals/my-experiment"

# 1. Calculate surprisals for all context lengths
for ngram in 2 3 5 7 10 20 1000; do
  python experiments/calc_surprisal_hf.py \
    -m $MODEL \
    -o ${OUTPUT_DIR}/arch_gpt2-ngram_${ngram}-contextfunc_delete \
    --batchsize 10 \
    -d data/DC/ngram_${ngram}-contextfunc_delete.json
done

# 2. Convert and analyze
python experiments/convert_scores.py --dir $OUTPUT_DIR
python experiments/dundee.py $OUTPUT_DIR

# 3. Aggregate results
python experiments/aggregate.py --dir $OUTPUT_DIR --file likelihood.txt > ${OUTPUT_DIR}/results.csv

# 4. View results
cat ${OUTPUT_DIR}/results.csv
```

## What If I Want to Use Fairseq Models?

You need:
1. A Fairseq checkpoint: `checkpoint.pt`
2. A SentencePiece tokenizer: `tokenizer.model`
3. Binarized training data in `data/en_sents/data-bin/`

Then use `calc_surprisal.py` instead of `calc_surprisal_hf.py`:

```bash
python experiments/calc_surprisal.py \
  -m /path/to/checkpoint.pt \
  -a lstm  # or transformer, gpt
  -o surprisals/my-fairseq-model \
  --batchsize 50 \
  -d data/DC/ngram_2-contextfunc_delete.json \
  -i \
  --corpus dundee
```

**Note**: This is more complex and requires the full Fairseq setup. HuggingFace is much easier!

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers torch
```

### "CUDA out of memory"
Reduce batch size: `--batchsize 4` or `--batchsize 1`

### "Model not found"
Make sure model is public or you're logged in:
```bash
huggingface-cli login
```

### "File not found: data/DC/ngram_X..."
Run preprocessing first:
```bash
cd preprocess/DC && bash modify_context.sh
```
