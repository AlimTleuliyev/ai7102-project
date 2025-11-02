# Copilot Instructions - Context Limitation Cognitive Modeling

## Project Overview
Research codebase for EMNLP 2022 paper "Context Limitations Make Neural Language Models More Human-Like" ([arXiv:2205.11463](https://aclanthology.org/2022.emnlp-main.712/)).

**Research Question**: Can constraining neural LMs' context access make them better models of human sentence processing?

**Core Finding**: LMs with limited context (even 2-gram!) better predict human gaze duration than full-context models. This gap grows with model size - larger models deviate more from human-like context access.

**Key Metric**: **Psychometric Predictive Power (PPP)** = per-token log-likelihood difference between regression models with/without surprisal features. Higher PPP = surprisal better predicts human reading times.

**Note**: This project focuses on **English data only** (Dundee Corpus). Japanese components (BCCWJ) are excluded from scope.

## Quick Start

### 1. Data Preparation
Before running any experiments, you need to download the preprocessed data (~133 MB):

```bash
# Make the script executable
chmod +x prepare_data.sh

# Download and extract data
./prepare_data.sh
```

This will download and extract:
- `data/DC/` - Dundee eye-tracking corpus with linguistic annotations
- `data/en_sents/` - Training data for Wiki-LMs (Fairseq format)

**Note**: The data is hosted by the original paper authors. If the download fails, contact `kuribayashi.research [at] gmail.com` for access.

### 2. Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# R packages (for statistical analysis)
# Install manually: cvms, mgcv, lme4, dplyr
```

### 3. Run a Quick Test
Test the pipeline with a single model configuration:

```bash
# Generate context variants (if not already done)
cd preprocess/DC && bash modify_context.sh && cd ../..

# Run surprisal calculation with GPT-2
python experiments/calc_surprisal_hf.py \
  -m gpt2 \
  -o surprisals/quick-test \
  -d data/DC/ngram_2-contextfunc_delete.json

# Statistical analysis
python experiments/convert_scores.py --dir surprisals/quick-test --corpus dundee
python experiments/dundee.py surprisals/quick-test/
```

## Architecture & Data Flow

### The Surprisal-Gaze Duration Pipeline
The codebase tests whether **lossy-context surprisal** (Eq. 1 in paper) predicts human reading times:

```
I_lossy(w_i, c_<i) = -log p_θ(w_i | <s> ◦ f([w_0,...,w_{i-1}]))
```

Where `f` is a **context noise function**:
1. **`delete`** (n-gram): Keep only last n-1 tokens → forces n-gram context
2. **`lossy-{slope}`** (probabilistic): Delete distant tokens with probability `min(j * slope, 1)` where j=distance

**Data Flow**: Preprocess → Generate context variants → Compute surprisals → Fit mixed-effects models → Measure PPP

### English Corpus (Dundee)
- **DC (Dundee)**: English SVO, mixed branching (`data/DC/`)
- **Reading units**: Space-separated words (avg 1.3±0.7 subwords/word)
- **Data points**: 212,649 gaze duration measurements after filtering
- **Surprisal aggregation**: Summed over subwords within each word (Eq. 2 in paper)

### The Training-Inference Mismatch Problem
**Problem**: During n-gram inference, models predict from mid-sentence with limited context, but training uses full documents.

**Solution (Wiki-LMs only)**: Modified training corpus with random sentence chunking:
```
... <b> _was _also _the _first _hotel <b> _on _4 _March <b> ...
```
Special token `<b>` signals context breaks. Models learn to predict from arbitrary positions with limited context. See `preprocess/DC/modify_context.py` implementation and Appendix E.

## Key Workflows

### Full Experimental Pipeline (`run.sh`)
Reproduces English experiments with multiple LM configurations:

1. **Preprocessing** (`preprocess/DC/`):
   - Extract sentences from Dundee eye-tracking corpus
   - Tokenize (space-split for English)
   - Add linguistic annotations (dependency structure, frequency, length)
   - Filter outliers (Appendix C criteria: zero gaze, special chars, first/last in line)
   - Generate context-modified variants via `modify_context.sh`

2. **Surprisal Calculation** (`experiments/en_surprisal*.sh`):
   - Run each LM configuration (3 architectures × 3 seeds × 7 n-grams × 5 lossy-slopes)
   - Compute batched cross-entropy loss per subword position
   - Aggregate to word level (Eq. 2)
   - Output: `scores.json` (article→surprisals) + `eval.txt` (PPL)

3. **Statistical Modeling** (`experiments/dundee.r`):
   - Fit baseline: `time ~ freq * length + controls + (1|article) + (1|subj)`
   - Fit test: Add `surprisals_sum` predictor
   - Compute PPP = ΔlogLik / N (likelihood ratio per token)
   - Output: `likelihood.txt` with log-likelihood values

4. **Aggregation** (`experiments/aggregate.py`):
   - Parse directory structure to extract experimental conditions
   - Combine PPP + PPL across seeds/conditions
   - Generate CSV for visualization

5. **Visualization** (`experiments/visualize.ipynb`):
   - Reproduce Figure 1 (PPP vs input length), Figure 2 (PPP gain by model size)
   - Analyze dependency locality effects (Section 6)

### Quick Testing Workflow
When modifying code, test on single condition before full sweep:

```bash
# 1. Regenerate ONE context variant (faster than full sweep)
python preprocess/DC/modify_context.py -n 7 --context-func delete

# 2. Run SINGLE surprisal calculation (one arch, one seed)
python experiments/calc_surprisal.py \
  -m models/en_lm/lstm/seed-1/checkpoint.pt \
  -o surprisals/test -a lstm --batchsize 50 \
  -d data/DC/ngram_7-contextfunc_delete.json -i --corpus dundee

# 3. Validate outputs exist and are reasonable
ls surprisals/test/  # Check for scores.json, eval.txt
cat surprisals/test/eval.txt  # PPL should be ~100-1000, not NaN

# 4. Run statistical analysis
python experiments/convert_scores.py --dir surprisals/test --corpus dundee
Rscript experiments/dundee.r surprisals/test/
cat surprisals/test/*/likelihood.txt  # Check log-likelihood values
```

### Model Variants & Where They Live
Three LM implementations, each with distinct characteristics:

**1. Wiki-LMs** (custom-trained, `experiments/calc_surprisal.py`) - **REQUIRES AUTHOR PERMISSION**:
- **Architectures**: LSTM-xs (27M), GPT2-xs (29M), GPT2-md (335M)
- **Training**: 4M English Wikipedia sentences with modified chunking (handles mismatch problem)
- **Location**: `models/en_lm/{lstm,transformer,gpt}/{seed-1,seed-39,seed-46}/checkpoint.pt`
- **Tokenizer**: SentencePiece model in `models/spm_en/en_wiki.model`
- **Data bins**: Pre-processed Fairseq format in `data/en_sents/data-bin{,-ngram}`
- **Key args**: `--add-itag` (use `<b>` tokens), `--add-bos` (explicit BOS handling), `--corpus dundee`
- **Why 3 seeds?** Results averaged across random seeds to reduce variance (see Figure 1 error bars)

**2. Pretrained GPT-2s** (OpenAI models, `experiments/calc_surprisal_hf.py`) - **NO PERMISSION NEEDED**:
- **Variants**: gpt2-sm (117M), gpt2-md (345M), gpt2-lg (774M), gpt2-xl (1558M)
- **Training**: 40GB WebText (no modified chunking - training-inference gap present)
- **Location**: Downloaded via HuggingFace `AutoModelForCausalLM.from_pretrained()`
- **Tokenizer**: GPT-2 BPE (vocab=50257), removes `▁` whitespace markers during preprocessing
- **Usage**: `experiments/en_surprisal_hf.sh` runs all 4 sizes across context conditions
- **🎯 USE THIS FOR YOUR OWN MODELS**: Works with ANY HuggingFace causal LM!

**3. Vanilla Wiki-LMs** (ablation study, `experiments/en_surprisal_vanilla.sh`) - **REQUIRES PERMISSION**:
- Same architectures as Wiki-LMs but trained WITHOUT modified chunking
- **Purpose**: Figure 3 ablation - proves training modification reduces mismatch bias
- **Location**: `models/en_lm_vanilla/`
- **Result**: Slightly underestimates short-context advantage vs modified LMs

## 🚀 Using Your Own Models

**You don't need the author's models!** This repo is for inference/analysis. See `RUN_WITH_YOUR_MODEL.md` for details.

**Quick start with any HuggingFace model**:
```bash
# Generate context variants
cd preprocess/DC && bash modify_context.sh

# Run your model
python experiments/calc_surprisal_hf.py \
  -m your-model-name \
  -o surprisals/my-test \
  -d data/DC/ngram_2-contextfunc_delete.json

# Analyze
python experiments/convert_scores.py --dir surprisals/my-test --corpus dundee
python experiments/dundee.py surprisals/my-test/
```

## What Lives Where - File Navigation Guide

### Data Files (`data/`)
```
data/
├── DC/  # English Dundee corpus
│   ├── all.txt.annotation                        # Raw eye-tracking with linguistic features
│   ├── all.txt.annotation.filtered.csv           # After outlier removal (Appendix C)
│   ├── *.data4modeling                           # Final regression input (212,649 data points)
│   ├── article_order.txt                         # Canonical article ordering for aggregation
│   ├── ngram_{2,3,5,7,10,20,1000}-contextfunc_{delete,lossy-*}.json  # Modified contexts
└── en_sents/data-bin{,-ngram}/                   # Fairseq binarized training data
```

### Experiment Scripts (`experiments/`)
```
experiments/
├── calc_surprisal.py           # Core: Compute surprisal with Fairseq models (Wiki-LMs)
├── calc_surprisal_hf.py        # Core: Compute surprisal with HuggingFace (GPT-2s)
├── en_surprisal.sh             # Run 105 settings (3 arch × 3 seeds × 7 n-grams × 5 slopes)
├── en_surprisal_vanilla.sh     # Ablation: models without training modification
├── en_surprisal_hf.sh          # Run 140 settings (4 sizes × 7 n-grams × 5 slopes)
├── convert_scores.py           # Transform scores.json → scores.csv for R input
├── dundee.r                    # Mixed-effects regression → PPP calculation
├── aggregate.py                # Collect results across seeds/conditions → CSV
└── visualize.ipynb             # Generate Figure 1, 2, dependency analyses (Section 6)
```

### Preprocessing (`preprocess/DC/`)
```
preprocess/DC/
├── add_annotation.py          # Add dependency, frequency, locality features
├── filter.py                  # Remove outliers (zero gaze, punctuation, etc.)
├── data_points4modeling.py    # Format for R regression input
├── modify_context.py          # Generate n-gram/lossy variants (implements Eq. 1 noise function)
└── modify_context.sh          # Loop: 6 n-grams × 5 slopes = 30 variants
```

### Results Storage (`surprisals/`)
Directory structure encodes experimental conditions:
```
surprisals/
├── DC/  # English Wiki-LM results
│   └── arch_{lstm,transformer,gpt}-ngram_{2-1000}-contextfunc_{delete,lossy-*}/
│       ├── seed-1/
│       │   ├── scores.json      # Article→surprisal mapping
│       │   ├── scores.csv       # Converted for R (with prev_1, prev_2 lags)
│       │   ├── eval.txt         # {"PPL": 156.3} - next-word prediction accuracy
│       │   └── likelihood.txt   # base_logLik: -X.XX\nsup_logLik: -Y.YY (PPP = ΔlogLik/N)
│       ├── seed-39/ ...
│       └── seed-46/ ...
├── DC-vanilla/  # Ablation without training modification
└── DC-hf/  # GPT-2 results (no seed subdirs - single pretrained model)
    └── arch_gpt2{,_medium,_large,_xl}-ngram_*-contextfunc_*/
```

### Model Checkpoints (`models/`)
```
models/
├── en_lm/{lstm,transformer,gpt}/{seed-1,39,46}/checkpoint.pt  # Wiki-LMs
├── en_lm_vanilla/...                                           # Ablation models
├── spm_en/en_wiki.{model,vocab}                               # SentencePiece for English
└── unigram_en.json                                            # Frequency dict for annotation
```

## Critical Dependencies

### Environment Setup
```bash
pip install -r requirements.txt  # Expected: torch, fairseq, transformers, sentencepiece, pydantic, pandas, numpy
# R packages: cvms, mgcv, lme4, dplyr (see experiments/dundee.r header)
```

### Data Access
All preprocessed data requires author permission (`kuribayashi.research [at] gmail.com`):
- `models/` (50GB): Trained LM checkpoints + SentencePiece models
- `surprisals/` (25GB): Precomputed scores (skip training if using)
- Raw corpora: Dundee eye-tracking annotations with dependency structure

## Research Concepts → Code Mapping

### Paper Equations → Implementation
| Concept | Equation | Code Location | Notes |
|---------|----------|---------------|-------|
| Lossy-context surprisal | Eq. 1: `I_lossy(w_i, c_<i) = -log p(w_i \| <s>◦f(c))` | `calc_surprisal.py` L59-94 | `batched_decode()` computes cross-entropy loss |
| N-gram noise | `f = last n-1 tokens` | `modify_context.py` L15-27 `_delete()` | Hard truncation at n-gram boundary |
| Probabilistic noise | Appendix B: `P(delete) = min(j*slope, 1)` | `modify_context.py` L31-57 `_delete_lossy()` | Linear decay with distance |
| Span aggregation | Eq. 2: Sum surprisals over subwords | `calc_surprisal.py` L88-91 | Word = multiple subwords |
| PPP calculation | ∆logLik / N | `dundee.r` L46-51 | Likelihood ratio test between models |
| Training modification | Appendix E: Random chunking with `<b>` | `run.sh` L1 (creates data-bin-ngram) | Only for Wiki-LMs, not GPT-2 |

### Figure Reproduction
| Figure | Script | Key Variables | Description |
|--------|--------|---------------|-------------|
| Figure 1 | `visualize.ipynb` | PPP vs input_length | Short-context advantage across LMs |
| Figure 2 | `visualize.ipynb` | ∆PPP vs model_size | Larger models benefit more from limiting context |
| Figure 3 | `visualize.ipynb` | Vanilla vs modified training | Training modification ablation |
| Figure 4a | `visualize.ipynb` + annotations | ELC vs dependency_locality | Syntactic locality effects |
| Figure 4b | `visualize.ipynb` + annotations | ELC vs dependency_type | Construction-specific effects |
| Table 4 | `aggregate.py` output | PPP by ngram/language | Main results table |

### Key Algorithm: Batched Surprisal Computation
`calc_surprisal.py` implements the core measurement loop:
```python
# For each article in corpus:
for article, pieces in article2piece.items():
    texts = []  # Full sequences
    target_spans = []  # Where to measure surprisal
    
    for context, target in pieces:
        # Apply noise function f (already done in modify_context.py)
        text = "<s>" + context + " " + target  # Or <b> if using -i flag
        span = (len(context.split()), len(context.split()) + len(target.split()) - 1)
        
    # Batch process with padding
    results = model(padded_inputs)  # Shape: (batch, seq_len, vocab)
    loss = CrossEntropyLoss(results, gold_ids)  # Per-position
    
    # Extract surprisals for target spans only
    surprisal = sum(loss[span[0]-1:span[1]])  # Note -1 offset!
```

**Critical detail**: The `-1` offset exists because gold IDs are shifted by 1 relative to inputs (standard LM target alignment).

## Code Conventions

### Surprisal Calculation Pattern
All `calc_surprisal*.py` scripts follow:
1. Load article→pieces JSON (context-target pairs)
2. Batch encode with padding (`torch.nn.utils.rnn.pad_sequence`)
3. Compute cross-entropy loss per token position
4. Sum surprisals within target spans (bunsetsu/word units)
5. Output: `scores.json` (article→surprisals) + `eval.txt` (PPL)

**Target span indexing**: Context split at position i creates span `[len(context.split()), len(context.split()) + len(target.split()) - 1]`. Output slices as `[span[0]-1:span[1]]` due to target-side index offset.

### File Naming Convention
Output directories encode experimental conditions:
```
surprisals/{BE|DC}/arch_{lstm|transformer|gpt}-ngram_{N}-contextfunc_{delete|lossy-SLOPE}/{seed-N}/
  ├── scores.json
  ├── scores.csv  (from convert_scores.py)
  ├── eval.txt
  └── likelihood.txt  (from R analysis)
```

### Statistical Analysis (R Scripts)
- `dundee.r`: Fit mixed-effects models with eye-tracking DV (gaze duration)
- Baseline model (Eq. 4): `time ~ freq * length + freq_prev_1 * length_prev_1 + screenN + lineN + segmentN + (1|article) + (1|subj)`
- Test model: Add `surprisals_sum` + `surprisals_sum_prev_{1,2}` predictors
- Compare via likelihood ratio test: PPP = (logLik_test - logLik_baseline) / N
- Output: `likelihood.txt` with `base_logLik` and `sup_logLik` values

### BOS Token Handling (Critical!)
Per README "Implementation Notes":
- **When context is empty** (i=0): Prepend `<s>` (beginning-of-sentence)
- **When context exists**: Use `</s> <i>` with `--add-itag/-i` flag for Wiki-LMs
- **Why?** Prevents confusing model with mid-sentence BOS tokens
- **Exception**: HuggingFace GPT-2s use `<|endoftext|>` instead of `<s>`

Code example from `calc_surprisal.py`:
```python
if args.add_bos:
    if args.add_itag:
        context = "</s> <i> " + context  # Non-empty context
    else:
        context = "</s> " + context
else:
    if context:
        if args.add_itag:
            context = "</s> <i> " + context
    else:
        context = "</s>"  # Empty context
```

## Common Gotchas

1. **Shell script syntax errors**: `en_surprisal.sh` has extra `do` on line 9 (typo)
2. **GPU memory**: Default batch size 50 may OOM on smaller GPUs; adjust `--batchsize`
3. **Path dependencies**: Scripts assume execution from project root; don't `cd` into subdirs
4. **Corpus-specific args**: Always set `--corpus dundee` matching data path
5. **Seed aggregation**: Full results average across 3 seeds (`seed-1`, `seed-39`, `seed-46`)
6. **Perplexity paradox**: Lower PPL doesn't always mean higher PPP (Figure 5). Different context constraints can yield same PPL but different human-likeness.

## Key Research Insights (What the Paper Found)

### Main Finding: 2-gram Models Are Most Human-Like
- **Counterintuitive result**: LMs with only bigram context (last 1 word) better predicted human reading times than full-context models
- **Effect size**: PPP improvement from full→2-gram: ∆=0.0-1.8 in English (Table 4)
- **Statistical significance**: 2-gram < full-context residual errors (p<0.001) for large models

### Scaling Trend (Figure 2)
- **Larger models = larger gap**: GPT2-xl (1558M) shows ∆=1.8 improvement, while LSTM-xs (27M) shows ∆=0.1
- **Interpretation**: Modern large LMs deviate MORE from human-like context access
- **Implication**: Model size alone doesn't guarantee cognitive plausibility

### Training Modification Impact (Figure 3)
- Vanilla LMs (standard training) slightly underestimate short-context advantage
- Modified training with `<b>` tokens increases 2-gram PPP by ~0.1-0.3
- **Why it matters**: Proves training-inference mismatch biases results toward longer context

### Syntactic Dependency Effects (Section 6, Figure 4)
Analyzed when long vs short context helps using **ELC score** = residual_2gram - residual_full:
- **Dependency locality**: Surprisingly, long dependencies DON'T always favor long context
- **Dependency type**: Construction-specific patterns (e.g., discourse relations in English favor long context, but most don't)
- **Key insight**: Human context access is construction-specific, not just distance-based

### Exclusion Criteria (Appendix C)
Eye-tracking data filtered to remove:
- Zero gaze duration or >3 standard deviations
- Punctuation/numeric tokens (English only)
- First/last tokens in line (controls for visual effects)
- Result: 212,649 English data points

## Testing Strategy
No formal unit tests. Validate changes via:
- Smoke test: Run single surprisal calculation with small n-gram (faster)
- Check `eval.txt` PPL is reasonable (not NaN/Inf)
- Verify `scores.csv` row count matches original annotation file count
- Compare log-likelihood values in R output to expected ranges
- Cross-check PPP trends match paper figures (short context ≥ long context)
