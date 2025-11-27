# Context Limitation Experiments - Mamba & Pythia

Modified version of [context_limitation_cognitive_modeling](https://github.com/kuribayashi4/context_limitation_cognitive_modeling) focusing on Mamba and Pythia model experiments with the Dundee eye-tracking corpus.

## Original Paper

**Context Limitations Make Neural Language Models More Human-Like** (EMNLP 2022)
Paper: https://aclanthology.org/2022.emnlp-main.712/

```bibtex
@inproceedings{kuribayashi-etal-2022-context,
    title = "Context Limitations Make Neural Language Models More Human-Like",
    author = "Kuribayashi, Tatsuki  and
      Oseki, Yohei  and
      Brassard, Ana  and
      Inui, Kentaro",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.712",
    pages = "10421--10436",
}
```

**Key Finding:** Language models with limited context (even 2-gram!) better predict human reading times than full-context models, measured by **Psychometric Predictive Power (PPP)** - how well surprisal values predict gaze duration.

## Modifications & Additions

### Modified from Original
- **[experiments/calc_surprisal_hf.py](experiments/calc_surprisal_hf.py)** - Added local checkpoint loading support for custom trained models
- **[experiments/dundee.py](experiments/dundee.py)** - Python implementation of statistical analysis (alternative to original `dundee.r`)

### Added to Original
- **Experiment scripts:**
  - [experiments/en_surprisal_mamba.sh](experiments/en_surprisal_mamba.sh) - Mamba model experiments (130M-2.8B parameters)
  - [experiments/en_surprisal_pretrained_mamba.sh](experiments/en_surprisal_pretrained_mamba.sh) - Fine-tuned Mamba experiments
  - [experiments/en_surprisal_pythia.sh](experiments/en_surprisal_pythia.sh) - Pythia model experiments (70M-6.9B parameters)
- **Tutorial notebooks** ([notebooks/01-06](notebooks/)) - Step-by-step walkthrough of the entire pipeline
- **Visualization notebooks:**
  - [plots_v1.ipynb](plots_v1.ipynb) - Main result visualizations
  - [plots_v2.ipynb](plots_v2.ipynb) - Alternative visualizations
- **Documentation:** [RUN_WITH_YOUR_MODEL.md](RUN_WITH_YOUR_MODEL.md) - Guide for using any HuggingFace model

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `transformers`, `torch`, `matplotlib`, `seaborn`

### 2. Download Preprocessed Data
```bash
chmod +x prepare_data.sh
./prepare_data.sh
```

This downloads (~133 MB):
- `data/DC/` - Dundee eye-tracking corpus with 31 context-modified variants
- Preprocessed by original paper authors

**Note:** If download fails, contact: `kuribayashi.research [at] gmail.com`

## Running Experiments

### Mamba Models
```bash
./experiments/en_surprisal_mamba.sh
```
Runs experiments on HuggingFace Mamba models (state-spaces/mamba-*).

### Pretrained Mamba (Fine-tuned)
1. **Download weights:** https://drive.google.com/drive/folders/1YnaOhntB0CVQSTTJvRrka0-1QdmNtiNS?usp=sharing
2. Place checkpoint in `mamba-130m-local/` directory
3. Run:
```bash
./experiments/en_surprisal_pretrained_mamba.sh
```

### Pythia Models
```bash
./experiments/en_surprisal_pythia.sh
```
Runs experiments on EleutherAI Pythia models.

### Experiment Pipeline

Each script executes three steps for each context configuration:

1. **Compute surprisal** (`calc_surprisal_hf.py`) - Calculate word-level surprisal values
2. **Merge data** (`convert_scores.py`) - Combine surprisals with eye-tracking measurements
3. **Statistical analysis** (`dundee.py`) - Mixed-effects regression to calculate PPP

**Results location:** `surprisals/DC-{model}/arch_{model}-ngram_{N}-contextfunc_{func}/`

Each experiment directory contains:
- `scores.json` - Surprisal values per word
- `scores.csv` - Merged with eye-tracking data
- `likelihood.txt` - PPP (Psychometric Predictive Power) value

**PPP (Psychometric Predictive Power)** = Per-token log-likelihood difference between regression models with/without surprisal. Higher PPP means better prediction of human reading times.

## Visualization & Analysis

After running experiments, analyze results with:
- **[plots_v1.ipynb](plots_v1.ipynb)** - Main visualizations (PPP vs n-gram, PPP vs perplexity)
- **[plots_v2.ipynb](plots_v2.ipynb)** - Alternative visualizations
- **[notebooks/01-06](notebooks/)** - Tutorial notebooks explaining the pipeline step-by-step

## Project Structure

```
.
├── experiments/
│   ├── calc_surprisal_hf.py      # Compute surprisal (HuggingFace models)
│   ├── convert_scores.py         # Merge surprisal with eye-tracking data
│   ├── dundee.py                 # Statistical analysis (Python)
│   ├── en_surprisal_mamba.sh     # Mamba experiments
│   ├── en_surprisal_pretrained_mamba.sh  # Fine-tuned Mamba
│   └── en_surprisal_pythia.sh    # Pythia experiments
├── notebooks/                    # Tutorial notebooks (01-06)
├── preprocess/DC/                # Data preprocessing scripts
├── plots_v1.ipynb               # Main visualizations
├── plots_v2.ipynb               # Alternative visualizations
├── prepare_data.sh              # Data download script
└── requirements.txt             # Python dependencies
```

## Citation

If you use this code or data, please cite the original paper:

```bibtex
@inproceedings{kuribayashi-etal-2022-context,
    title = "Context Limitations Make Neural Language Models More Human-Like",
    author = "Kuribayashi, Tatsuki  and
      Oseki, Yohei  and
      Brassard, Ana  and
      Inui, Kentaro",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    url = "https://aclanthology.org/2022.emnlp-main.712",
}
```

## Credits

- **Original repository:** https://github.com/kuribayashi4/context_limitation_cognitive_modeling
- **Original authors:** Tatsuki Kuribayashi, Yohei Oseki, Ana Brassard, Kentaro Inui
- **Data preprocessing and corpus access:** Provided by original paper authors
- **This fork:** Extended with Mamba and Pythia experiments, Python statistical analysis, and tutorial notebooks
