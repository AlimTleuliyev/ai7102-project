# 📚 Interactive Pipeline Walkthrough

## 🎯 Goal
Understand the ENTIRE pipeline step-by-step through interactive Jupyter notebooks.

## 📋 The 6-Part Journey

### ✅ Part 1: Understanding the Raw Data (START HERE!)
**File**: `notebooks/01_understanding_raw_data.ipynb`

**What you'll learn**:
- What is the Dundee eye-tracking corpus?
- What does "gaze duration" mean?
- See real sentences people read
- Understand the 515,010 observations
- Why word length/frequency affects reading time

**Time**: ~10-15 minutes

**Prerequisites**: None - start here!

---

### 🔜 Part 2: Context Modification (The Core Idea)
**File**: `notebooks/02_context_modification.ipynb` (will create after you finish Part 1)

**What you'll learn**:
- WHY do we modify context?
- See 2-gram vs full context examples
- Understand the 31 JSON files
- Visualize delete vs lossy functions
- The training-inference mismatch problem

---

### 🔜 Part 3: What is Surprisal?
**File**: `notebooks/03_surprisal_calculation.ipynb`

**What you'll learn**:
- What is surprisal? (information theory)
- How language models calculate it
- Run tiny example with 1 sentence
- Why sum over subwords?
- Surprisal vs probability

---

### 🔜 Part 4: Running the Model (Inference)
**File**: `notebooks/04_model_inference.ipynb`

**What you'll learn**:
- Load GPT-2 model
- Feed modified contexts
- Get surprisal scores
- Understand scores.json structure
- What eval.txt PPL means

---

### 🔜 Part 5: Statistical Analysis (dundee.py)
**File**: `notebooks/05_statistical_analysis.ipynb`

**What you'll learn**:
- What is mixed-effects regression?
- How surprisal predicts reading times
- Understand PPP metric
- Interpret likelihood.txt output
- Why we need random effects

---

### 🔜 Part 6: Putting It All Together
**File**: `notebooks/06_full_pipeline.ipynb`

**What you'll learn**:
- Run complete workflow on small example
- Compare 2-gram vs full context
- Visualize results
- Understand the paper's main finding
- How to use your own models

---

## 🚀 How to Use This Guide

### 1. Open the first notebook:
```bash
# From project root
cd notebooks
jupyter notebook 01_understanding_raw_data.ipynb
```

Or in VS Code: Just open the file and click "Run Cell"

### 2. Read the markdown cells (explanations)

### 3. Run each code cell and observe the output

### 4. Answer the checkpoint questions at the end

### 5. Provide feedback:
- ✅ "I understand, ready for next part"
- ❓ "I'm confused about [specific thing]"
- 🔄 "Can you explain [concept] differently?"

### 6. Only move to next notebook when you're ready!

---

## 📝 Current Progress

- ✅ **Part 1: Understanding the Raw Data** - READY TO RUN!
- ⏳ Part 2: Context Modification - Waiting for your feedback
- ⏳ Part 3: What is Surprisal? - Waiting
- ⏳ Part 4: Running the Model - Waiting
- ⏳ Part 5: Statistical Analysis - Waiting
- ⏳ Part 6: Full Pipeline - Waiting

---

## 💡 Tips

1. **Take your time** - Don't rush through
2. **Run every cell** - See the actual output
3. **Modify code** - Change numbers, try different things
4. **Ask questions** - If confused, ask before moving on
5. **Save checkpoints** - Note down what you learned

---

## 🆘 Getting Help

If something doesn't work:
1. Check you're in the conda environment: `conda activate dl`
2. Make sure you're in project root: `cd /path/to/dl-project`
3. Check file paths are correct
4. Ask me to fix it!

---

## 📊 Expected Time Per Part

- Part 1: 10-15 min
- Part 2: 15-20 min
- Part 3: 15-20 min
- Part 4: 20-25 min
- Part 5: 20-25 min
- Part 6: 30-40 min

**Total: ~2 hours** (but take breaks between parts!)

---

## ✅ Let's Start!

Open `notebooks/01_understanding_raw_data.ipynb` and run through it. Come back when you're done and let me know:
- What you learned
- What was confusing (if anything)
- If you're ready for Part 2

**Remember**: No question is too simple! The goal is for you to TRULY understand every step. 🎓
