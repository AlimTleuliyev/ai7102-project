#!/bin/bash
# CORRECTED Quick Test - Fixed the issues!

set -e  # Exit on error

# Activate conda environment

echo "🚀 Quick Test: GPT-2 with 2-gram context"
echo "==========================================="
echo ""

# Step 1: Calculate surprisal
echo "📊 Step 1: Calculate surprisal..."
python experiments/calc_surprisal_hf.py \
  -m gpt2 \
  -o surprisals/quick-test/arch_gpt2-ngram_2-contextfunc_delete \
  --batchsize 10 \
  -d data/DC/ngram_2-contextfunc_delete.json

if [ $? -eq 0 ]; then
  echo "✅ Surprisal calculation complete!"
else
  echo "❌ Surprisal calculation failed!"
  exit 1
fi

echo ""

# Step 2: Convert to CSV
echo "📊 Step 2: Convert scores to CSV format..."
python experiments/convert_scores.py --dir surprisals/quick-test

if [ $? -eq 0 ]; then
  echo "✅ Conversion complete!"
else
  echo "❌ Conversion failed!"
  exit 1
fi

echo ""

# Step 3: Statistical analysis
echo "📊 Step 3: Run mixed-effects regression (calculate PPP)..."
python experiments/dundee.py surprisals/quick-test/

if [ $? -eq 0 ]; then
  echo "✅ Statistical analysis complete!"
else
  echo "❌ Statistical analysis failed!"
  exit 1
fi

echo ""
echo "🎉 SUCCESS! All steps completed."
echo ""
echo "📁 Results location:"
echo "   surprisals/quick-test/arch_gpt2-ngram_2-contextfunc_delete/"
echo ""
echo "📊 View your results:"
echo "   - Perplexity:  cat surprisals/quick-test/arch_gpt2-ngram_2-contextfunc_delete/eval.txt"
echo "   - PPP scores:  cat surprisals/quick-test/arch_gpt2-ngram_2-contextfunc_delete/likelihood.txt"
echo ""
