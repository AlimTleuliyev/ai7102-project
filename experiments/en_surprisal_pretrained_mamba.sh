#!/bin/bash
# Pretrained Mamba model experiments for English Dundee corpus
# Model: Local fine-tuned mamba-130m-local checkpoint
# This is a fine-tuned version of state-spaces/mamba-130m-hf
# This script runs the FULL pipeline: surprisal → convert → statistical analysis

set -e  # Exit on error

# Batch size (adjust if you run into OOM)
batch=80

# Local checkpoint configuration
checkpoint_path="mamba-130m-local"
model_name="pretrained-mamba-130m"

echo "🐍 Pretrained Mamba Model Experiments - Full Pipeline"
echo "==========================================="
echo "Checkpoint path: ${checkpoint_path}"
echo "Model identifier: ${model_name}"
echo "Total experiments: 31 (6 n-grams × 5 context functions + 1 n-gram-1000 delete)"
echo ""

echo "-------------------------------------------"
echo "Starting experiments for: ${model_name}"
echo "-------------------------------------------"


# Special case: ngram_1000 only has delete variant
ngram="1000"
func="delete"
result_dir="surprisals/DC-pretrained-mamba/arch_${model_name}-ngram_${ngram}-contextfunc_${func}"

echo "=========================================="
echo "📊 Processing: ngram=${ngram}, func=${func}"
echo "=========================================="

# Step 1: Calculate surprisal
echo "📊 Step 1/3: Calculate surprisal..."
python experiments/calc_surprisal_hf.py \
    -m "${model_name}" \
    -c "${checkpoint_path}" \
    -o "${result_dir}" \
    --batchsize ${batch} \
    -d "data/DC/ngram_${ngram}-contextfunc_${func}.json"

if [ $? -eq 0 ]; then
    echo "✅ Surprisal calculation complete!"
else
    echo "❌ Surprisal calculation failed!"
    exit 1
fi

# Step 2: Convert to CSV
echo "📊 Step 2/3: Convert scores to CSV format..."
python experiments/convert_scores.py --dir ${result_dir}

if [ $? -eq 0 ]; then
    echo "✅ Conversion complete!"
else
    echo "❌ Conversion failed!"
    exit 1
fi

# Step 3: Statistical analysis
echo "📊 Step 3/3: Run mixed-effects regression (calculate PPP)..."
python experiments/dundee.py ${result_dir}/

if [ $? -eq 0 ]; then
    echo "✅ Statistical analysis complete!"
else
    echo "❌ Statistical analysis failed!"
    exit 1
fi

echo "✅ Completed: ${model_name} - ngram_${ngram} - ${func}"
echo ""


for ngram in "20" "10" "7" "5" "3" "2"
do
    for func in "lossy-0.0625" "lossy-0.125" "lossy-0.25" "lossy-0.5" "delete"
    do
        result_dir="surprisals/DC-pretrained-mamba/arch_${model_name}-ngram_${ngram}-contextfunc_${func}"
    
    echo "=========================================="
    echo "📊 Processing: ngram=${ngram}, func=${func}"
    echo "=========================================="
    
    # Step 1: Calculate surprisal
        echo "📊 Step 1/3: Calculate surprisal..."
        python experiments/calc_surprisal_hf.py \
            -m "${model_name}" \
            -c "${checkpoint_path}" \
            -o "${result_dir}" \
            --batchsize ${batch} \
            -d "data/DC/ngram_${ngram}-contextfunc_${func}.json"
    
    if [ $? -eq 0 ]; then
        echo "✅ Surprisal calculation complete!"
    else
        echo "❌ Surprisal calculation failed!"
        exit 1
    fi
    
    # Step 2: Convert to CSV
    echo "📊 Step 2/3: Convert scores to CSV format..."
    python experiments/convert_scores.py --dir ${result_dir}
    
    if [ $? -eq 0 ]; then
        echo "✅ Conversion complete!"
    else
        echo "❌ Conversion failed!"
        exit 1
    fi
    
    # Step 3: Statistical analysis
    echo "📊 Step 3/3: Run mixed-effects regression (calculate PPP)..."
    python experiments/dundee.py ${result_dir}/
    
    if [ $? -eq 0 ]; then
        echo "✅ Statistical analysis complete!"
    else
        echo "❌ Statistical analysis failed!"
        exit 1
    fi
    
        echo "✅ Completed: ${model_name} - ngram_${ngram} - ${func}"
        echo ""
    done
done


echo "✅ Finished all experiments for model: ${model_name}"
echo ""

echo ""
echo "🎉 SUCCESS! All pretrained Mamba experiments complete!"
echo ""
echo "📊 Results saved to: surprisals/DC-pretrained-mamba/"
echo "   Each experiment directory contains:"
echo "   - scores.json (surprisal values)"
echo "   - scores.csv (merged with eye-tracking data)"
echo "   - likelihood.txt (PPP value)"
echo ""
