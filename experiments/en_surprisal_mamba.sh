#!/bin/bash
# Mamba model experiments for English Dundee corpus
# Model: state-spaces/mamba-130m-hf (130M parameters)
# Similar setup to en_surprisal_hf.sh but for Mamba architecture
# This script runs the FULL pipeline: surprisal → convert → statistical analysis

set -e  # Exit on error

# Batch size (adjust if you run into OOM for larger models)
batch=10

# Mamba models to run (added variants)
models=(
    "state-spaces/mamba-130m-hf"
    "state-spaces/mamba-370m-hf"
    "state-spaces/mamba-790m-hf"
    "state-spaces/mamba-1.4b-hf"
    "state-spaces/mamba-2.8b-hf"
)

echo "🐍 Mamba Model Experiments - Full Pipeline"
echo "==========================================="
echo "Models: ${models[*]}"
echo "Total experiments per model: 31 (6 n-grams × 5 context functions + 1 n-gram-1000 delete)"
echo ""

for model in "${models[@]}"
do
    arc=$(basename "${model}")

    echo "-------------------------------------------"
    echo "Starting experiments for model: ${model} (id: ${arc})"
    echo "-------------------------------------------"

    for ngram in "2" "3" "5" "7" "10" "20"
    do
        for func in "delete" "lossy-0.5" "lossy-0.25" "lossy-0.125" "lossy-0.0625"
        do
            result_dir="surprisals/DC-mamba/arch_${arc}-ngram_${ngram}-contextfunc_${func}"
        
        echo "=========================================="
        echo "📊 Processing: ngram=${ngram}, func=${func}"
        echo "=========================================="
        
        # Step 1: Calculate surprisal
            echo "📊 Step 1/3: Calculate surprisal..."
            python experiments/calc_surprisal_hf.py \
                -m "${model}" \
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
        
            echo "✅ Completed: ${arc} - ngram_${ngram} - ${func}"
            echo ""
        done
    done

    # Special case: ngram_1000 only has delete variant
    ngram="1000"
    func="delete"
    result_dir="surprisals/DC-mamba/arch_${arc}-ngram_${ngram}-contextfunc_${func}"
    
    echo "=========================================="
    echo "📊 Processing: ngram=${ngram}, func=${func}"
    echo "=========================================="
    
    # Step 1: Calculate surprisal
    echo "📊 Step 1/3: Calculate surprisal..."
    python experiments/calc_surprisal_hf.py \
        -m "${model}" \
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
    
    echo "✅ Completed: ${arc} - ngram_${ngram} - ${func}"
    echo ""

    echo "✅ Finished all experiments for model: ${arc}"
    echo ""
done

echo ""
echo "🎉 SUCCESS! All Mamba experiments complete!"
echo ""
echo "📊 Results saved to: surprisals/DC-mamba/"
echo "   Each experiment directory contains:"
echo "   - scores.json (surprisal values)"
echo "   - scores.csv (merged with eye-tracking data)"
echo "   - likelihood.txt (PPP value)"
echo ""
