#!/bin/bash
set -e  # Exit on error

echo "=================================================="
echo "Downloading preprocessed data (133 MB)..."
echo "=================================================="
echo ""

# Download with resume support and rate limiting
wget -c --show-progress --limit-rate=3m http://www.cl.ecei.tohoku.ac.jp/~kuribayashi/EMNLP2022/data.tar.gz

echo ""
echo "=================================================="
echo "Extracting data archive..."
echo "=================================================="
echo ""

# Extract with progress
tar -xzvf data.tar.gz

echo ""
echo "=================================================="
echo "Cleaning up..."
echo "=================================================="
rm data.tar.gz

echo ""
echo "✓ Data preparation complete!"
echo ""
echo "Downloaded and extracted:"
echo "  - data/DC/           (Dundee corpus with annotations)"
echo "  - data/en_sents/     (Training data for Wiki-LMs)"
echo ""