#!/usr/bin/env bash
set -e

echo "==========================================================================="
echo "⚡ BitMC-SSM: Environment Setup & Build"
echo "==========================================================================="

# Check C++ compiler
if ! command -v g++ &> /dev/null; then
    echo "❌ g++ compiler not found. Please install build-essential or g++."
    exit 1
fi

# Build C++ inference engine
echo "🔨 Building C++20 Zero-GEMM Inference Engine..."
make clean
make -j$(nproc 2>/dev/null || echo 2)

echo "🐍 Setting up Python environment..."
pip install -r requirements.txt

echo "==========================================================================="
echo "✅ Setup completed successfully!"
echo "   Run inference with: ./infer model_medium-30M.bin 60"
echo "==========================================================================="
