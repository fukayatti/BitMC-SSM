#!/usr/bin/env bash
set -e

MODEL_PATH="${1:-model_medium-30M.bin}"
NUM_TOKENS="${2:-60}"

if [ ! -f "$MODEL_PATH" ]; then
    if [ -f "model.bin" ]; then
        MODEL_PATH="model.bin"
    else
        echo "❌ Model file not found: $MODEL_PATH"
        echo "   Please train or provide a valid .bin model file."
        exit 1
    fi
fi

if [ ! -f "./infer" ]; then
    echo "🔨 Compiling inference engine..."
    make
fi

echo "🚀 Launching BitMC-SSM Zero-GEMM Inference..."
./infer "$MODEL_PATH" "$NUM_TOKENS"
