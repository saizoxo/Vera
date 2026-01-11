#!/bin/bash
# Vera_XT Production Startup Script
# Generated on 2025-12-01 15:34:52

echo "🚀 Starting Vera_XT Production Server..."

# Start llama-server with production configuration
llama-server \
    -m Models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
    --port 8080 \
    --host 0.0.0.0 \
    --ctx-size 4096 \
    --threads 4 \
    --parallel 4 \
    --batch-size 512 \
    --ubatch-size 512 \
    --n-gpu-layers -1 \
    --temp 0.7 \
    --top-p 0.9 \
    --cont-batching \
    --embedding \
    --api-key YOUR_API_KEY_IF_NEEDED

echo "✅ Vera_XT Server running on http://0.0.0.0:8080"
echo "🔗 OpenAI-compatible API available at /v1/"
