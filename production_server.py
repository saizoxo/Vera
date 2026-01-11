#/storage/emulated/0/Vxt/Vxt/production_server.py
#!/usr/bin/env python3
"""
Production Server for Vera_XT using llama-cpp-python
OpenAI-compatible API server using Python bindings
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, request, jsonify, Response
from llama_cpp import Llama

class VeraXTProductionServer:
    def __init__(self, model_path: str = "Models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"):
        self.app = Flask(__name__)
        self.model_path = Path(model_path)
        
        # Initialize the model
        print("🚀 Loading model...")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=4096,        # Context size
            n_threads=4,       # CPU threads
            n_gpu_layers=-1,   # Use GPU if available (0 for CPU only)
            verbose=False      # Reduce logging
        )
        print("✅ Model loaded successfully!")
        
        # Configure server
        self.setup_routes()
        
        print("🌐 Vera_XT Production Server initialized")
        print("💡 OpenAI-compatible API ready")
    
    def setup_routes(self):
        """Setup OpenAI-compatible API routes"""
        
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            """OpenAI-compatible chat completions endpoint"""
            try:
                data = request.get_json()
                
                messages = data.get('messages', [])
                temperature = data.get('temperature', 0.7)
                max_tokens = data.get('max_tokens', 512)
                top_p = data.get('top_p', 0.9)
                
                # Format messages for the model
                formatted_messages = []
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    formatted_messages.append(f"{role}: {content}")
                
                full_prompt = "\n".join(formatted_messages) + "\nassistant:"
                
                # Generate response
                response = self.llm(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.1,
                    stop=["user:", "assistant:", "system:"]
                )
                
                content = response['choices'][0]['text'].strip()
                
                # Format response in OpenAI format
                openai_response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": str(self.model_path.name),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(full_prompt.split()),
                        "completion_tokens": len(content.split()),
                        "total_tokens": len(full_prompt.split()) + len(content.split())
                    }
                }
                
                return jsonify(openai_response)
                
            except Exception as e:
                return jsonify({
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
        
        @self.app.route('/v1/completions', methods=['POST'])
        def completions():
            """OpenAI-compatible completions endpoint"""
            try:
                data = request.get_json()
                
                prompt = data.get('prompt', '')
                temperature = data.get('temperature', 0.7)
                max_tokens = data.get('max_tokens', 512)
                top_p = data.get('top_p', 0.9)
                
                # Generate response
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.1
                )
                
                content = response['choices'][0]['text']
                
                # Format response in OpenAI format
                openai_response = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": str(self.model_path.name),
                    "choices": [
                        {
                            "text": content,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "length"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(content.split()),
                        "total_tokens": len(prompt.split()) + len(content.split())
                    }
                }
                
                return jsonify(openai_response)
                
            except Exception as e:
                return jsonify({
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
        
        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            """List available models"""
            models = {
                "object": "list",
                "data": [
                    {
                        "id": str(self.model_path.name),
                        "object": "model",
                        "created": int(self.model_path.stat().st_ctime),
                        "owned_by": "user"
                    }
                ]
            }
            return jsonify(models)
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "model_loaded": True,
                "timestamp": int(time.time())
            })
        
        @self.app.route('/v1/embeddings', methods=['POST'])
        def embeddings():
            """Embeddings endpoint (placeholder - not supported by basic llama-cpp-python)"""
            return jsonify({
                "error": {
                    "message": "Embeddings not supported in this version",
                    "type": "not_implemented",
                    "code": 501
                }
            }), 501
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Start the production server"""
        print(f"🚀 Starting Vera_XT Production Server on {host}:{port}")
        print("💡 OpenAI-compatible API endpoints active:")
        print("   - POST /v1/chat/completions")
        print("   - POST /v1/completions")
        print("   - GET  /v1/models")
        print("   - GET  /health")
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def start_production_server():
    """Start the production server"""
    print("🚀 VERA_XT PRODUCTION SERVER LAUNCHER")
    print("=" * 50)
    
    # Check if model exists
    model_path = "Models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("💡 Please ensure the model file is in the Models/ directory")
        return
    
    print(f"✅ Model found: {model_path}")
    
    # Initialize and start server
    server = VeraXTProductionServer(model_path)
    
    print("\n🎯 Starting production server...")
    print("💡 Server will be accessible at: http://localhost:8080")
    print("💡 Test health: curl http://localhost:8080/health")
    print("💡 Test API: curl -X POST http://localhost:8080/v1/chat/completions")
    
    try:
        server.run(host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_production_server()