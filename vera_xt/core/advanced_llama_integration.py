#/storage/emulated/0/Vxt/Vxt/vera_xt/core/advanced_llama_integration.py
#!/usr/bin/env python3
"""
Advanced Llama Integration - Leverages latest llama.cpp features
Supports multimodal, OpenAI-compatible API, and advanced inference
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

class AdvancedLlamaIntegration:
    def __init__(self, config_manager):
        self.config = config_manager
        self.models_dir = Path("Models")
        self.memory_handler = None  # Will be set by external system
        self.server_process = None
        self.server_url = f"http://localhost:{config_manager.get_int('SERVER_PORT', 8080)}"
        
        # Advanced features tracking
        self.multimodal_enabled = False
        self.speculative_decoding_enabled = False
        self.quantization_level = "Q4_K_M"  # Default quantization
        
        print("🚀 Advanced Llama Integration initialized")
        print(f"🌐 Server URL: {self.server_url}")
        print("💡 Supports multimodal, OpenAI API, and advanced inference")
    
    def set_memory_handler(self, memory_handler):
        """Set memory handler for integration"""
        self.memory_handler = memory_handler
        print("✅ Memory handler connected to advanced integration")
    
    def start_llama_server(self, model_path: str, enable_multimodal: bool = False, 
                          enable_speculative: bool = False, draft_model_path: str = None) -> bool:
        """Start llama-server with advanced features"""
        try:
            # Build server command with advanced features
            cmd = [
                "llama-server",
                "-m", str(model_path),
                "--port", str(self.config.get_int('SERVER_PORT', 8080)),
                "--host", self.config.get('SERVER_HOST', '127.0.0.1'),
                "--ctx-size", "4096",  # Context size
                "--threads", "4",      # CPU threads
                "--n-gpu-layers", "-1"  # Use GPU if available
            ]
            
            # Add multimodal support if requested
            if enable_multimodal:
                cmd.extend(["--mmproj", "path/to/mmproj-model-f16.gguf"])  # Placeholder
                self.multimodal_enabled = True
                print("🖼️  Multimodal support enabled")
            
            # Add speculative decoding if requested
            if enable_speculative and draft_model_path:
                cmd.extend(["-md", draft_model_path])
                self.speculative_decoding_enabled = True
                print("⚡ Speculative decoding enabled")
            
            # Add other advanced parameters
            cmd.extend([
                "--temp", str(self.config.get_float('TEMPERATURE', 0.7)),
                "--top-p", str(self.config.get_float('TOP_P', 0.9)),
                "--max-tokens", str(self.config.get_int('MAX_TOKENS', 512))
            ])
            
            print(f"🔧 Starting server with command: {' '.join(cmd)}")
            
            # Start the server process
            self.server_process = subprocess.Popen(cmd)
            
            # Wait a bit for server to start
            time.sleep(3)
            
            # Test if server is running
            if self._test_server_connection():
                print("✅ Llama server started successfully!")
                return True
            else:
                print("❌ Server started but not responding")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start llama server: {e}")
            return False
    
    def _test_server_connection(self) -> bool:
        """Test if the llama server is responding"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_with_openai_api(self, user_input: str, image_data: Optional[str] = None) -> str:
        """Generate response using OpenAI-compatible API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Prepare messages for chat completion
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ]
            
            # If multimodal is enabled and image data is provided
            if self.multimodal_enabled and image_data:
                # Add image to messages (this is a simplified approach)
                messages[-1]["content"] = [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            
            payload = {
                "model": "local-model",  # Placeholder for local model
                "messages": messages,
                "max_tokens": self.config.get_int('MAX_TOKENS', 512),
                "temperature": self.config.get_float('TEMPERATURE', 0.7),
                "top_p": self.config.get_float('TOP_P', 0.9)
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ API request failed with status {response.status_code}")
                return "Sorry, I couldn't generate a response right now."
                
        except Exception as e:
            print(f"❌ OpenAI API request failed: {e}")
            return "Sorry, there was an error processing your request."
    
    def quantize_model(self, input_model_path: str, output_model_path: str, 
                      quantization_type: str = "Q4_K_M") -> bool:
        """Quantize model using llama.cpp tools"""
        try:
            # Use llama-quantize tool (if available)
            cmd = [
                "llama-quantize",
                input_model_path,
                output_model_path,
                quantization_type
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Model quantized successfully: {quantization_type}")
                self.quantization_level = quantization_type
                return True
            else:
                print(f"❌ Quantization failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Quantization error: {e}")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and capabilities"""
        status = {
            "server_running": self.server_process is not None,
            "multimodal_enabled": self.multimodal_enabled,
            "speculative_decoding_enabled": self.speculative_decoding_enabled,
            "quantization_level": self.quantization_level,
            "server_url": self.server_url,
            "config": {
                "port": self.config.get_int('SERVER_PORT', 8080),
                "temperature": self.config.get_float('TEMPERATURE', 0.7),
                "max_tokens": self.config.get_int('MAX_TOKENS', 512)
            }
        }
        
        # Test server health if running
        if status["server_running"]:
            status["server_responding"] = self._test_server_connection()
        
        return status
    
    def benchmark_model(self, model_path: str) -> Dict[str, Any]:
        """Run benchmark on model to get performance metrics"""
        try:
            cmd = [
                "llama-bench",
                "-m", model_path,
                "-n", "128",  # Number of tokens to generate
                "-p", "128"   # Prompt size
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse benchmark results (simplified)
                lines = result.stdout.strip().split('\n')
                benchmark_data = {
                    "raw_output": result.stdout,
                    "success": True,
                    "model_path": model_path
                }
                return benchmark_data
            else:
                print(f"❌ Benchmark failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_server(self):
        """Stop the llama server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            print("⏹️  Llama server stopped")
    
    def get_available_quantization_types(self) -> List[str]:
        """Get list of available quantization types"""
        # Based on llama.cpp capabilities
        return [
            "Q4_0", "Q4_1", "Q5_0", "Q5_1", 
            "Q8_0", "Q2_K", "Q3_K_S", "Q3_K_M", 
            "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", 
            "Q5_K_M", "Q6_K", "Q8_K", "IQ2_XXS", 
            "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ2_S",
            "IQ2_M", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS"
        ]

class AdvancedLocalTrainer:
    """Advanced trainer that can leverage local llama.cpp server for enhanced training"""
    
    def __init__(self, llama_integration, memory_handler):
        self.llama_integration = llama_integration
        self.memory_handler = memory_handler
        self.training_sessions = []
        
        print("🎓 Advanced Local Trainer initialized")
        print("💡 Can leverage local server for enhanced training")
    
    def generate_training_data_locally(self, topic: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Generate training data using local model for enhanced learning"""
        training_data = []
        
        # Create training prompts focused on the topic
        training_prompts = [
            f"Explain the concept of {topic} in detail.",
            f"What are the key aspects of {topic}?",
            f"Provide examples of {topic} in practice.",
            f"How does {topic} relate to other concepts?",
            f"What are the applications of {topic}?"
        ]
        
        # Generate responses using local server
        for i, prompt in enumerate(training_prompts[:num_samples]):
            if self.llama_integration._test_server_connection():
                response = self.llama_integration.generate_with_openai_api(prompt)
                
                training_entry = {
                    "id": f"local_train_{int(time.time())}_{i}",
                    "topic": topic,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": time.time(),
                    "quality_score": self._assess_response_quality(prompt, response)
                }
                
                training_data.append(training_entry)
                
                # Store in memory for training
                self.memory_handler.add_to_memory(
                    "local_training_data",
                    f"Prompt: {prompt}\nResponse: {response}",
                    category="knowledge",
                    context={
                        "importance": 7,
                        "training_data": True,
                        "topic": topic
                    }
                )
        
        self.training_sessions.append({
            "topic": topic,
            "timestamp": time.time(),
            "samples_generated": len(training_data),
            "training_data": training_data
        })
        
        return training_data
    
    def _assess_response_quality(self, prompt: str, response: str) -> float:
        """Assess quality of generated response"""
        # Simple quality assessment
        response_length = len(response.split())
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        relevance = len(prompt_words & response_words) / max(len(prompt_words), 1)
        
        # Quality score based on length and relevance
        length_score = min(response_length / 100, 1.0)  # Normalize length
        quality = (length_score * 0.4) + (relevance * 0.6)
        
        return min(quality, 1.0)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about local training sessions"""
        if not self.training_sessions:
            return {"status": "No training sessions yet"}
        
        total_samples = sum(session["samples_generated"] for session in self.training_sessions)
        avg_quality = sum(
            sum(entry["quality_score"] for entry in session["training_data"]) / max(len(session["training_data"]), 1)
            for session in self.training_sessions
        ) / len(self.training_sessions)
        
        return {
            "total_training_sessions": len(self.training_sessions),
            "total_samples_generated": total_samples,
            "average_response_quality": round(avg_quality, 3),
            "recent_topics": [session["topic"] for session in self.training_sessions[-5:]]
        }
