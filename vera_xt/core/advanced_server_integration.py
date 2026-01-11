#/storage/emulated/0/Vxt/Vxt/vera_xt/core/advanced_server_integration.py
#!/usr/bin/env python3
"""
Advanced Server Integration - Leverages latest llama.cpp server features
Supports multimodal, OpenAI API, and advanced inference capabilities
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime

class AdvancedServerIntegration:
    def __init__(self, config_manager):
        self.config = config_manager
        self.models_dir = Path("Models")
        self.memory_handler = None
        self.server_process = None
        self.server_url = f"http://localhost:{config_manager.get_int('SERVER_PORT', 8080)}"
        
        # Advanced features
        self.multimodal_enabled = False
        self.speculative_decoding_enabled = False
        self.openai_compatible = True
        
        # Performance tracking
        self.performance_log = []
        
        print("🚀 Advanced Server Integration initialized")
        print(f"🌐 Server URL: {self.server_url}")
        print("💡 Supports multimodal, OpenAI API, speculative decoding")
    
    def set_memory_handler(self, memory_handler):
        """Connect memory handler for enhanced training"""
        self.memory_handler = memory_handler
        print("✅ Memory handler connected to server integration")
    
    def start_advanced_server(self, model_path: str, 
                            enable_multimodal: bool = False,
                            enable_speculative: bool = False,
                            draft_model_path: str = None,
                            enable_embeddings: bool = False) -> bool:
        """Start llama-server with advanced features"""
        try:
            cmd = [
                "llama-server",
                "-m", str(model_path),
                "--port", str(self.config.get_int('SERVER_PORT', 8080)),
                "--host", self.config.get('SERVER_HOST', '127.0.0.1'),
                "--ctx-size", "4096",  # Context size
                "--threads", "4",      # CPU threads
                "--n-gpu-layers", "-1",  # Use GPU if available
                "--parallel", "1",     # Parallel processing
                "--cont-batching"      # Continuous batching
            ]
            
            # Add multimodal support
            if enable_multimodal:
                # This would require a multimodal projector model
                # cmd.extend(["--mmproj", "path/to/mmproj-model-f16.gguf"])
                self.multimodal_enabled = True
                print("🖼️  Multimodal support prepared")
            
            # Add speculative decoding
            if enable_speculative and draft_model_path:
                cmd.extend(["-md", draft_model_path])
                self.speculative_decoding_enabled = True
                print("⚡ Speculative decoding enabled")
            
            # Add embeddings support
            if enable_embeddings:
                cmd.extend(["--embedding", "--pooling", "cls"])
                print("📊 Embeddings support enabled")
            
            # Add performance parameters
            cmd.extend([
                "--temp", str(self.config.get_float('TEMPERATURE', 0.7)),
                "--top-p", str(self.config.get_float('TOP_P', 0.9)),
                "--batch-size", "512",
                "--ubatch-size", "512"
            ])
            
            print(f"🔧 Starting advanced server...")
            
            # Start the server process
            self.server_process = subprocess.Popen(cmd)
            
            # Wait for server to start
            time.sleep(5)
            
            # Test connection
            if self._test_server_health():
                print("✅ Advanced server started successfully!")
                return True
            else:
                print("❌ Server started but health check failed")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start advanced server: {e}")
            return False
    
    def _test_server_health(self) -> bool:
        """Test server health endpoint"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def generate_with_openai_compatibility(self, user_input: str, 
                                         image_data: Optional[str] = None,
                                         use_embeddings: bool = False) -> str:
        """Generate response using OpenAI-compatible API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ]
            
            # If multimodal and image data provided
            if self.multimodal_enabled and image_data:
                messages[-1]["content"] = [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            
            payload = {
                "model": "vera-xt-local",
                "messages": messages,
                "max_tokens": self.config.get_int('MAX_TOKENS', 512),
                "temperature": self.config.get_float('TEMPERATURE', 0.7),
                "top_p": self.config.get_float('TOP_P', 0.9),
                "stream": False
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Log performance
                self._log_performance("openai_api", len(user_input), len(content), time.time())
                
                return content
            else:
                print(f"❌ API request failed: {response.status_code}")
                return "Sorry, I couldn't generate a response right now."
                
        except Exception as e:
            print(f"❌ OpenAI API request failed: {e}")
            return "Sorry, there was an error processing your request."
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Get embeddings using server endpoint"""
        if not self._test_server_health():
            return None
        
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "vera-xt-local",
                "input": text
            }
            
            response = requests.post(
                f"{self.server_url}/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['data'][0]['embedding']
            else:
                return None
                
        except Exception as e:
            print(f"❌ Embeddings request failed: {e}")
            return None
    
    def _log_performance(self, method: str, input_len: int, output_len: int, timestamp: float):
        """Log performance metrics"""
        self.performance_log.append({
            "timestamp": timestamp,
            "method": method,
            "input_length": input_len,
            "output_length": output_len,
            "datetime": datetime.now().isoformat()
        })
        
        # Keep only recent logs
        if len(self.performance_log) > 100:
            self.performance_log = self.performance_log[-50:]
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.performance_log:
            return {"status": "No performance data collected yet"}
        
        total_requests = len(self.performance_log)
        avg_input_len = sum(log["input_length"] for log in self.performance_log) / total_requests
        avg_output_len = sum(log["output_length"] for log in self.performance_log) / total_requests
        
        return {
            "total_requests": total_requests,
            "average_input_length": round(avg_input_len, 2),
            "average_output_length": round(avg_output_len, 2),
            "methods_used": list(set(log["method"] for log in self.performance_log)),
            "recent_logs": self.performance_log[-5:]  # Last 5 logs
        }
    
    def get_server_capabilities(self) -> Dict[str, Any]:
        """Get current server capabilities"""
        return {
            "multimodal_enabled": self.multimodal_enabled,
            "speculative_decoding": self.speculative_decoding_enabled,
            "openai_compatible": self.openai_compatible,
            "server_responding": self._test_server_health(),
            "config": {
                "port": self.config.get_int('SERVER_PORT', 8080),
                "temperature": self.config.get_float('TEMPERATURE', 0.7),
                "max_tokens": self.config.get_int('MAX_TOKENS', 512)
            }
        }
    
    def stop_server(self):
        """Stop the llama server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            print("⏹️  Advanced server stopped")

class EnhancedTrainingSystem:
    """Enhanced training using server capabilities"""
    
    def __init__(self, server_integration, memory_handler):
        self.server_integration = server_integration
        self.memory_handler = memory_handler
        self.training_sessions = []
        
        print("🎓 Enhanced Training System initialized")
        print("💡 Leverages server capabilities for advanced training")
    
    def generate_training_data_with_server(self, topic: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Generate training data using server for enhanced quality"""
        training_data = []
        
        # Create diverse training prompts
        base_prompts = [
            f"Explain {topic} in detail with examples.",
            f"What are the key concepts related to {topic}?",
            f"Provide practical applications of {topic}.",
            f"How does {topic} connect to other related concepts?",
            f"What are the advanced aspects of {topic}?"
        ]
        
        for i, prompt in enumerate(base_prompts[:num_samples]):
            # Generate response using server
            response = self.server_integration.generate_with_openai_compatibility(prompt)
            
            if response and "Sorry" not in response:  # Check if response is valid
                training_entry = {
                    "id": f"server_train_{int(time.time())}_{i}",
                    "topic": topic,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": time.time(),
                    "quality_score": self._assess_response_quality(prompt, response),
                    "semantic_tags": self._extract_semantic_tags(response)
                }
                
                training_data.append(training_entry)
                
                # Store in memory with high importance
                self.memory_handler.add_to_memory(
                    "server_training_data",
                    f"Prompt: {prompt}\nResponse: {response}",
                    category="knowledge",
                    context={
                        "importance": 8,
                        "training_data": True,
                        "topic": topic,
                        "quality_score": training_entry["quality_score"],
                        "semantic_tags": training_entry["semantic_tags"]
                    }
                )
        
        session_info = {
            "topic": topic,
            "timestamp": time.time(),
            "samples_generated": len(training_data),
            "training_data": training_data
        }
        
        self.training_sessions.append(session_info)
        
        return training_data
    
    def _assess_response_quality(self, prompt: str, response: str) -> float:
        """Assess quality of generated response"""
        # More sophisticated quality assessment
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Relevance: how many prompt words appear in response
        relevance = len(prompt_words & response_words) / max(len(prompt_words), 1)
        
        # Coherence: response length and structure
        response_length = len(response.split())
        coherence = min(response_length / 100, 1.0)  # Normalize to 0-1
        
        # Quality score
        quality = (relevance * 0.6) + (coherence * 0.4)
        return min(quality, 1.0)
    
    def _extract_semantic_tags(self, response: str) -> List[str]:
        """Extract semantic tags from response"""
        tags = []
        
        # Technical terms
        technical_terms = ["python", "code", "function", "algorithm", "data", "model", "system"]
        for term in technical_terms:
            if term in response.lower():
                tags.append(f"technical:{term}")
        
        # Educational terms
        education_terms = ["example", "explain", "understand", "concept", "principle"]
        for term in education_terms:
            if term in response.lower():
                tags.append(f"educational:{term}")
        
        # Add unique keywords
        words = response.lower().split()
        keywords = [word for word in words if len(word) > 4 and word.isalpha()][:10]
        for keyword in keywords:
            tags.append(f"keyword:{keyword}")
        
        return list(set(tags))  # Remove duplicates
    
    def get_training_insights(self) -> Dict[str, Any]:
        """Get insights about training effectiveness"""
        if not self.training_sessions:
            return {"status": "No training sessions yet"}
        
        total_samples = sum(session["samples_generated"] for session in self.training_sessions)
        recent_topics = [session["topic"] for session in self.training_sessions[-5:]]
        
        return {
            "total_training_sessions": len(self.training_sessions),
            "total_samples_generated": total_samples,
            "recent_topics": recent_topics,
            "server_integration_active": self.server_integration._test_server_health()
        }
