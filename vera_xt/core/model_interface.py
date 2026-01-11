#/storage/emulated/0/Vxt/Vxt/vera_xt/core/model_interface.py
#!/usr/bin/env python3
"""
Model Interface Module - Handles model loading and inference
"""

from pathlib import Path
from typing import Callable, Optional

class ModelInterface:
    def __init__(self):
        self.current_model_path = None
        self.model_loaded = False
        self.model_interface = None  # Will be set dynamically
        self.llm = None  # Llama model instance
        self.models_dir = Path("Models")
    
    def set_model_interface(self, interface_func: Callable):
        """Set the model interface function (dynamic, not hardcoded)"""
        self.model_interface = interface_func
    
    def load_model(self, model_name: str) -> bool:
        """Load a model from Models/ directory"""
        model_path = self.models_dir / model_name
        if model_path.exists():
            self.current_model_path = model_path
            self.model_loaded = True
            print(f"✅ Model loaded: {model_name}")
            
            # Setup model interface if possible
            self.setup_model_interface()
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
    
    def setup_model_interface(self):
        """Setup the model interface using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            if self.current_model_path and self.model_loaded:
                # Initialize the model
                self.llm = Llama(
                    model_path=str(self.current_model_path),
                    n_ctx=2048,  # Context length
                    n_threads=4,  # Number of threads (adjust based on your CPU)
                    verbose=False  # Reduce logging
                )
                
                # Set the model interface function
                self.set_model_interface(self._llama_model_call)
                print(f"✅ Model interface connected to {self.current_model_path.name}")
                return True
            else:
                print("❌ No model loaded to connect interface to")
                return False
                
        except ImportError:
            print("❌ llama-cpp-python not available")
            print("💡 Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            print(f"❌ Error setting up model interface: {e}")
            return False

    def _llama_model_call(self, user_input: str) -> str:
        """Internal method to call the Llama model"""
        try:
            # Create a simple chat format
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ]
            
            # Generate response
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,  # Limit response length
                temperature=0.7,  # Creativity level
                top_p=0.9,      # Sampling threshold
                repeat_penalty=1.1  # Reduce repetition
            )
            
            # Extract the response text with safety checks
            if response and 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                if content:
                    return content.strip()
                else:
                    return "I received an empty response. Could you try asking again?"
            else:
                return "I couldn't generate a response. Please try again."
            
        except Exception as e:
            print(f"❌ Model inference error: {e}")
            return "Sorry, I encountered an issue processing your request."

    def get_available_models(self):
        """Get list of available models in Models/ directory"""
        if self.models_dir.exists():
            return [f.name for f in self.models_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.gguf', '.bin']]
        return []