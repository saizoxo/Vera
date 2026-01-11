#/storage/emulated/0/Vxt/Vxt/test_grok_integration.py
#!/usr/bin/env python3
"""
Test script for Grok API integration with Vera_XT
"""

import os
import requests
import json
from typing import Dict, Any

# Import config manager to load .env
from vera_xt.core.config_manager import config_manager

class GrokAPIClient:
    def __init__(self):
        # Get OpenRouter API key from environment variable (loaded from .env)
        self.api_key = config_manager.get('OPENROUTER_API_KEY')
        if not self.api_key or self.api_key == 'your_openrouter_api_key_here':
            raise ValueError("OPENROUTER_API_KEY not set in .env file")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "x-ai/grok-4.1-fast:free"
        self.max_tokens = config_manager.get_int('MAX_TOKENS', 512)
        self.temperature = config_manager.get_float('TEMPERATURE', 0.7)
        
        print("🌐 Grok API Client initialized")
        print(f"📦 Using model: {self.model}")
        print(f"⚙️  Max tokens: {self.max_tokens}, Temperature: {self.temperature}")
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using Grok API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return "Sorry, I couldn't get a response from the API right now."
        except Exception as e:
            print(f"❌ Error processing API response: {e}")
            return "Sorry, there was an error processing the response."

def setup_env_file():
    """Create .env file if it doesn't exist"""
    env_file = config_manager.env_file
    
    if not env_file.exists():
        print("📝 Creating .env file...")
        
        # Create a template .env file
        env_content = """# API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model Configuration
DEFAULT_MODEL=tinyllama-1.1b-chat-v1.0-q4_k_m.gguf

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8080

# Memory Configuration
MEMORY_DIR=Memory_Data
WORKSPACE_DIR=Workspace

# Performance Configuration
MAX_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print(f"✅ .env file created at {env_file}")
        print("💡 Please edit the .env file and add your OpenRouter API key")
    else:
        print(f"📋 .env file already exists at {env_file}")

def test_memory_system():
    """Test the new memory folder system"""
    print("\n🧪 Testing Memory Folder System...")
    
    from vera_xt.core.memory_handler import MemoryHandler
    
    memory = MemoryHandler()
    
    # Test adding memories to different categories
    print("📝 Adding test memories...")
    memory.add_to_memory("user_input", "Hello, how are you?", category="conversation")
    memory.add_to_memory("assistant_response", "I'm doing well, thank you!", category="conversation")
    memory.add_to_memory("knowledge", "Python is a programming language", category="knowledge")
    memory.add_to_memory("skill", "How to write functions", category="skill")
    
    print("✅ Memories added successfully!")
    
    # Check folder statistics
    stats = memory.get_memory_stats()
    print(f"📊 Memory stats: {stats}")
    
    # Test conversation creation
    conv_id = memory.create_conversation("testing")
    print(f"💬 New conversation created: {conv_id}")
    
    # Add to conversation
    memory.add_to_memory("user_input", "Testing conversation memory", category="conversation")
    
    print("✅ Memory folder system test completed!")

def test_grok_integration():
    """Test Grok API integration"""
    print("\n🧪 Testing Grok API Integration...")
    
    # Check if API key is set
    api_key = config_manager.get('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ OPENROUTER_API_KEY not set in .env file")
        print("💡 Please add your OpenRouter API key to the .env file")
        return None
    
    try:
        client = GrokAPIClient()
        
        print("Sending test message to Grok...")
        response = client.generate_response("Hello, who are you?")
        
        print(f"🤖 Grok response: {response[:100]}...")  # First 100 chars
        print("✅ Grok API test completed successfully!")
        
        return client
        
    except ValueError as e:
        print(f"❌ {e}")
        print("💡 Add your OpenRouter API key to the .env file to test API integration")
        return None

def test_consistency_with_api():
    """Test consistency between local and API brain"""
    print("\n🧪 Testing Consistency System...")
    
    # Initialize local brain
    from vera_xt.core.basic_brain import BasicBrain
    
    brain = BasicBrain()
    
    # Test without API (offline mode)
    print("Testing offline mode...")
    offline_response = brain.think_human_like("Hello")
    print(f"Offline response: {offline_response[:50]}...")
    
    # If API client is available, test with API
    api_client = test_grok_integration()
    if api_client:
        brain.set_api_client(api_client)
        print("Testing with API consistency...")
        consistent_response = brain.think_human_like("Hello")
        print(f"Consistent response: {consistent_response[:50]}...")
    
    print("✅ Consistency test completed!")

def test_config():
    """Test configuration system"""
    print("\n🧪 Testing Configuration System...")
    
    config_summary = config_manager.get_config_summary()
    print(f"📋 Configuration summary: {config_summary}")
    
    print("✅ Configuration test completed!")

if __name__ == "__main__":
    print("🚀 Vera_XT Testing Suite")
    print("=" * 50)
    
    # Setup .env file if needed
    setup_env_file()
    
    # Test configuration
    test_config()
    
    # Test memory system
    test_memory_system()
    
    # Test API integration
    test_grok_integration()
    
    # Test consistency
    test_consistency_with_api()
    
    print("\n✅ All tests completed!")
    print("💡 Remember to set your OpenRouter API key in the .env file")