#/storage/emulated/0/Vxt/Vxt/vera_xt/interfaces/cli_interface.py
#!/usr/bin/env python3
"""
CLI Interface - Clean, fast terminal interface for Vera_XT
Optimized for Termux environment
"""

import sys
import time
from typing import Dict, Any, Optional

class CLIInterface:
    def __init__(self, brain_instance):
        self.brain = brain_instance
        self.running = True
        self.current_mode = "normal"  # normal, security, debug
        
        # Command mapping
        self.commands = {
            'help': self.show_help,
            'quit': self.quit,
            'exit': self.quit,
            'status': self.show_status,
            'models': self.show_models,
            'memory': self.show_memory,
            'security': self.show_security,
            'load_model': self.load_model,
            'clear': self.clear_conversation,
            'debug': self.toggle_debug,
            'confirm': self.confirm_identity
        }
        
        print("🖥️  CLI Interface initialized")
        print("💡 Optimized for Termux - fast, clean, responsive")
    
    def run(self):
        """Main interface loop"""
        print("\n" + "="*60)
        print("🚀 Vera_XT - Your AI Partner is ready!")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("="*60)
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n>You> ").strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.lower() in self.commands:
                    self.commands[user_input.lower()]()
                elif user_input.lower().startswith('load_model '):
                    model_name = user_input[11:].strip()  # Remove 'load_model ' prefix
                    self.load_model(model_name)
                else:
                    # Process as regular input through security
                    self.process_regular_input(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Ctrl+C detected. Type 'quit' to exit safely.")
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 System continuing...")
    
    def process_regular_input(self, user_input: str):
        """Process regular user input with security"""
        # Security check
        if hasattr(self.brain, 'security_system') and self.brain.security_system:
            security_result = self.brain.security_system.process_query_with_security(
                user_input, 
                self.brain.analyze_context(user_input) if hasattr(self.brain, 'analyze_context') else {}
            )
            
            response_type = security_result['response_type']
            
            if response_type == 'security_challenge':
                print(f"\n🛡️  Security Challenge: {self.brain.security_system.security_challenge(user_input)}")
                return
            elif response_type == 'limited_response':
                response = self.brain.security_system.provide_limited_response(user_input)
                print(f"\n🤖 {response}")
                return
        else:
            # No security system yet, proceed normally
            pass
        
        # Process with brain
        print("\n[Processing...] ", end='', flush=True)
        start_time = time.time()
        
        response = self.brain.process_input(user_input)
        
        processing_time = time.time() - start_time
        print(f"\r🤖 {response}")
        
        if processing_time > 2:  # If took more than 2 seconds
            print(f"⏱️  Processed in {processing_time:.2f}s")
    
    def show_help(self):
        """Show available commands"""
        print("\n📚 Available Commands:")
        print("  help     - Show this help")
        print("  quit/exit - Exit Vera_XT")
        print("  status   - Show system status")
        print("  models   - Show available models")
        print("  memory   - Show memory status") 
        print("  security - Show security status")
        print("  load_model <name> - Load a model")
        print("  clear    - Clear current conversation")
        print("  debug    - Toggle debug mode")
        print("  confirm  - Confirm your identity")
        print("\n💡 Just type your message to chat with Vera_XT!")
    
    def quit(self):
        """Quit the interface"""
        print("\n👋 Thank you for using Vera_XT!")
        print("💡 Your AI Partner will remember this session.")
        self.running = False
    
    def show_status(self):
        """Show system status"""
        if hasattr(self.brain, 'get_brain_status'):
            status = self.brain.get_brain_status()
            print(f"\n🧠 Brain Status:")
            print(f"  Model Loaded: {status.get('model_loaded', 'Unknown')}")
            print(f"  Current Model: {status.get('current_model', 'None')}")
            print(f"  Memory Count: {status.get('short_term_memory_count', 0)}")
            print(f"  Adaptation Level: {status.get('adaptation_level', 0):.2f}")
            print(f"  Emotional Sensitivity: {status.get('emotional_sensitivity', 0):.2f}")
        else:
            print("\n🧠 Brain status not available")
    
    def show_models(self):
        """Show available models"""
        if hasattr(self.brain, 'get_available_models'):
            models = self.brain.get_available_models()
            if models:
                print(f"\n📦 Available Models ({len(models)}):")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
            else:
                print("\n📦 No models found in Models/ directory")
                print("💡 Place GGUF models in Vxt/Vxt/Models/ folder")
        else:
            print("\n📦 Model system not available")
    
    def show_memory(self):
        """Show memory status"""
        if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
            stats = self.brain.memory_manager.get_memory_stats()
            print(f"\n🧠 Memory Status:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}: {len(value) if isinstance(value, (dict, list)) else value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("\n🧠 Memory system not connected")
    
    def show_security(self):
        """Show security status"""
        if hasattr(self.brain, 'security_system') and self.brain.security_system:
            security_status = self.brain.security_system.get_security_status()
            print(f"\n🛡️  Security Status:")
            print(f"  Trust Level: {security_status['trust_level']:.2f}")
            print(f"  Total Accesses: {security_status['total_accesses']}")
            print(f"  Suspicious Activities: {security_status['suspicious_activities_count']}")
            print(f"  Known Topics: {len(security_status['known_user_patterns']['common_topics'])}")
        else:
            print("\n🛡️  Security system not connected")
    
    def load_model(self, model_name: str = None):
        """Load a model"""
        if model_name is None:
            print("\n💡 Usage: load_model <model_name>")
            self.show_models()
            return
        
        if hasattr(self.brain, 'load_model'):
            success = self.brain.load_model(model_name)
            if success:
                print(f"✅ Model '{model_name}' loaded successfully!")
            else:
                print(f"❌ Failed to load model '{model_name}'")
                print("💡 Check if the model exists in Models/ directory")
        else:
            print("\n❌ Model loading not available")
    
    def clear_conversation(self):
        """Clear current conversation"""
        if hasattr(self.brain, 'short_term_memory'):
            self.brain.short_term_memory.clear()
            print("\n🧹 Conversation cleared!")
        else:
            print("\n❌ Memory system not available")
    
    def toggle_debug(self):
        """Toggle debug mode"""
        if self.current_mode == "debug":
            self.current_mode = "normal"
            print("\n🐛 Debug mode: OFF")
        else:
            self.current_mode = "debug"
            print("\n🐛 Debug mode: ON")
            print("💡 Detailed processing information will be shown")
    
    def confirm_identity(self):
        """Confirm user identity"""
        if hasattr(self.brain, 'security_system') and self.brain.security_system:
            self.brain.security_system.confirm_identity()
            print("\n✅ Identity confirmed!")
            print("💡 Full access restored.")
        else:
            print("\n🛡️  Security system not available")
