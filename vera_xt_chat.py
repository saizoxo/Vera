#/storage/emulated/0/Vxt/Vxt/vera_xt_chat.py
#!/usr/bin/env python3
"""
Vera_XT Chat Interface - Real conversational CLI experience
Uses the live production server API for real-time chat
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, List

class VeraXTChat:
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.conversation_history = []
        self.session_active = False
        
        print("💬 VERA_XT CHAT INTERFACE INITIALIZED")
        print(f"🔗 Connected to server: {server_url}")
    
    def check_server_health(self) -> bool:
        """Check if the production server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_conversation(self):
        """Start a new chat session"""
        if not self.check_server_health():
            print("❌ Server is not running!")
            print("💡 Please start the production server first:")
            print("   python production_server.py")
            return False
        
        print("\n" + "="*60)
        print("💬 WELCOME TO VERA_XT CHAT")
        print("✨ Your AI partner is ready for conversation!")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("="*60)
        
        self.session_active = True
        
        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": "You are Vera_XT, a helpful AI assistant. Engage in natural conversation.",
            "timestamp": time.time()
        })
        
        return True
    
    def display_help(self):
        """Display available commands"""
        print("\n📋 AVAILABLE COMMANDS:")
        print("   'quit' or 'exit' - End the conversation")
        print("   'clear' - Clear conversation history")
        print("   'history' - Show conversation history")
        print("   'model' - Show current model info")
        print("   'help' - Show this help message")
        print("   'new' - Start a new conversation")
        print("   'save' - Save conversation to file")
        print("\n💬 Just type your message to chat!")
    
    def get_model_info(self):
        """Get current model information"""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    model = data['data'][0]
                    print(f"🤖 Current model: {model['id']}")
                    return model['id']
        except:
            pass
        print("🤖 Model: Unknown (server connection issue)")
        return "Unknown"
    
    def save_conversation(self):
        """Save current conversation to file"""
        if not self.conversation_history:
            print("💬 No conversation to save")
            return
        
        # Create chat log directory
        log_dir = Path("Chat_Logs")
        log_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = log_dir / f"chat_log_{timestamp}.json"
        
        # Save conversation
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.time(),
                "server_url": self.server_url,
                "messages": self.conversation_history
            }, f, indent=2)
        
        print(f"💾 Conversation saved to: {filename}")
    
    def clear_conversation(self):
        """Clear current conversation history"""
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are Vera_XT, a helpful AI assistant. Engage in natural conversation.",
                "timestamp": time.time()
            }
        ]
        print("🧹 Conversation history cleared")
    
    def show_history(self):
        """Show recent conversation history"""
        if len(self.conversation_history) <= 1:  # Skip system message
            print("💬 No conversation history yet")
            return
        
        print("\n📖 RECENT CONVERSATION:")
        for msg in self.conversation_history[1:]:  # Skip system message
            role = msg['role'].upper()
            content = msg['content']
            print(f"   {role}: {content}")
    
    def chat_with_model(self, user_message: str) -> str:
        """Send message to model and get response"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": time.time()
            })
            
            # Prepare API request
            payload = {
                "model": "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",  # Current model
                "messages": self.conversation_history[-10:],  # Use last 10 messages for context
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False  # For now, non-streaming
            }
            
            # Send request to server
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": time.time()
                })
                
                return assistant_response
            else:
                error_msg = f"Server error: {response.status_code}"
                print(f"❌ {error_msg}")
                return error_msg
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def run_chat_loop(self):
        """Main chat loop"""
        if not self.start_conversation():
            return
        
        print(f"\n🤖 Vera_XT: Hello! I'm your AI partner. How can I help you today?")
        
        while self.session_active:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Vera_XT: Goodbye! It was great chatting with you! 👋")
                    self.session_active = False
                    break
                
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                
                elif user_input.lower() == 'model':
                    self.get_model_info()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_conversation()
                    print("🤖 Vera_XT: I've cleared our conversation. What would you like to talk about?")
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                elif user_input.lower() == 'new':
                    self.clear_conversation()
                    print("🤖 Vera_XT: Starting a new conversation! What's on your mind?")
                    continue
                
                # Process regular message
                print("🤖 Vera_XT: [Thinking...]", end='', flush=True)
                
                # Get response from model
                response = self.chat_with_model(user_input)
                
                # Clear the thinking indicator
                print("\r", end='')
                
                # Display response
                print(f"🤖 Vera_XT: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Vera_XT: Goodbye! 👋")
                self.session_active = False
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def start_vera_xt_chat():
    """Start the Vera_XT chat interface"""
    print("🚀 STARTING VERA_XT CHAT INTERFACE")
    print("=" * 50)
    
    chat = VeraXTChat()
    chat.run_chat_loop()
    
    print("\n🎉 CHAT SESSION COMPLETED!")
    print("💡 Vera_XT is ready for more conversations!")

if __name__ == "__main__":
    start_vera_xt_chat()
