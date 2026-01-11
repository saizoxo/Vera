#/storage/emulated/0/Vxt/Vxt/vera_xt_enhanced.py
#!/usr/bin/env python3
"""
Enhanced Vera_XT - With Contact/Context Grabbing & Online API Training
Prepares for powerful local model integration
"""

import json
import requests
import time
import os
from pathlib import Path
from typing import Dict, Any, List

class EnhancedVeraXT:
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.conversation_history = []
        self.context_memory = {}  # For grabbing and storing context
        self.online_api_enabled = False
        self.api_client = None
        
        # Load configuration
        self.load_config()
        
        print("🚀 ENHANCED VERA_XT INITIALIZED")
        print(f"🔗 Local server: {server_url}")
        print(f"🌐 Online API training: {'ENABLED' if self.online_api_enabled else 'DISABLED'}")
    
    def load_config(self):
        """Load configuration from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
        
        # Check for OpenRouter API key
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if api_key and api_key != 'your_openrouter_api_key_here':
            self.online_api_enabled = True
            print("🌐 Online API access enabled")
        else:
            print("⚠️  Online API access disabled (no valid API key found)")
    
    def grab_context(self, text: str) -> Dict[str, Any]:
        """Grab and analyze context from input text"""
        context = {
            "entities": self._extract_entities(text),
            "keywords": self._extract_keywords(text),
            "sentiment": self._analyze_sentiment(text),
            "intent": self._analyze_intent(text),
            "personal_info": self._extract_personal_info(text),
            "context_tags": self._generate_context_tags(text)
        }
        
        # Store in context memory
        self.context_memory[time.time()] = context
        
        return context
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        entities = []
        text_lower = text.lower()
        
        # Simple entity extraction (would be enhanced with NLP in real system)
        personal_indicators = ["i", "me", "my", "myself", "we", "us", "our"]
        if any(indicator in text_lower.split() for indicator in personal_indicators):
            entities.append("personal_reference")
        
        # Time references
        time_words = ["today", "yesterday", "tomorrow", "week", "month", "year", "morning", "evening"]
        entities.extend([word for word in time_words if word in text_lower])
        
        # Technical terms
        tech_terms = ["python", "code", "ai", "ml", "programming", "model", "system"]
        entities.extend([term for term in tech_terms if term in text_lower])
        
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        words = text.lower().split()
        # Remove common stop words and get significant words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(keywords))[:10]  # Top 10 keywords
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment (simplified)"""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "excited"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "sad", "disappointed"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _analyze_intent(self, text: str) -> str:
        """Analyze user intent"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["what", "how", "why", "when", "where", "who"]):
            return "question"
        elif any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif any(word in text_lower for word in ["help", "assist", "support", "need"]):
            return "request_help"
        elif any(word in text_lower for word in ["code", "program", "python", "write"]):
            return "technical_task"
        elif any(word in text_lower for word in ["learn", "teach", "explain", "understand"]):
            return "learning"
        else:
            return "general"
    
    def _extract_personal_info(self, text: str) -> Dict[str, str]:
        """Extract personal information references"""
        personal_info = {}
        text_lower = text.lower()
        words = text_lower.split()
        
        # Look for name patterns
        if "i'm" in text_lower or "i am" in text_lower:
            # Extract potential name after "I'm" or "I am"
            parts = text_lower.split()
            for i, part in enumerate(parts):
                if part in ["i'm", "i", "am"] and i + 1 < len(parts):
                    potential_name = parts[i + 1].capitalize()
                    if potential_name.isalpha() and len(potential_name) > 2:
                        personal_info["name"] = potential_name
        
        # Look for other personal references
        if "my name is" in text_lower:
            parts = text_lower.split("my name is")
            if len(parts) > 1:
                name = parts[1].strip().split()[0].capitalize()
                if name.isalpha():
                    personal_info["name"] = name
        
        return personal_info
    
    def _generate_context_tags(self, text: str) -> List[str]:
        """Generate context tags for memory organization"""
        tags = []
        text_lower = text.lower()
        
        if "python" in text_lower or "code" in text_lower or "programming" in text_lower:
            tags.append("technical:programming")
        
        if "learn" in text_lower or "study" in text_lower or "education" in text_lower:
            tags.append("educational")
        
        if "personal" in text_lower or "private" in text_lower or "private" in text_lower:
            tags.append("personal")
        
        if self._analyze_sentiment(text) > 0.3:
            tags.append("positive")
        elif self._analyze_sentiment(text) < -0.3:
            tags.append("negative")
        
        return tags
    
    def train_with_online_api(self, user_input: str, local_response: str) -> str:
        """Train with online API and get improved response"""
        if not self.online_api_enabled:
            return local_response
        
        try:
            api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                return local_response
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "x-ai/grok-4.1-fast:free",
                "messages": [
                    {"role": "system", "content": "You are an educational AI assistant focused on comprehensive, structured responses."},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 512,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                api_response = result['choices'][0]['message']['content'].strip()
                
                # Store for training (in real system, this would be used to fine-tune local model)
                self._store_training_interaction(user_input, api_response, local_response)
                
                return api_response
            else:
                return local_response  # Fallback to local response
                
        except Exception as e:
            print(f"⚠️ API training failed: {e}")
            return local_response
    
    def _store_training_interaction(self, user_input: str, api_response: str, local_response: str):
        """Store interaction for future local model training"""
        training_data = {
            "timestamp": time.time(),
            "input": user_input,
            "api_response": api_response,
            "local_response": local_response,
            "quality_comparison": self._compare_response_quality(local_response, api_response)
        }
        
        # Save to training data directory
        training_dir = Path("Training_Data")
        training_dir.mkdir(exist_ok=True)
        
        training_file = training_dir / f"training_{int(time.time())}.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
    
    def _compare_response_quality(self, local_resp: str, api_resp: str) -> Dict[str, float]:
        """Compare quality of local vs API responses"""
        # Simple comparison metrics
        local_len = len(local_resp.split())
        api_len = len(api_resp.split())
        
        # Length comparison (API responses often more detailed)
        length_score = min(api_len / max(local_len, 1), 2.0)  # Cap at 2.0
        
        # This would be enhanced with more sophisticated metrics
        return {
            "length_advantage": length_score,
            "api_longer": api_len > local_len
        }
    
    def chat_with_context_grabbing(self, user_input: str) -> str:
        """Chat with context grabbing and optional API training"""
        # Grab context from user input
        context = self.grab_context(user_input)
        
        # Add to conversation history with context
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "context": context,
            "timestamp": time.time()
        })
        
        # Get response from local server
        try:
            payload = {
                "model": "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
                "messages": self.conversation_history[-10:],  # Use last 10 messages for context
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                local_response = result['choices'][0]['message']['content']
                
                # Train with online API if available
                final_response = self.train_with_online_api(user_input, local_response)
                
                # Add response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response,
                    "context": context,  # Same context for response
                    "timestamp": time.time()
                })
                
                return final_response
            else:
                return "Sorry, I couldn't get a response right now."
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return "Sorry, there was an error processing your request."
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from context memory"""
        if not self.context_memory:
            return {"status": "No context memory yet"}
        
        # Analyze memory patterns
        all_entities = []
        all_keywords = []
        all_intents = []
        
        for context in self.context_memory.values():
            all_entities.extend(context.get("entities", []))
            all_keywords.extend(context.get("keywords", []))
            all_intents.append(context.get("intent", "unknown"))
        
        return {
            "total_contexts": len(self.context_memory),
            "most_common_entities": self._get_top_items(all_entities),
            "most_common_intents": self._get_top_items(all_intents),
            "memory_size": len(self.context_memory)
        }
    
    def _get_top_items(self, items: List[str], top_n: int = 5) -> List[str]:
        """Get top N most common items"""
        if not items:
            return []
        
        item_counts = {}
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
        
        # Sort by count and return top N
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:top_n]]

class EnhancedChatInterface:
    def __init__(self):
        self.chat = EnhancedVeraXT()
        self.session_active = False
    
    def check_server_health(self) -> bool:
        """Check if the production server is running"""
        try:
            response = requests.get(f"{self.chat.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_conversation(self):
        """Start enhanced conversation with context grabbing"""
        if not self.check_server_health():
            print("❌ Server is not running!")
            print("💡 Please start the production server first:")
            print("   python production_server.py")
            return False
        
        print("\n" + "="*60)
        print("💬 ENHANCED VERA_XT CHAT")
        print("✨ Context-aware AI partner with memory capabilities!")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("🧠 Context grabbing: ON")
        print("🌐 Online training: " + ("ON" if self.chat.online_api_enabled else "OFF"))
        print("="*60)
        
        self.session_active = True
        return True
    
    def display_help(self):
        """Display enhanced commands"""
        print("\n📋 ENHANCED COMMANDS:")
        print("   'quit' or 'exit' - End the conversation")
        print("   'memory' - Show context memory insights")
        print("   'context' - Show current context")
        print("   'history' - Show conversation history")
        print("   'train' - Force online API training")
        print("   'help' - Show this help message")
        print("\n💬 Just type your message for context-aware responses!")
    
    def show_memory_insights(self):
        """Show memory insights"""
        insights = self.chat.get_memory_insights()
        print(f"\n🧠 MEMORY INSIGHTS:")
        for key, value in insights.items():
            print(f"   {key}: {value}")
    
    def show_current_context(self):
        """Show context from last interaction"""
        if not self.chat.conversation_history:
            print("💬 No conversation yet")
            return
        
        last_msg = self.chat.conversation_history[-1]
        context = last_msg.get('context', {})
        if context:
            print(f"\n🔍 CURRENT CONTEXT:")
            for key, value in context.items():
                print(f"   {key}: {value}")
        else:
            print("🔍 No context available for this message")
    
    def force_online_training(self, user_input: str = "Explain machine learning"):
        """Force online API training"""
        if not self.chat.online_api_enabled:
            print("🌐 Online API training not enabled (no API key)")
            return
        
        print("🔄 Forcing online API training...")
        response = self.chat.train_with_online_api(user_input, "Local response placeholder")
        print(f"🤖 API-trained response: {response[:100]}...")
    
    def run_enhanced_chat(self):
        """Run the enhanced chat loop"""
        if not self.start_conversation():
            return
        
        print(f"\n🤖 Enhanced Vera_XT: Hello! I can grab context and learn from online APIs. How can I help you today?")
        
        while self.session_active:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle enhanced commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Enhanced Vera_XT: Goodbye! I've learned from our conversation! 👋")
                    self.session_active = False
                    break
                
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                
                elif user_input.lower() == 'memory':
                    self.show_memory_insights()
                    continue
                
                elif user_input.lower() == 'context':
                    self.show_current_context()
                    continue
                
                elif user_input.lower() == 'history':
                    if len(self.chat.conversation_history) > 1:
                        print("\n📖 RECENT CONVERSATION:")
                        for msg in self.chat.conversation_history[1:]:  # Skip system
                            role = msg['role'].upper()
                            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                            print(f"   {role}: {content}")
                    else:
                        print("💬 No conversation history yet")
                    continue
                
                elif user_input.lower() == 'train':
                    self.force_online_training()
                    continue
                
                # Process regular message with context grabbing
                print("🤖 Enhanced Vera_XT: [Understanding context...]", end='', flush=True)
                
                response = self.chat.chat_with_context_grabbing(user_input)
                
                # Clear the thinking indicator
                print("\r", end='')
                
                print(f"🤖 Enhanced Vera_XT: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Enhanced Vera_XT: Goodbye! 👋")
                self.session_active = False
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def start_enhanced_chat():
    """Start the enhanced Vera_XT chat"""
    print("🚀 STARTING ENHANCED VERA_XT CHAT")
    print("=" * 50)
    
    chat = EnhancedChatInterface()
    chat.run_enhanced_chat()
    
    print("\n🎉 ENHANCED CHAT SESSION COMPLETED!")
    print("💡 Your AI partner now has context grabbing and online training!")

if __name__ == "__main__":
    start_enhanced_chat()
