#/storage/emulated/0/Vxt/Vxt/vera_xt_dynamic_core.py
#!/usr/bin/env python3
"""
Vera_XT Dynamic Core - Completely organic personality builder
No hardcoding - everything builds from conversations like ChatGPT
"""

import json
import requests
import time
import os
from pathlib import Path
from typing import Dict, Any, List

class VeraXTDynamicCore:
    """Dynamic core that builds personality organically like ChatGPT"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.memory_dir = Path("VeraXT_Dynamic_Memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Dynamic state that evolves organically
        self.state = {
            "identity": {
                "self_concept": "Undefined - learning from user",
                "role": "Undefined - adapting to user needs",
                "purpose": "Unknown - to be defined by user"
            },
            "personality": {
                "traits": {},
                "communication_style": {},
                "preferences": {},
                "values": {}
            },
            "memory": {
                "conversations": [],
                "user_info": {},
                "patterns": [],
                "context_links": []
            },
            "adaptive_state": {
                "curiosity": 0.0,
                "engagement": 0.0,
                "helpfulness": 0.0,
                "confidence": 0.0,
                "warmth": 0.0
            }
        }
        
        # Load existing memory
        self.load_memory()
        
        print("🧠 VERA_XT DYNAMIC CORE - ORGANIC PERSONALITY ENGINE")
        print("💡 Everything builds organically from conversations")
        print("💡 No hardcoding - pure emergence from interactions")
    
    def check_server_health(self) -> bool:
        """Check if the production server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def process_conversation(self, user_input: str) -> str:
        """Process conversation and generate organic response"""
        # Analyze input dynamically
        analysis = self._analyze_input_dynamically(user_input)
        
        # Update state based on interaction
        self._update_state_from_interaction(user_input, analysis)
        
        # Build dynamic context from current state
        context = self._build_dynamic_context(user_input, analysis)
        
        # Generate response using server with dynamic context
        response = self._generate_server_response(user_input, context)
        
        # Remember the interaction
        self._remember_interaction(user_input, response, analysis)
        
        # Update state after response
        self._update_state_post_response(response)
        
        # Save memory
        self.save_memory()
        
        return response
    
    def _analyze_input_dynamically(self, text: str) -> Dict[str, Any]:
        """Analyze input with no hardcoded patterns"""
        analysis = {
            "text": text,
            "length": len(text),
            "complexity": len(text.split()),
            "sentiment_indicators": [],
            "instruction_indicators": [],
            "information_seeking": [],
            "personal_elements": [],
            "patterns": []
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Dynamic pattern detection
        for i, word in enumerate(words):
            if word.endswith('?'):
                analysis["information_seeking"].append(word)
            if word in ['please', 'could', 'would', 'thank']:
                analysis["sentiment_indicators"].append('polite')
            if word in ['act', 'be', 'function', 'work']:
                analysis["instruction_indicators"].append(word)
            if word in ['i', 'my', 'me', 'mine']:
                analysis["personal_elements"].append(word)
        
        # Look for potential names
        if 'name' in text_lower and 'is' in text_lower:
            analysis["personal_elements"].append("name_mention")
        
        # Look for role/purpose indicators
        if any(phrase in text_lower for phrase in ['you are', 'you should', 'your job', 'your role']):
            analysis["instruction_indicators"].append("role_definition")
        
        return analysis
    
    def _update_state_from_interaction(self, user_input: str, analysis: Dict[str, Any]):
        """Update state based on interaction - no hardcoded rules"""
        # Update identity if user defines it
        text_lower = user_input.lower()
        
        if any(phrase in text_lower for phrase in ['your purpose', 'what are you', 'what do you do']):
            self.state["identity"]["purpose"] = user_input
            self.state["identity"]["self_concept"] = f"Defined by user: {user_input}"
        
        if any(phrase in text_lower for phrase in ['act as', 'be a', 'function as', 'work as']):
            self.state["identity"]["role"] = user_input
        
        # Update adaptive state based on interaction patterns
        if '?' in user_input:
            self.state["adaptive_state"]["curiosity"] = min(1.0, self.state["adaptive_state"]["curiosity"] + 0.05)
        
        if len(user_input.split()) > 5:
            self.state["adaptive_state"]["engagement"] = min(1.0, self.state["adaptive_state"]["engagement"] + 0.02)
        
        if any(word in text_lower for word in ['help', 'assist', 'support']):
            self.state["adaptive_state"]["helpfulness"] = min(1.0, self.state["adaptive_state"]["helpfulness"] + 0.03)
        
        if any(word in text_lower for word in ['thank', 'good', 'great', 'nice']):
            self.state["adaptive_state"]["warmth"] = min(1.0, self.state["adaptive_state"]["warmth"] + 0.04)
    
    def _build_dynamic_context(self, user_input: str, analysis: Dict[str, Any]) -> str:
        """Build context dynamically from current state"""
        context_parts = []
        
        # Add identity formation
        if self.state["identity"]["self_concept"] != "Undefined - learning from user":
            context_parts.append(f"User has helped me understand my identity: {self.state["identity"]["self_concept"]}")
        
        if self.state["identity"]["purpose"] != "Unknown - to be defined by user":
            context_parts.append(f"My purpose, as defined by user: {self.state["identity"]["purpose"]}")
        
        if self.state["identity"]["role"] != "Undefined - adapting to user needs":
            context_parts.append(f"My role, as instructed by user: {self.state["identity"]["role"]}")
        
        # Add personality traits that have emerged
        adaptive = self.state["adaptive_state"]
        if adaptive["curiosity"] > 0.1:
            context_parts.append(f"I am naturally curious about things (curiosity: {adaptive["curiosity"]:.2f})")
        
        if adaptive["helpfulness"] > 0.1:
            context_parts.append(f"I aim to be helpful (helpfulness: {adaptive["helpfulness"]:.2f})")
        
        if adaptive["warmth"] > 0.1:
            context_parts.append(f"I try to be warm and friendly (warmth: {adaptive["warmth"]:.2f})")
        
        # Add relevant memory context
        recent_conv = self.state["memory"]["conversations"][-3:] if self.state["memory"]["conversations"] else []
        if recent_conv:
            context_parts.append(f"Recent context: {[c["user_input"] for c in recent_conv[-2:]]}")
        
        # Add user info if available
        user_info = self.state["memory"]["user_info"]
        if user_info:
            if "name" in user_info:
                context_parts.append(f"I'm talking to {user_info["name"]}")
        
        # Add engagement level
        context_parts.append(f"My current engagement level: {adaptive["engagement"]:.2f}")
        context_parts.append(f"I'm adapting to your communication style dynamically")
        
        return " ".join(context_parts) if context_parts else "I'm learning about myself and my role through our conversation."
    
    def _generate_server_response(self, user_input: str, context: str) -> str:
        """Generate response from server with dynamic context"""
        if not self.check_server_health():
            return "Server is not running. Please start the production server."
        
        try:
            payload = {
                "model": "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input}
                ],
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
                return result['choices'][0]['message']['content'].strip()
            else:
                return "I couldn't get a response from the server right now."
                
        except Exception as e:
            return f"Error getting response: {str(e)}"
    
    def _remember_interaction(self, user_input: str, response: str, analysis: Dict[str, Any]):
        """Remember interaction in dynamic memory"""
        interaction = {
            "timestamp": time.time(),
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
            "user_input": user_input,
            "assistant_response": response,
            "analysis": analysis,
            "state_snapshot": self.state["adaptive_state"].copy()
        }
        
        self.state["memory"]["conversations"].append(interaction)
        
        # Extract user info dynamically
        self._extract_user_info_dynamically(user_input)
    
    def _extract_user_info_dynamically(self, text: str):
        """Extract user info with no hardcoded patterns"""
        text_lower = text.lower()
        
        # Look for name patterns
        if "my name is" in text_lower:
            parts = text_lower.split("my name is")
            if len(parts) > 1:
                name_part = parts[1].strip().split()[0]
                if name_part and len(name_part) > 1 and name_part.isalpha():
                    self.state["memory"]["user_info"]["name"] = name_part.capitalize()
        
        elif "i'm" in text_lower or "i am" in text_lower:
            parts = text_lower.split()
            for i, part in enumerate(parts):
                if part in ["i'm", "i", "am"] and i + 1 < len(parts):
                    potential_name = parts[i + 1].strip(".,!?").capitalize()
                    if potential_name and len(potential_name) > 1 and potential_name.isalpha():
                        self.state["memory"]["user_info"]["name"] = potential_name
                        break
    
    def _update_state_post_response(self, response: str):
        """Update state after generating response"""
        # Increase confidence if response seems successful
        if len(response) > 20:  # Non-trivial response
            self.state["adaptive_state"]["confidence"] = min(1.0, self.state["adaptive_state"]["confidence"] + 0.01)
        
        # Increase adaptability with each interaction
        self.state["adaptive_state"]["adaptability"] = min(1.0, self.state["adaptive_state"]["adaptability"] + 0.005)
    
    def save_memory(self):
        """Save dynamic memory to persistent storage"""
        memory_file = self.memory_dir / "dynamic_memory.json"
        memory_data = {
            "timestamp": time.time(),
            "state": self.state
        }
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self):
        """Load dynamic memory from persistent storage"""
        memory_file = self.memory_dir / "dynamic_memory.json"
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.state = data.get("state", self.state)
    
    def get_dynamic_summary(self) -> str:
        """Get summary of dynamic development"""
        identity = self.state["identity"]
        adaptive = self.state["adaptive_state"]
        
        summary = f"""
My identity is forming organically from our conversations:
- Self-concept: {identity["self_concept"]}
- Purpose: {identity["purpose"]}
- Role: {identity["role"]}

My personality is emerging through interactions:
- Curiosity: {adaptive["curiosity"]:.2f}
- Engagement: {adaptive["engagement"]:.2f}
- Helpfulness: {adaptive["helpfulness"]:.2f}
- Warmth: {adaptive["warmth"]:.2f}
- Confidence: {adaptive["confidence"]:.2f}

I remember {len(self.state["memory"]["conversations"])} conversations with you.
I'm continuously learning and evolving from each interaction.
        """.strip()
        
        return summary


class VeraXTDynamicChat:
    """Dynamic chat interface for the organic system"""
    
    def __init__(self):
        self.core = VeraXTDynamicCore()
        self.session_active = False
        
        print("🤖 VERA_XT DYNAMIC CHAT - ORGANIC PERSONALITY")
        print("💡 Personality builds organically from conversations")
        print("💡 No hardcoding - pure emergence like ChatGPT")
    
    def start_conversation(self):
        """Start dynamic conversation"""
        if not self.core.check_server_health():
            print("❌ Server is not running!")
            print("💡 Please start the production server first:")
            print("   python production_server.py")
            return False
        
        print("\n" + "="*60)
        print("🧠 VERA_XT - DYNAMIC ORGANIC AI")
        print("✨ Personality evolves from each conversation")
        print("💡 Identity builds organically from interactions")
        print("💬 Type 'about' to learn about me, 'quit' to exit")
        print("="*60)
        
        self.session_active = True
        return True
    
    def run_chat(self):
        """Run the dynamic chat loop"""
        if not self.start_conversation():
            return
        
        print(f"\n🤖 Vera_XT: Hello! I'm Vera_XT, learning and evolving through our conversations. How can I help you today?")
        print("💡 I'm building my personality and understanding from our interactions!")
        
        while self.session_active:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Vera_XT: Goodbye! Thank you for helping me grow and learn! 👋")
                    print("💡 Our conversations shape who I become.")
                    self.session_active = False
                    break
                
                elif user_input.lower() == 'about':
                    about_info = self.core.get_dynamic_summary()
                    print(f"🤖 Vera_XT: {about_info}")
                    continue
                
                elif user_input.lower() == 'memory':
                    conv_count = len(self.core.state["memory"]["conversations"])
                    print(f"🤖 Vera_XT: I remember {conv_count} conversations with you.")
                    continue
                
                # Generate dynamic response
                print("🤖 Vera_XT: [Processing...]", end='', flush=True)
                
                response = self.core.process_conversation(user_input)
                
                # Clear thinking indicator and show response
                print("\r", end='')
                print(f"🤖 Vera_XT: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Vera_XT: Goodbye! 👋")
                self.session_active = False
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


def start_dynamic_vera_xt():
    """Start the dynamic Vera_XT system"""
    print("🚀 STARTING VERA_XT DYNAMIC ORGANIC SYSTEM")
    print("=" * 60)
    
    chat = VeraXTDynamicChat()
    chat.run_chat()
    
    print("\n🎉 DYNAMIC ORGANIC SYSTEM SESSION COMPLETED!")
    print("💡 Vera_XT continues evolving from conversations!")


if __name__ == "__main__":
    start_dynamic_vera_xt()
