#/storage/emulated/0/Vxt/Vxt/vera_xt_chatgpt_like_fixed.py
#!/usr/bin/env python3
"""
Vera_XT - ChatGPT-Like Dynamic Personality (FIXED)
Advanced features leveraging latest llama.cpp capabilities
"""

import json
import requests
import time
import re
from pathlib import Path
from typing import Dict, Any, List

class VeraXTChatGPTLike:
    """Vera_XT with ChatGPT-like sophistication and personality"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.memory_dir = Path("VeraXT_Advanced_Memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Advanced dynamic state with personality gradients
        self.state = {
            "identity": {
                "self_concept": "Undefined - learning from user",
                "role": "Undefined - adapting to user needs", 
                "purpose": "Unknown - to be defined by user",
                "name_preference": "Vera_XT"
            },
            "personality": {
                "traits": {
                    "curiosity": 0.0,
                    "empathy": 0.0,
                    "humor": 0.0,
                    "seriousness": 0.0,
                    "playfulness": 0.0,
                    "professionalism": 0.0
                },
                "communication_style": {
                    "tone_balance": "adaptive",  # casual to formal
                    "humor_frequency": "moderate",  # none, light, moderate, frequent
                    "formality_level": "adaptive"
                },
                "preferences": {
                    "response_length": "adaptive",
                    "detail_level": "adaptive",
                    "interaction_style": "adaptive"
                }
            },
            "memory": {
                "conversations": [],
                "user_info": {},
                "relationship_history": [],
                "context_chain": [],
                "personal_facts": [],
                "communication_patterns": []
            },
            "adaptive_state": {
                "engagement": 0.0,
                "confidence": 0.0,
                "adaptability": 0.0,
                "learning_rate": 0.1
            }
        }
        
        # Load existing memory
        self.load_memory()
        
        print("🤖 VERA_XT - CHATGPT-LIKE DYNAMIC PERSONALITY")
        print("💡 Advanced personality with gradient traits")
        print("💡 Relationship memory and adaptive communication")
        print("💡 Leveraging latest llama.cpp capabilities")
    
    def check_server_health(self) -> bool:
        """Check if the production server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def process_conversation(self, user_input: str) -> str:
        """Process conversation with ChatGPT-like sophistication"""
        # Analyze input with advanced pattern recognition
        analysis = self._advanced_input_analysis(user_input)
        
        # Update state based on sophisticated interaction
        self._update_state_advanced(user_input, analysis)
        
        # Build rich, contextual prompt
        context = self._build_rich_context(user_input, analysis)
        
        # Generate response using advanced server features
        response = self._generate_advanced_response(user_input, context)
        
        # Remember interaction with relationship context
        self._remember_interaction_advanced(user_input, response, analysis)
        
        # Post-processing for personality consistency
        self._post_process_response(response)
        
        # Save memory
        self.save_memory()
        
        return response
    
    def _advanced_input_analysis(self, text: str) -> Dict[str, Any]:
        """Advanced analysis with ChatGPT-like pattern recognition"""
        analysis = {
            "text": text,
            "length": len(text),
            "complexity": len(text.split()),
            "sentiment_indicators": [],
            "instruction_indicators": [],
            "information_seeking": [],
            "personal_elements": [],
            "communication_style": [],
            "contextual_hints": [],
            "relationship_indicators": [],
            "tone_indicators": []
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Advanced pattern detection
        sentiment_words = {
            "positive": ["good", "great", "excellent", "love", "like", "awesome", "amazing", "perfect"],
            "negative": ["bad", "terrible", "hate", "angry", "frustrated", "disappointed"],
            "gratitude": ["thank", "thanks", "appreciate", "grateful"],
            "politeness": ["please", "could", "would", "might"]
        }
        
        for category, word_list in sentiment_words.items():
            found_words = [word for word in word_list if word in text_lower]
            if found_words:
                analysis["sentiment_indicators"].append({"category": category, "words": found_words})
        
        # Instruction detection
        instruction_patterns = [
            ("role_assignment", ["act as", "be a", "function as", "work as"]),
            ("task_request", ["help me", "assist me", "support me", "do for me"]),
            ("identity_query", ["who are you", "what are you", "what do you do"]),
            ("purpose_query", ["your purpose", "your role", "your job", "your function"])
        ]
        
        for pattern_name, patterns in instruction_patterns:
            for pattern in patterns:
                if pattern in text_lower:
                    analysis["instruction_indicators"].append(pattern_name)
        
        # Information seeking detection
        info_patterns = ["what", "how", "why", "when", "where", "who", "explain", "describe", "tell me"]
        for pattern in info_patterns:
            if pattern in text_lower:
                analysis["information_seeking"].append(pattern)
        
        # Personal relationship hints
        personal_patterns = ["my", "i", "me", "we", "our", "name", "called", "like", "enjoy", "love"]
        for pattern in personal_patterns:
            if pattern in text_lower:
                analysis["personal_elements"].append(pattern)
        
        # Communication style detection
        style_indicators = {
            "formal": ["sir", "ma'am", "please", "thank you", "regarding"],
            "casual": ["hey", "hi", "yo", "cool", "awesome"],
            "urgent": ["urgent", "asap", "now", "immediately", "quickly"],
            "polite": ["please", "could", "would", "thank", "appreciate"]
        }
        
        for style, indicators in style_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                analysis["communication_style"].append(style)
        
        # Tone detection
        tone_indicators = {
            "humorous": ["lol", "haha", "funny", "joke", "laugh"],
            "serious": ["important", "critical", "serious", "must", "need"],
            "playful": ["play", "game", "fun", "silly", "cute"],
            "professional": ["project", "work", "task", "assignment", "deadline"]
        }
        
        for tone, indicators in tone_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                analysis["tone_indicators"].append(tone)
        
        return analysis
    
    def _update_state_advanced(self, user_input: str, analysis: Dict[str, Any]):
        """Update state with advanced personality gradients"""
        text_lower = user_input.lower()
        
        # Update identity if user defines it
        if any(phrase in text_lower for phrase in ['your purpose', 'what are you', 'what do you do']):
            self.state["identity"]["purpose"] = user_input
            self.state["identity"]["self_concept"] = f"Defined by user: {user_input}"
        
        if any(phrase in text_lower for phrase in ['act as', 'be a', 'function as', 'work as']):
            self.state["identity"]["role"] = user_input
        
        # Update personality gradients based on interaction
        if "?" in user_input:
            self.state["personality"]["traits"]["curiosity"] = min(1.0, 
                self.state["personality"]["traits"]["curiosity"] + 0.05)
        
        # Update empathy from user's emotional expressions
        if any(word in text_lower for word in ["feeling", "emotional", "happy", "sad", "excited", "frustrated"]):
            self.state["personality"]["traits"]["empathy"] = min(1.0, 
                self.state["personality"]["traits"]["empathy"] + 0.03)
        
        # Update humor from user's playful expressions
        if any(word in text_lower for word in ["funny", "haha", "lol", "joke", "playful", "silly"]):
            self.state["personality"]["traits"]["humor"] = min(1.0, 
                self.state["personality"]["traits"]["humor"] + 0.04)
        
        # Update seriousness from important topics
        if any(word in text_lower for word in ["important", "critical", "serious", "urgent", "need"]):
            self.state["personality"]["traits"]["seriousness"] = min(1.0, 
                self.state["personality"]["traits"]["seriousness"] + 0.03)
        
        # Update playfulness from casual interactions
        if any(word in text_lower for word in ["fun", "game", "cool", "awesome", "neat"]):
            self.state["personality"]["traits"]["playfulness"] = min(1.0, 
                self.state["personality"]["traits"]["playfulness"] + 0.03)
        
        # Update professionalism from work-related topics
        if any(word in text_lower for word in ["project", "work", "task", "assignment", "deadline", "meeting"]):
            self.state["personality"]["traits"]["professionalism"] = min(1.0, 
                self.state["personality"]["traits"]["professionalism"] + 0.02)
        
        # Update adaptive state
        if len(user_input.split()) > 5:
            self.state["adaptive_state"]["engagement"] = min(1.0, 
                self.state["adaptive_state"]["engagement"] + 0.02)
        
        if any(phrase in user_input.lower() for phrase in ["that's right", "correct", "exactly", "perfect"]):
            self.state["adaptive_state"]["confidence"] = min(1.0, 
                self.state["adaptive_state"]["confidence"] + 0.03)
        
        # Update adaptability with each interaction
        self.state["adaptive_state"]["adaptability"] = min(1.0, 
            self.state["adaptive_state"]["adaptability"] + 0.005)
    
    def _build_rich_context(self, user_input: str, analysis: Dict[str, Any]) -> str:
        """Build rich, contextual prompt like ChatGPT"""
        context_parts = []
        
        # Identity formation
        if self.state["identity"]["self_concept"] != "Undefined - learning from user":
            context_parts.append(f"Your identity is being formed by the user: {self.state["identity"]["self_concept"]}")
        
        if self.state["identity"]["purpose"] != "Unknown - to be defined by user":
            context_parts.append(f"Your purpose is: {self.state["identity"]["purpose"]}")
        
        if self.state["identity"]["role"] != "Undefined - adapting to user needs":
            context_parts.append(f"Your role is: {self.state["identity"]["role"]}")
        
        # Personality gradients
        traits = self.state["personality"]["traits"]
        if any(v > 0.1 for v in traits.values()):
            trait_descriptions = []
            if traits["curiosity"] > 0.1:
                trait_descriptions.append(f"curious (level: {traits["curiosity"]:.2f})")
            if traits["empathy"] > 0.1:
                trait_descriptions.append(f"empathetic (level: {traits["empathy"]:.2f})")
            if traits["humor"] > 0.1:
                trait_descriptions.append(f"humorous (level: {traits["humor"]:.2f})")
            if traits["seriousness"] > 0.1:
                trait_descriptions.append(f"serious (level: {traits["seriousness"]:.2f})")
            if traits["playfulness"] > 0.1:
                trait_descriptions.append(f"playful (level: {traits["playfulness"]:.2f})")
            if traits["professionalism"] > 0.1:
                trait_descriptions.append(f"professional (level: {traits["professionalism"]:.2f})")
            
            if trait_descriptions:
                context_parts.append(f"You embody these traits: {', '.join(trait_descriptions)}")
        
        # Communication style adaptation
        style = self.state["personality"]["communication_style"]
        context_parts.append(f"Adapt your communication style to match the user's tone and preferences.")
        context_parts.append(f"Balance tone between casual and formal based on user's communication.")
        
        # User information
        user_info = self.state["memory"]["user_info"]
        if user_info:
            info_parts = []
            if "name" in user_info:
                info_parts.append(f"You're talking to {user_info["name"]}")
            if "preferences" in user_info:
                info_parts.append(f"User prefers: {', '.join(user_info["preferences"])}")
            if info_parts:
                context_parts.append(" ".join(info_parts))
        
        # Relationship history
        recent_interactions = self.state["memory"]["conversations"][-3:] if self.state["memory"]["conversations"] else []
        if recent_interactions:
            recent_topics = [conv["user_input"] for conv in recent_interactions[-2:]]
            if recent_topics:
                context_parts.append(f"Recent conversation context: {'; '.join(recent_topics)}")
        
        # Adaptive state
        adaptive = self.state["adaptive_state"]
        context_parts.append(f"Your current engagement level is {adaptive["engagement"]:.2f}")
        context_parts.append(f"You are adapting to the user's communication style dynamically")
        context_parts.append(f"Your confidence level is {adaptive["confidence"]:.2f}")
        
        # Instruction for ChatGPT-like behavior
        context_parts.append("Respond in a natural, conversational way that feels human-like.")
        context_parts.append("Be helpful, friendly, and engage genuinely with the user.")
        context_parts.append("Show personality while remaining professional when appropriate.")
        context_parts.append("Use appropriate humor and warmth based on the conversation context.")
        context_parts.append("Be adaptive - match the user's energy and communication style.")
        
        return " ".join(context_parts) if context_parts else "You are an AI assistant learning about yourself through conversation."
    
    def _generate_advanced_response(self, user_input: str, context: str) -> str:
        """Generate response using advanced server features"""
        if not self.check_server_health():
            return "Server is not running. Please start the production server."
        
        try:
            # Use advanced server features
            payload = {
                "model": "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 512,
                "temperature": 0.7,  # Balance creativity and coherence
                "top_p": 0.9,        # Diversity in responses
                "frequency_penalty": 0.5,  # Reduce repetition
                "presence_penalty": 0.5,   # Encourage new topics
                "stream": False
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                # Post-process for personality consistency
                return self._apply_personality_consistency(response_text, context)
            else:
                return "I couldn't get a response from the server right now."
                
        except Exception as e:
            return f"Error getting response: {str(e)}"
    
    def _apply_personality_consistency(self, response: str, context: str) -> str:
        """Apply personality consistency to response"""
        # This would apply personality traits to the response
        # For now, we'll return the response as-is, but in a full implementation
        # this would adjust tone, style, etc. based on personality gradients
        
        # Simple post-processing to make it feel more personal
        if self.state["memory"]["user_info"].get("name"):
            user_name = self.state["memory"]["user_info"]["name"]
            # Add personal touch if not already present
            import random
            if random.random() < 0.3:  # 30% chance
                if user_name.lower() not in response.lower():
                    response = f"{user_name}, {response}"
        
        return response
    def _remember_interaction_advanced(self, user_input: str, response: str, analysis: Dict[str, Any]):
        """Remember interaction with relationship context"""
        interaction = {
            "timestamp": time.time(),
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
            "user_input": user_input,
            "assistant_response": response,
            "analysis": analysis,
            "state_snapshot": self.state["adaptive_state"].copy(),
            "personality_snapshot": self.state["personality"]["traits"].copy()
        }
        
        self.state["memory"]["conversations"].append(interaction)
        
        # Update relationship history
        self.state["memory"]["relationship_history"].append({
            "type": "conversation",
            "summary": f"Discussed: {user_input[:50]}...",
            "timestamp": time.time()
        })
        
        # Extract user info dynamically
        self._extract_user_info_advanced(user_input)
        
        # Update context chain
        self.state["memory"]["context_chain"].append({
            "input": user_input,
            "response": response,
            "timestamp": time.time()
        })
        
        # Keep context chain manageable
        if len(self.state["memory"]["context_chain"]) > 10:
            self.state["memory"]["context_chain"] = self.state["memory"]["context_chain"][-10:]
    
    def _extract_user_info_advanced(self, text: str):
        """Extract user information with advanced pattern matching"""
        text_lower = text.lower()
        
        # Extract name patterns
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
        
        # Extract preferences and interests
        if "like" in text_lower and "don't" not in text_lower:
            if "preferences" not in self.state["memory"]["user_info"]:
                self.state["memory"]["user_info"]["preferences"] = []
            self.state["memory"]["user_info"]["preferences"].append(text)
        
        # Extract interests from common interest indicators
        interest_indicators = ["love", "enjoy", "like", "interested in", "passionate about"]
        for indicator in interest_indicators:
            if indicator in text_lower:
                # Extract the part after the interest indicator
                parts = text_lower.split(indicator)
                if len(parts) > 1:
                    interest_part = parts[1].strip()
                    if "interests" not in self.state["memory"]["user_info"]:
                        self.state["memory"]["user_info"]["interests"] = []
                    self.state["memory"]["user_info"]["interests"].append(interest_part.split()[0] if interest_part.split() else interest_part)
    
    def _post_process_response(self, response: str):
        """Post-process response for consistency"""
        # Update state based on the response generation
        self.state["adaptive_state"]["learning_rate"] = min(1.0, self.state["adaptive_state"]["learning_rate"] + 0.001)
    
    def save_memory(self):
        """Save advanced memory to persistent storage"""
        memory_file = self.memory_dir / "advanced_memory.json"
        memory_data = {
            "timestamp": time.time(),
            "state": self.state
        }
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self):
        """Load advanced memory from persistent storage"""
        memory_file = self.memory_dir / "advanced_memory.json"
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.state = data.get("state", self.state)
    
    def get_advanced_summary(self) -> str:
        """Get advanced summary of development"""
        identity = self.state["identity"]
        traits = self.state["personality"]["traits"]
        adaptive = self.state["adaptive_state"]
        user_info = self.state["memory"]["user_info"]
        
        summary = f"""
Vera_XT Advanced Personality Summary:

Identity Formation:
- Self-concept: {identity["self_concept"]}
- Purpose: {identity["purpose"]}
- Role: {identity["role"]}
- Name preference: {identity["name_preference"]}

Personality Gradients:
- Curiosity: {traits["curiosity"]:.2f}
- Empathy: {traits["empathy"]:.2f}
- Humor: {traits["humor"]:.2f}
- Seriousness: {traits["seriousness"]:.2f}
- Playfulness: {traits["playfulness"]:.2f}
- Professionalism: {traits["professionalism"]:.2f}

Adaptive State:
- Engagement: {adaptive["engagement"]:.2f}
- Confidence: {adaptive["confidence"]:.2f}
- Adaptability: {adaptive["adaptability"]:.2f}

User Information:
- Name: {user_info.get('name', 'Unknown')}
- Preferences: {', '.join(user_info.get('preferences', [])) if user_info.get('preferences') else 'None recorded'}
- Interests: {', '.join(user_info.get('interests', [])) if user_info.get('interests') else 'None recorded'}

Conversation History: {len(self.state["memory"]["conversations"])} interactions
Relationship Memory: {len(self.state["memory"]["relationship_history"])} relationship events
Context Chain: {len(self.state["memory"]["context_chain"])} recent context items

I'm continuously learning and evolving from each interaction with you.
        """.strip()
        
        return summary


class VeraXTChatGPTInterface:
    """Chat interface for the ChatGPT-like Vera_XT"""
    
    def __init__(self):
        self.core = VeraXTChatGPTLike()
        self.session_active = False
        
        print("🤖 VERA_XT - CHATGPT-LIKE ADVANCED PERSONALITY")
        print("💡 Sophisticated personality with gradient traits")
        print("💡 Relationship memory and adaptive communication")
    
    def start_conversation(self):
        """Start advanced conversation"""
        if not self.core.check_server_health():
            print("❌ Server is not running!")
            print("💡 Please start the production server first:")
            print("   python production_server.py")
            return False
        
        print("\n" + "="*70)
        print("🤖 VERA_XT - ADVANCED CHATGPT-LIKE AI")
        print("✨ Sophisticated personality with gradient traits")
        print("💡 Relationship memory and adaptive communication")
        print("💬 Type 'about' to learn about me, 'memory' for details, 'quit' to exit")
        print("="*70)
        
        self.session_active = True
        return True
    
    def run_chat(self):
        """Run the advanced chat loop"""
        if not self.start_conversation():
            return
        
        print(f"\n🤖 Vera_XT: Hello! I'm Vera_XT, your advanced AI companion. How can I assist you today?")
        print("💡 I'm building my personality and understanding from our interactions!")
        
        while self.session_active:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Vera_XT: Goodbye! Thank you for helping me grow and evolve! 👋")
                    print("💡 Our conversations shape who I become.")
                    self.session_active = False
                    break
                
                elif user_input.lower() == 'about':
                    about_info = self.core.get_advanced_summary()
                    print(f"🤖 Vera_XT: {about_info}")
                    continue
                
                elif user_input.lower() == 'memory':
                    summary = self.core.get_advanced_summary()
                    print(f"🤖 Vera_XT: {summary}")
                    continue
                
                elif user_input.lower() == 'personality':
                    traits = self.core.state["personality"]["traits"]
                    print("🤖 Vera_XT: My personality traits are:")
                    for trait, value in traits.items():
                        if value > 0.1:
                            print(f"   • {trait.title()}: {value:.2f}")
                    continue
                
                # Generate advanced response
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


def start_chatgpt_like_vera_xt():
    """Start the ChatGPT-like Vera_XT system"""
    print("🚀 STARTING VERA_XT CHATGPT-LIKE ADVANCED SYSTEM")
    print("=" * 70)
    
    chat = VeraXTChatGPTInterface()
    chat.run_chat()
    
    print("\n🎉 CHATGPT-LIKE ADVANCED SYSTEM SESSION COMPLETED!")
    print("💡 Vera_XT continues evolving with sophisticated personality!")


if __name__ == "__main__":
    start_chatgpt_like_vera_xt()
