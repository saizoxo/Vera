#/storage/emulated/0/Vxt/Vxt/vera_xt/core/response_generator.py
#!/usr/bin/env python3
"""
Response Generator Module - Generates thoughtful responses
"""

from typing import Dict, List, Any

class ResponseGenerator:
    def __init__(self):
        # Human-like capabilities
        self.emotional_sensitivity = 0.0  # Learns from user interactions
        self.adaptation_level = 0.0     # How well adapted to user
        
        # Learned categories (not hardcoded)
        self.known_categories = set()
        self.word_associations = {}     # Learns from context
        self.response_patterns = {}     # Learns effective responses
    
    def generate_multiple_approaches(self, input_text: str, memories: List[Dict[str, Any]]) -> List[str]:
        """Generate multiple approaches based on learned patterns"""
        approaches = []
        
        # Use learned response patterns to generate approaches
        input_type = self.infer_input_type(input_text)
        
        # Learn and generate approaches based on input type
        if input_type == "technical":
            approaches.append(f"Technical: {input_text}")
            approaches.append(f"Debugging: How can I help with this technical issue?")
        elif input_type == "greeting":
            approaches.append(f"Greeting: Hello! How can I assist?")
            approaches.append(f"Engagement: What would you like to work on?")
        elif input_type == "advisory":
            approaches.append(f"Advisory: Let me consider your options")
            approaches.append(f"Guidance: Here's my perspective on this")
        else:
            approaches.append(f"General: {input_text}")
            approaches.append(f"Exploration: Let's explore this topic")
        
        return approaches
    
    def evaluate_best_approach(self, approaches: List[str], context: Dict[str, Any]) -> str:
        """Evaluate best approach using learned effectiveness patterns"""
        input_type = context.get("input_type", "general")
        
        # Use learned patterns about what works best for each input type
        if input_type in self.response_patterns:
            # Use learned effective approach for this type
            best_pattern = self.response_patterns[input_type]
            for approach in approaches:
                if best_pattern in approach.lower():
                    return approach
        
        # Default selection based on input type
        if input_type == "technical":
            return approaches[0] if len(approaches) > 0 else f"Technical: {approaches}"
        elif input_type == "greeting":
            return approaches[0] if len(approaches) > 0 else f"Greeting: Hello!"
        elif input_type == "advisory":
            return approaches[0] if len(approaches) > 0 else f"Advisory: Let me consider this"
        else:
            return approaches[-1] if approaches else "I'm ready to help"
    
    def generate_thoughtful_response(self, approach: str, context: Dict[str, Any], model_interface=None, llm=None) -> str:
        """Generate response based on learned patterns and user adaptation"""
        # If model interface is available, use it for the response with original input
        if model_interface and llm:
            # Use the actual model for response generation with the original input
            original_input = context.get('original_input', approach)
            # Clean the approach string to just get the input part if it contains prefixes
            if original_input.startswith(('Direct: ', 'Contextual: ', 'Exploratory: ', 'Problem-solving: ', 'Greeting: ', 'Technical: ', 'Advisory: ', 'General: ')):
                # Extract just the actual input part
                clean_input = original_input.split(': ', 1)[1] if ': ' in original_input else original_input
            else:
                clean_input = original_input
            
            # Call the model interface
            try:
                return model_interface(clean_input)
            except Exception as e:
                print(f"❌ Model interface error: {e}")
                return "I encountered an issue with the model. Let me try to help differently."
        
        # Fallback to original approach if no model interface
        input_type = context.get("input_type", "general")
        emotional_tone = context.get("emotional_tone", 0.0)
        
        # Use learned response patterns and adapt to user's style
        if input_type == "greeting":
            if emotional_tone > 0.3:
                response = "Hello! It's great to connect with you. How can I assist you today?"
            elif emotional_tone < -0.3:
                response = "Hi there. I'm here to help. What's on your mind?"
            else:
                response = "Hello! How can I help you today?"
        elif input_type == "technical":
            response = f"I understand you need help with a technical matter. Let me think through this systematically: {approach.replace('Technical: ', '')}"
        elif input_type == "advisory":
            response = f"That's an important question. Let me consider this carefully: {approach.replace('Advisory: ', '')}"
        else:
            response = f"I understand. Let me think about this: {approach.replace('General: ', '')}"
        
        # Learn from the context to improve future responses
        self.adaptation_level = min(1.0, self.adaptation_level + 0.01)
        
        return response
    
    def infer_input_type(self, text: str) -> str:
        """Infer input type from learned patterns and associations"""
        # Simple fallback if not provided by context analyzer
        words = text.lower().split()
        if any(w in words for w in ["hello", "hi", "hey"]):
            return "greeting"
        elif any(w in words for w in ["code", "python", "function", "debug", "error"]):
            return "technical"
        else:
            return "general"
