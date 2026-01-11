#/storage/emulated/0/Vxt/Vxt/vera_xt/core/learning_system.py
#!/usr/bin/env python3
"""
Learning System Module - Handles learning and adaptation
"""

from typing import Dict, Any

class LearningSystem:
    def __init__(self):
        # Human-like capabilities
        self.emotional_sensitivity = 0.0  # Learns from user interactions
        self.adaptation_level = 0.0     # How well adapted to user
        
        # Learned categories (not hardcoded)
        self.known_categories = set()
        self.word_associations = {}     # Learns from context
        self.response_patterns = {}     # Learns effective responses
    
    def learn_from_interaction(self, user_input: str, response: str, context: Dict[str, Any]):
        """Learn from this interaction to improve future responses"""
        input_type = context.get("input_type", "general")
        
        # Learn effectiveness of approach for this input type
        if input_type not in self.response_patterns:
            self.response_patterns[input_type] = response[:50].lower()  # Store first 50 chars as pattern
        
        # Learn from emotional response
        if context.get("emotional_tone", 0) != 0:
            self.emotional_sensitivity = min(1.0, self.emotional_sensitivity + 0.05)
    
    def get_learning_status(self):
        """Get current learning status"""
        return {
            "emotional_sensitivity": self.emotional_sensitivity,
            "adaptation_level": self.adaptation_level,
            "learned_word_count": len(self.word_associations),
            "learned_categories": list(self.known_categories),
            "learned_response_patterns": dict(self.response_patterns)
        }
