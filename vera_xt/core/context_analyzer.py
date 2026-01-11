#/storage/emulated/0/Vxt/Vxt/vera_xt/core/context_analyzer.py
#!/usr/bin/env python3
"""
Context Analyzer Module - Analyzes input context and infers meaning
"""

import re
from typing import Dict, List, Any

class ContextAnalyzer:
    def __init__(self):
        # Learned categories (not hardcoded)
        self.known_categories = set()
        self.word_associations = {}     # Learns from context
        self.response_patterns = {}     # Learns effective responses
    
    def analyze_context(self, input_text: str) -> Dict[str, Any]:
        """Analyze context using learned patterns, not hardcoded rules"""
        # Use learned patterns to classify
        input_type = self.infer_input_type(input_text)
        
        # Learn emotional tone from context clues
        emotional_tone = self.infer_emotional_tone(input_text)
        
        # Learn urgency from learned patterns
        urgency_level = self.infer_urgency(input_text)
        
        # Learn complexity from learned associations
        complexity_level = self.infer_complexity(input_text)
        
        # Learn personal reference from learned patterns
        personal_reference = self.infer_personal_reference(input_text)
        
        # Extract context clues using learned associations
        context_clues = self.extract_context_clues(input_text)
        
        context = {
            "input_type": input_type,
            "emotional_tone": emotional_tone,
            "urgency_level": urgency_level,
            "complexity_level": complexity_level,
            "personal_reference": personal_reference,
            "context_clues": context_clues
        }
        
        return context
    
    def infer_input_type(self, text: str) -> str:
        """Infer input type from learned patterns and associations"""
        # Look for learned associations with technical terms
        technical_indicators = 0
        advisory_indicators = 0
        greeting_indicators = 0
        planning_indicators = 0
        
        words = text.lower().split()
        
        # Check learned word associations
        for word in words:
            if word in self.word_associations:
                category = self.word_associations[word]
                if category == "technical":
                    technical_indicators += 1
                elif category == "advisory":
                    advisory_indicators += 1
                elif category == "greeting":
                    greeting_indicators += 1
                elif category == "planning":
                    planning_indicators += 1
        
        # If no learned associations, use basic heuristics but learn from them
        if max(technical_indicators, advisory_indicators, greeting_indicators, planning_indicators) == 0:
            # Basic heuristics for first-time encounters
            if any(w in words for w in ["code", "python", "function", "debug", "error", "program"]):
                # Learn this association
                for w in ["code", "python", "function", "debug", "error", "program"]:
                    if w in words:
                        self.word_associations[w] = "technical"
                return "technical"
            elif any(w in words for w in ["think", "should", "advice", "suggest", "help"]):
                for w in ["think", "should", "advice", "suggest", "help"]:
                    if w in words:
                        self.word_associations[w] = "advisory"
                return "advisory"
            elif any(w in words for w in ["hello", "hi", "hey"]):
                for w in ["hello", "hi", "hey"]:
                    if w in words:
                        self.word_associations[w] = "greeting"
                return "greeting"
            elif any(w in words for w in ["plan", "organize", "schedule"]):
                for w in ["plan", "organize", "schedule"]:
                    if w in words:
                        self.word_associations[w] = "planning"
                return "planning"
            else:
                return "general"
        else:
            # Use learned patterns
            max_indicators = max(technical_indicators, advisory_indicators, greeting_indicators, planning_indicators)
            if max_indicators == technical_indicators:
                return "technical"
            elif max_indicators == advisory_indicators:
                return "advisory"
            elif max_indicators == greeting_indicators:
                return "greeting"
            elif max_indicators == planning_indicators:
                return "planning"
            else:
                return "general"
    
    def infer_emotional_tone(self, text: str) -> float:
        """Infer emotional tone from learned emotional associations"""
        words = text.lower().split()
        total_score = 0.0
        word_count = 0
        
        for word in words:
            if word in self.word_associations and isinstance(self.word_associations[word], dict) and "emotion" in self.word_associations[word]:
                # Use learned emotional association
                emotion_val = self.word_associations[word]["emotion"]
                total_score += emotion_val
                word_count += 1
            else:
                # For new words, use basic heuristics but learn
                positive_words = ["good", "great", "love", "like", "happy", "excited", "wonderful", "amazing"]
                negative_words = ["bad", "hate", "angry", "frustrated", "annoying", "terrible", "awful"]
                
                if word in positive_words:
                    self.word_associations[word] = {"emotion": 0.5, "category": "positive"}
                    total_score += 0.5
                    word_count += 1
                elif word in negative_words:
                    self.word_associations[word] = {"emotion": -0.5, "category": "negative"}
                    total_score += -0.5
                    word_count += 1
        
        # Update emotional sensitivity based on learned patterns
        if word_count > 0:
            avg_emotion = total_score / word_count
            return avg_emotion
        else:
            return 0.0
    
    def infer_urgency(self, text: str) -> float:
        """Infer urgency from learned urgency indicators"""
        words = text.lower().split()
        urgency_score = 0.0
        
        for word in words:
            if word in self.word_associations and isinstance(self.word_associations[word], dict) and "urgency" in self.word_associations[word]:
                # Use learned urgency association
                urgency_score = max(urgency_score, self.word_associations[word]["urgency"])
            else:
                # Learn urgency for new words
                urgent_words = ["urgent", "asap", "immediately", "now", "quick", "fast", "soon", "today"]
                if word in urgent_words:
                    self.word_associations[word] = {"urgency": 0.8, "category": "time_sensitive"}
                    urgency_score = max(urgency_score, 0.8)
        
        return min(1.0, urgency_score)
    
    def infer_complexity(self, text: str) -> float:
        """Infer complexity from learned complexity patterns"""
        # Learn from sentence structure and learned complexity indicators
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Learn complexity based on learned technical terms
        technical_words = sum(1 for word in words if word.lower() in self.word_associations and 
                             isinstance(self.word_associations[word.lower()], dict) and 
                             self.word_associations[word.lower()].get("category") == "technical")
        
        complexity_score = (avg_word_length / 10.0 + technical_words / len(words) if words else 0) / 2.0
        return min(1.0, complexity_score)
    
    def infer_personal_reference(self, text: str) -> bool:
        """Infer personal reference from learned personal patterns"""
        personal_words = ["my", "i", "me", "myself", "our", "we", "us"]
        words = text.lower().split()
        
        for word in words:
            if word in personal_words:
                # Learn this as a personal reference pattern
                if word not in self.word_associations:
                    self.word_associations[word] = {"category": "personal_ref"}
                return True
        return False
    
    def extract_context_clues(self, text: str) -> List[str]:
        """Extract context clues using learned associations"""
        clues = []
        
        # Extract potential project names, file names from learned patterns
        file_matches = re.findall(r'\b\w+\.\w+\b', text)
        for match in file_matches:
            clues.append(match)
            # Learn this as a file reference
            self.word_associations[match] = {"category": "file_reference"}
        
        # Extract quoted phrases
        quoted_matches = re.findall(r'"([^"]*)"', text) + re.findall(r"'([^']*)'", text)
        for match in quoted_matches:
            clues.append(match)
            # Learn this as a quoted reference
            self.word_associations[match.lower()] = {"category": "quoted_content"}
        
        return list(set(clues))