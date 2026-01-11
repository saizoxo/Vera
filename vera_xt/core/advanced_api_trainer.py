#/storage/emulated/0/Vxt/Vxt/vera_xt/core/advanced_api_trainer.py
#!/usr/bin/env python3
"""
Advanced API Trainer - Enhanced training with latest llama.cpp features
Leverages OpenRouter API for sophisticated learning and memory enhancement
"""

import os
import requests
import json
import time
from typing import Dict, Any, List
from pathlib import Path

class AdvancedAPITrainer:
    def __init__(self, memory_handler, config_manager):
        self.memory_handler = memory_handler
        self.config = config_manager
        self.api_key = config_manager.get('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Use advanced models from OpenRouter
        self.models = {
            "grok": "x-ai/grok-4.1-fast:free",
            "gemma": "google/gemma-2-9b-it",
            "mistral": "mistralai/mistral-nemo:free"
        }
        
        # Training data storage
        self.training_data_dir = Path("Training_Data")
        self.training_data_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.session_log = []
        self.quality_metrics = {
            "response_quality": [],
            "training_effectiveness": [],
            "memory_enrichment": []
        }
        
        print("🎓 Advanced API Trainer initialized")
        print(f"🤖 Available models: {list(self.models.keys())}")
    
    def generate_with_advanced_api(self, user_input: str, model_type: str = "grok") -> str:
        """Generate response using advanced API with quality assessment"""
        if not self.api_key:
            return "API key not available for training"
        
        model = self.models.get(model_type, self.models["grok"])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Enhanced prompt with training context
        enhanced_prompt = f"""
        User query: {user_input}
        
        Please provide a comprehensive, educational response that:
        1. Addresses the core question directly
        2. Provides context and background information
        3. Includes practical examples or applications
        4. Suggests related concepts for deeper understanding
        
        Format your response in a structured, educational manner.
        """
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an educational AI assistant focused on comprehensive, structured responses."},
                {"role": "user", "content": enhanced_prompt}
            ],
            "max_tokens": self.config.get_int('MAX_TOKENS', 512),
            "temperature": self.config.get_float('TEMPERATURE', 0.7),
            "top_p": self.config.get_float('TOP_P', 0.9)
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            api_response = result['choices'][0]['message']['content'].strip()
            
            processing_time = time.time() - start_time
            
            # Store this interaction for advanced training
            self._store_advanced_training_interaction(
                user_input, api_response, model, processing_time
            )
            
            return api_response
            
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return "Sorry, I couldn't get a response from the API right now."
    
    def _store_advanced_training_interaction(self, user_input: str, api_response: str, model: str, processing_time: float):
        """Store advanced training interaction with rich metadata"""
        # Analyze the interaction for training value
        interaction_analysis = self._analyze_interaction_quality(user_input, api_response)
        
        interaction = {
            "timestamp": time.time(),
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
            "input": user_input,
            "api_response": api_response,
            "model_used": model,
            "processing_time": processing_time,
            "analysis": interaction_analysis,
            "semantic_tags": self._extract_training_tags(user_input, api_response),
            "quality_score": interaction_analysis["overall_quality"],
            "training_value": interaction_analysis["training_value"],
            "memory_enhancement": interaction_analysis["memory_enhancement"]
        }
        
        # Add to memory with high importance for training
        self.memory_handler.add_to_memory(
            "advanced_training_interaction", 
            f"Input: {user_input}\nAPI Response: {api_response}", 
            category="knowledge",
            context={
                "importance": 9,  # High importance for training
                "training_data": True,
                "model_used": model,
                "quality_score": interaction_analysis["overall_quality"],
                "semantic_tags": interaction["semantic_tags"]
            }
        )
        
        # Save to advanced training data file
        training_file = self.training_data_dir / f"advanced_training_{int(time.time())}.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(interaction, f, indent=2)
        
        # Update quality metrics
        self.quality_metrics["response_quality"].append(interaction_analysis["overall_quality"])
        self.quality_metrics["training_effectiveness"].append(interaction_analysis["training_value"])
        self.quality_metrics["memory_enhancement"].append(interaction_analysis["memory_enhancement"])
    
    def _analyze_interaction_quality(self, user_input: str, api_response: str) -> Dict[str, Any]:
        """Analyze the quality and training value of an interaction"""
        analysis = {
            "input_complexity": len(user_input.split()),
            "response_length": len(api_response.split()),
            "coherence_score": self._assess_coherence(user_input, api_response),
            "educational_value": self._assess_educational_value(api_response),
            "relevance_score": self._assess_relevance(user_input, api_response),
            "training_value": 0.0,
            "memory_enhancement": 0.0,
            "overall_quality": 0.0
        }
        
        # Calculate composite scores
        analysis["training_value"] = (
            analysis["coherence_score"] * 0.3 +
            analysis["educational_value"] * 0.4 +
            analysis["relevance_score"] * 0.3
        )
        
        analysis["memory_enhancement"] = (
            analysis["educational_value"] * 0.5 +
            analysis["relevance_score"] * 0.3 +
            min(analysis["response_length"] / 100, 1.0) * 0.2  # Longer responses = more info
        )
        
        analysis["overall_quality"] = (
            analysis["training_value"] * 0.6 +
            analysis["memory_enhancement"] * 0.4
        )
        
        return analysis
    
    def _assess_coherence(self, user_input: str, api_response: str) -> float:
        """Assess how coherent and well-structured the response is"""
        # Simple coherence assessment
        response_lower = api_response.lower()
        
        # Check for structured elements
        structure_indicators = [
            "first", "second", "third", "finally",
            "1.", "2.", "3.", "4.",
            "for example", "in conclusion", "additionally"
        ]
        
        structure_score = sum(1 for indicator in structure_indicators if indicator in response_lower)
        structure_score = min(structure_score / 5, 1.0)  # Normalize to 0-1
        
        # Check for comprehensive coverage
        input_words = set(user_input.lower().split())
        response_words = set(response_lower.split())
        coverage = len(input_words & response_words) / max(len(input_words), 1)
        
        return (structure_score * 0.6 + coverage * 0.4)
    
    def _assess_educational_value(self, api_response: str) -> float:
        """Assess the educational value of the response"""
        educational_indicators = [
            "example", "explain", "understand", "concept", "principle",
            "application", "practical", "theory", "method", "approach",
            "benefit", "advantage", "disadvantage", "consideration"
        ]
        
        response_lower = api_response.lower()
        education_score = sum(1 for indicator in educational_indicators if indicator in response_lower)
        return min(education_score / 10, 1.0)  # Normalize to 0-1
    
    def _assess_relevance(self, user_input: str, api_response: str) -> float:
        """Assess how relevant the response is to the input"""
        input_words = set(user_input.lower().split())
        response_words = set(api_response.lower().split())
        
        if not input_words:
            return 0.0
        
        # Calculate overlap and semantic similarity
        overlap = len(input_words & response_words)
        relevance = overlap / len(input_words)
        
        return min(relevance, 1.0)
    
    def _extract_training_tags(self, user_input: str, api_response: str) -> List[str]:
        """Extract tags that would be useful for training"""
        tags = []
        
        # Content type tags
        if any(q in user_input.lower() for q in ["what", "how", "why", "when", "where", "who"]):
            tags.append("question:inquiry")
        
        if any(term in api_response.lower() for term in ["python", "code", "function", "algorithm", "programming"]):
            tags.append("technical:programming")
        
        if any(term in api_response.lower() for term in ["example", "illustrate", "demonstrate", "show"]):
            tags.append("educational:example")
        
        # Extract key concepts from response
        response_words = api_response.lower().split()
        key_concepts = [word for word in response_words if len(word) > 5 and word.isalpha()][:5]
        for concept in key_concepts:
            tags.append(f"concept:{concept}")
        
        return list(set(tags))  # Remove duplicates
    
    def get_training_insights(self) -> Dict[str, Any]:
        """Get insights about training effectiveness"""
        if not self.quality_metrics["response_quality"]:
            return {"status": "No training data available yet"}
        
        avg_quality = sum(self.quality_metrics["response_quality"]) / len(self.quality_metrics["response_quality"])
        avg_training = sum(self.quality_metrics["training_effectiveness"]) / len(self.quality_metrics["training_effectiveness"])
        avg_enhancement = sum(self.quality_metrics["memory_enhancement"]) / len(self.quality_metrics["memory_enhancement"])
        
        return {
            "total_interactions": len(self.quality_metrics["response_quality"]),
            "average_response_quality": round(avg_quality, 3),
            "average_training_effectiveness": round(avg_training, 3),
            "average_memory_enhancement": round(avg_enhancement, 3),
            "models_used": list(self.models.keys()),
            "training_data_files": len(list(self.training_data_dir.glob("*.json")))
        }
    
    def train_local_model(self):
        """Prepare training data for local model improvement"""
        # This would be where you prepare data to fine-tune your local model
        # For now, we'll just return the collected training data
        training_files = list(self.training_data_dir.glob("*.json"))
        
        training_data = []
        for file in training_files:
            with open(file, 'r', encoding='utf-8') as f:
                training_data.append(json.load(f))
        
        return training_data