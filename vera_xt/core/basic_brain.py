#/storage/emulated/0/Vxt/Vxt/vera_xt/core/basic_brain.py
#!/usr/bin/env python3
"""
Basic Brain System - Human-like intelligence
Orchestrates all brain modules with consistency management
"""

from pathlib import Path
from typing import Dict, List, Any

# Import modular components
from .model_interface import ModelInterface
from .context_analyzer import ContextAnalyzer
from .response_generator import ResponseGenerator
from .memory_handler import MemoryHandler
from .learning_system import LearningSystem
from .consistency_manager import ConsistencyManager

class BasicBrain:
    def __init__(self):
        print("🧠 Human-like Brain initialized")
        print("💡 Nothing hardcoded - learns from every interaction")
        
        # Initialize modular components
        self.model_interface = ModelInterface()
        self.context_analyzer = ContextAnalyzer()
        self.response_generator = ResponseGenerator()
        self.memory_handler = MemoryHandler()
        self.learning_system = LearningSystem()
        self.consistency_manager = ConsistencyManager()
        
        # Expose key attributes for compatibility
        self.current_model_path = self.model_interface.current_model_path
        self.model_loaded = self.model_interface.model_loaded
        self.llm = self.model_interface.llm
        self.short_term_memory = self.memory_handler.short_term_memory
        self.long_term_memory = self.memory_handler.long_term_memory
        self.word_associations = self.context_analyzer.word_associations
        self.response_patterns = self.response_generator.response_patterns
        self.emotional_sensitivity = self.learning_system.emotional_sensitivity
        self.adaptation_level = self.learning_system.adaptation_level
        self.known_categories = self.context_analyzer.known_categories
        
        # Advanced server integration (new)
        self.server_integration = None
        self.enhanced_trainer = None
        self.server_training_mode = False
    
    def initialize_server_integration(self):
        """Initialize server-based training capabilities"""
        try:
            from .advanced_server_integration import AdvancedServerIntegration, EnhancedTrainingSystem
            
            from .config_manager import config_manager  # Import config manager
            
            self.server_integration = AdvancedServerIntegration(config_manager)
            self.server_integration.set_memory_handler(self.memory_handler)
            
            self.enhanced_trainer = EnhancedTrainingSystem(
                self.server_integration, 
                self.memory_handler
            )
            
            self.server_training_mode = True
            print("🌐 Advanced server integration initialized")
            return True
        except ImportError as e:
            print(f"⚠️ Server integration not available: {e}")
            return False
    
    def enhanced_thinking_process(self, input_text: str) -> str:
        """Enhanced thinking process using server capabilities"""
        if self.server_training_mode and self.server_integration:
            # Check if server is responding
            if self.server_integration._test_server_health():
                return self.server_integration.generate_with_openai_compatibility(input_text)
            else:
                # Fallback to local processing
                return self._get_local_response(input_text)
        else:
            # Fallback to local processing
            return self._get_local_response(input_text)
    
    def set_api_client(self, api_client):
        """Set API client for training integration"""
        self.api_client = api_client
        self.consistency_manager.set_api_client(api_client)
        self.api_training_mode = True
        print("🌐 API client connected for training")
    
    def load_model(self, model_name: str) -> bool:
        """Load a model - delegates to model interface"""
        return self.model_interface.load_model(model_name)
    
    def get_available_models(self):
        """Get available models - delegates to model interface"""
        return self.model_interface.get_available_models()
    
    def think_human_like(self, input_text: str) -> str:
        """Simulate human-like thinking process"""
        # If in API training mode, sync with API brain
        if hasattr(self, 'api_training_mode') and self.api_training_mode and hasattr(self, 'api_client'):
            # Get local response first
            local_response = self._get_local_response(input_text)
            
            # Sync with API for consistency
            final_response = self.consistency_manager.sync_with_api_brain(
                input_text, local_response
            )
            
            return final_response
        else:
            # Normal offline operation
            return self._get_local_response(input_text)
    
    def _get_local_response(self, input_text: str) -> str:
        """Get response from local brain only"""
        # If model interface is available, use it for the response with clean input
        if self.model_interface.model_interface and self.model_interface.llm:
            # Use the actual model for response generation with clean input
            return self.model_interface.model_interface(input_text)
        
        # If no model interface, use fallback brain logic
        # 1. Analyze the input context (learns patterns)
        context_analysis = self.context_analyzer.analyze_context(input_text)
        context_analysis['original_input'] = input_text  # Add original input for reference
        
        # 2. Retrieve relevant memories
        relevant_memories = self.memory_handler.retrieve_relevant_memories(input_text)
        
        # 3. Consider multiple perspectives
        multiple_approaches = self.response_generator.generate_multiple_approaches(input_text, relevant_memories)
        
        # 4. Evaluate best approach
        best_approach = self.response_generator.evaluate_best_approach(multiple_approaches, context_analysis)
        
        # 5. Generate human-like response
        response = self.response_generator.generate_thoughtful_response(
            best_approach, 
            context_analysis,
            self.model_interface.model_interface,
            self.model_interface.llm
        )
        
        # Learn from this interaction
        self.learning_system.learn_from_interaction(input_text, response, context_analysis)
        
        return response
    
    def process_input(self, user_input: str) -> str:
        """Process user input with human-like intelligence"""
        if not self.model_interface.model_loaded and not self.model_interface.model_interface:
            return "I'm ready to help! Please load a model or set up the model interface first."
        
        # Add to short-term memory
        self.memory_handler.add_to_memory("user_input", user_input, self.context_analyzer.analyze_context(user_input))
        
        # Think like a human would (learns from every interaction)
        response = self.think_human_like(user_input)
        
        # Add response to short-term memory
        self.memory_handler.add_to_memory("assistant_response", response)
        
        # Update learned patterns
        self.learning_system.learn_from_interaction(user_input, response, self.context_analyzer.analyze_context(user_input))
        
        return response
    
    def get_brain_status(self) -> Dict[str, Any]:
        """Get current brain status"""
        return {
            "model_loaded": self.model_interface.model_loaded,
            "current_model": self.model_interface.current_model_path.name if self.model_interface.model_loaded else None,
            "short_term_memory_count": len(self.memory_handler.short_term_memory),
            "emotional_sensitivity": self.learning_system.emotional_sensitivity,
            "adaptation_level": self.learning_system.adaptation_level,
            "learned_word_count": len(self.context_analyzer.word_associations),
            "learned_categories": list(self.context_analyzer.known_categories),
            "available_models": self.model_interface.get_available_models(),
            "learned_response_patterns": dict(self.response_generator.response_patterns),
            "model_interface_connected": self.model_interface.model_interface is not None,
            "api_training_mode": getattr(self, 'api_training_mode', False),
            "consistency_status": self.consistency_manager.get_consistency_status(),
            "server_integration_active": self.server_training_mode,
            "server_capabilities": self.server_integration.get_server_capabilities() if self.server_integration else None
        }