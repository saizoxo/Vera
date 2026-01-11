#/storage/emulated/0/Vxt/Vxt/vera_xt/core/consistency_manager.py
#!/usr/bin/env python3
"""
Consistency Manager - Synchronizes local brain with API brain
Maintains consistency between offline and online systems
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class ConsistencyManager:
    def __init__(self, memory_dir: str = "Memory_Data"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Consistency tracking
        self.sync_log = []
        self.knowledge_diffs = []  # Differences between local and API
        self.pattern_transfers = []  # Patterns transferred from API to local
        
        # API integration
        self.api_client = None  # Will be set by external system
        self.last_sync_time = 0
        self.sync_interval = 300  # Sync every 5 minutes
        
        print("🔄 Consistency Manager initialized")
        print("💡 Synchronizes local brain with API brain")
    
    def set_api_client(self, api_client):
        """Set the API client for communication"""
        self.api_client = api_client
        print("✅ API client connected to consistency manager")
    
    def sync_with_api_brain(self, user_input: str, local_response: str) -> str:
        """Sync with API brain and get improved response if needed"""
        if not self.api_client:
            return local_response  # Use local response if no API available
        
        try:
            # Get API response for the same input
            api_response = self.api_client.generate_response(user_input)
            
            # Compare responses and learn from differences
            improvement = self.compare_and_learn(user_input, local_response, api_response)
            
            # Decide which response to use based on quality
            final_response = self.select_best_response(local_response, api_response, improvement)
            
            # Update local patterns based on API knowledge
            self.update_local_patterns(user_input, api_response)
            
            # Log the sync event
            self.log_sync_event(user_input, local_response, api_response, final_response)
            
            return final_response
            
        except Exception as e:
            print(f"⚠️ API sync failed: {e}")
            return local_response  # Fallback to local response
    
    def compare_and_learn(self, input_text: str, local_resp: str, api_resp: str) -> Dict[str, Any]:
        """Compare responses and identify learning opportunities"""
        # Analyze differences between local and API responses
        local_words = set(local_resp.lower().split())
        api_words = set(api_resp.lower().split())
        
        # Identify new information from API
        new_info = api_words - local_words
        common_info = local_words & api_words
        
        # Calculate improvement potential
        improvement_score = len(new_info) / len(api_words) if api_words else 0
        
        return {
            "new_info_count": len(new_info),
            "common_info_count": len(common_info),
            "improvement_score": improvement_score,
            "new_concepts": list(new_info)[:10],  # Top 10 new concepts
            "similarity": len(common_info) / len(local_words) if local_words else 0
        }
    
    def select_best_response(self, local_resp: str, api_resp: str, improvement: Dict[str, Any]) -> str:
        """Select the best response based on quality metrics"""
        # If API provides significant improvement, use API response
        if improvement["improvement_score"] > 0.3:  # 30% new info threshold
            return api_resp
        # If responses are similar, prefer local (faster, private)
        elif improvement["similarity"] > 0.7:
            return local_resp
        # Otherwise, use local response as default
        else:
            return local_resp
    
    def update_local_patterns(self, input_text: str, api_response: str):
        """Update local brain patterns based on API knowledge"""
        # Extract key concepts from API response
        words = api_response.lower().split()
        
        # Update context analyzer word associations
        for word in words[:20]:  # Top 20 words
            if len(word) > 3:  # Only significant words
                # Add to learned patterns
                if word not in self.pattern_transfers:
                    self.pattern_transfers.append(word)
        
        # Store successful patterns for future learning
        success_entry = {
            "timestamp": time.time(),
            "input": input_text,
            "api_response": api_response,
            "learned_patterns": words[:20]
        }
        
        self.knowledge_diffs.append(success_entry)
        
        # Limit memory usage
        if len(self.knowledge_diffs) > 100:
            self.knowledge_diffs = self.knowledge_diffs[-50:]  # Keep last 50
    
    def log_sync_event(self, input_text: str, local_resp: str, api_resp: str, final_resp: str):
        """Log synchronization event for tracking"""
        sync_event = {
            "timestamp": time.time(),
            "input": input_text,
            "local_response": local_resp,
            "api_response": api_resp,
            "final_response": final_resp,
            "sync_time": time.time()
        }
        
        self.sync_log.append(sync_event)
        
        # Limit log size
        if len(self.sync_log) > 50:
            self.sync_log = self.sync_log[-25:]  # Keep last 25 events
    
    def get_consistency_status(self) -> Dict[str, Any]:
        """Get current consistency status"""
        return {
            "sync_events_count": len(self.sync_log),
            "knowledge_diffs_count": len(self.knowledge_diffs),
            "pattern_transfers_count": len(self.pattern_transfers),
            "last_sync_time": self.last_sync_time,
            "sync_interval": self.sync_interval
        }
    
    def train_local_from_api_history(self):
        """Use API interaction history to train local model"""
        if not self.knowledge_diffs:
            return
        
        # Create training data from successful API interactions
        training_data = []
        for diff in self.knowledge_diffs[-20:]:  # Last 20 interactions
            training_data.append({
                "input": diff["input"],
                "output": diff["api_response"],
                "learned_patterns": diff["learned_patterns"]
            })
        
        # This would be used to fine-tune local model
        # (Implementation depends on your specific fine-tuning approach)
        print(f"📚 Prepared {len(training_data)} training examples from API interactions")
        return training_data
    
    def save_consistency_state(self):
        """Save consistency state to persistent storage"""
        state = {
            "sync_log": self.sync_log,
            "knowledge_diffs": self.knowledge_diffs,
            "pattern_transfers": self.pattern_transfers,
            "last_sync_time": self.last_sync_time
        }
        
        state_file = self.memory_dir / "consistency_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 Consistency state saved to {state_file}")
    
    def load_consistency_state(self):
        """Load consistency state from persistent storage"""
        state_file = self.memory_dir / "consistency_state.json"
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.sync_log = state.get("sync_log", [])
            self.knowledge_diffs = state.get("knowledge_diffs", [])
            self.pattern_transfers = state.get("pattern_transfers", [])
            self.last_sync_time = state.get("last_sync_time", 0)
            
            print(f"📥 Consistency state loaded from {state_file}")
        else:
            print("📋 No consistency state found, starting fresh")
