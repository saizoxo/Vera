#/storage/emulated/0/Vxt/Vxt/vera_xt/memory/memory_manager.py
#!/usr/bin/env python3
"""
Memory Manager - Coordinates all memory operations
Integrates with the learning brain system
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

class MemoryManager:
    def __init__(self, memory_dir: str = "Memory_Data"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Memory system components
        self.simple_memory = None  # Will be set by external system
        self.conversation_context = {}
        self.personal_context = {}  # Learned personal patterns
        
        # Memory optimization
        self.memory_cache = {}  # Cache recent memories
        self.cache_size_limit = 100
        self.archived_conversations = set()
        
        print("🧠 Memory Manager initialized")
        print("💡 Coordinates with learning brain, optimizes storage")
    
    def set_simple_memory(self, simple_memory_instance):
        """Set the simple memory instance"""
        self.simple_memory = simple_memory_instance
        print("✅ Simple memory connected to manager")
    
    def store_interaction(self, user_input: str, ai_response: str, context: Dict[str, Any] = None) -> bool:
        """Store a complete interaction with context"""
        if not self.simple_memory:
            print("❌ Simple memory not connected")
            return False
        
        # Create tags from context
        tags = self._extract_tags_from_context(context or {})
        
        # Store user input
        user_success = self.simple_memory.add_memory(
            content=user_input,
            memory_type="user_input",
            tags=tags + ["user"],
            context=context or {}
        )
        
        # Store AI response
        ai_success = self.simple_memory.add_memory(
            content=ai_response,
            memory_type="ai_response", 
            tags=tags + ["assistant"],
            context=context or {}
        )
        
        return user_success and ai_success
    
    def _extract_tags_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Extract tags from brain context"""
        tags = []
        
        # Extract from brain context
        input_type = context.get("input_type", "general")
        tags.append(input_type)
        
        # Emotional tone tags
        emotional_tone = context.get("emotional_tone", 0.0)
        if emotional_tone > 0.3:
            tags.append("positive")
        elif emotional_tone < -0.3:
            tags.append("negative")
        else:
            tags.append("neutral")
        
        # Urgency tags
        urgency = context.get("urgency_level", 0.0)
        if urgency > 0.5:
            tags.append("high_urgency")
        elif urgency > 0.2:
            tags.append("medium_urgency")
        
        # Add any specific context clues
        context_clues = context.get("context_clues", [])
        tags.extend([clue.lower().replace(' ', '_') for clue in context_clues])
        
        return list(set(tags))  # Remove duplicates
    
    def retrieve_context(self, query: str, context_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant context for a query"""
        if not self.simple_memory:
            return {"relevant_memories": [], "personal_context": {}}
        
        # Find memories by content
        content_memories = self.simple_memory.find_memories_by_content(query, limit=5)
        
        # Find memories by tags (if context requirements specify tags)
        tag_memories = []
        if context_requirements and "required_tags" in context_requirements:
            tag_memories = self.simple_memory.find_memories_by_tags(
                context_requirements["required_tags"], limit=5
            )
        
        # Combine and deduplicate
        all_memories = content_memories + tag_memories
        unique_memories = []
        seen_ids = set()
        for memory in all_memories:
            mem_id = memory.get("id")
            if mem_id and mem_id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(mem_id)
        
        # Get personal context
        personal_context = self._build_personal_context(unique_memories)
        
        return {
            "relevant_memories": unique_memories[:10],  # Limit to 10
            "personal_context": personal_context,
            "contextual_relevance": len(unique_memories)
        }
    
    def _build_personal_context(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build personal context from memories"""
        personal_context = {
            "communication_style": self._analyze_communication_style(memories),
            "interests": self._extract_interests(memories),
            "preferences": self._extract_preferences(memories),
            "interaction_patterns": self._analyze_interaction_patterns(memories)
        }
        return personal_context
    
    def _analyze_communication_style(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's communication style from memories"""
        styles = {"direct": 0, "detailed": 0, "casual": 0, "formal": 0}
        
        for memory in memories:
            content = memory.get("content", "").lower()
            if any(word in content for word in ["just", "quick", "fast", "simple"]):
                styles["direct"] += 1
            elif len(content.split()) > 50:  # Long messages
                styles["detailed"] += 1
            elif any(word in content for word in ["hey", "hi", "cool", "awesome"]):
                styles["casual"] += 1
            elif any(word in content for word in ["please", "thank", "appreciate", "regards"]):
                styles["formal"] += 1
        
        # Return dominant style
        dominant_style = max(styles, key=styles.get)
        return {
            "dominant": dominant_style,
            "scores": styles,
            "adaptation_needed": styles[dominant_style] > 2
        }
    
    def _extract_interests(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract user interests from memories"""
        interests = set()
        
        for memory in memories:
            content = memory.get("content", "").lower()
            tags = memory.get("tags", [])
            
            # Extract from content
            technical_words = ["python", "code", "programming", "debug", "algorithm", "data", "ai", "ml"]
            creative_words = ["write", "story", "creative", "idea", "design", "art", "music"]
            planning_words = ["plan", "organize", "schedule", "todo", "task", "project"]
            
            for word in technical_words:
                if word in content:
                    interests.add(f"technical_{word}")
            for word in creative_words:
                if word in content:
                    interests.add(f"creative_{word}")
            for word in planning_words:
                if word in content:
                    interests.add(f"planning_{word}")
            
            # Extract from tags
            for tag in tags:
                if tag not in ["user", "assistant", "positive", "negative", "neutral"]:
                    interests.add(tag)
        
        return list(interests)
    
    def _extract_preferences(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract user preferences from memories"""
        preferences = {
            "response_length": "balanced",  # "short", "detailed", "balanced"
            "technical_depth": "moderate",  # "basic", "moderate", "advanced"
            "tone": "friendly"  # "formal", "friendly", "casual"
        }
        
        # Analyze based on interaction patterns
        total_memories = len(memories)
        if total_memories == 0:
            return preferences
        
        short_response_indicators = 0
        detailed_response_indicators = 0
        
        for memory in memories:
            content = memory.get("content", "")
            if len(content.split()) < 10:
                short_response_indicators += 1
            elif len(content.split()) > 50:
                detailed_response_indicators += 1
        
        if short_response_indicators / total_memories > 0.6:
            preferences["response_length"] = "short"
        elif detailed_response_indicators / total_memories > 0.4:
            preferences["response_length"] = "detailed"
        
        return preferences
    
    def _analyze_interaction_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction patterns"""
        patterns = {
            "frequency": len(memories),
            "time_spans": self._analyze_time_spans(memories),
            "topic_clusters": self._identify_topic_clusters(memories),
            "response_effectiveness": {}  # Will be populated by brain feedback
        }
        return patterns
    
    def _analyze_time_spans(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time patterns in memories"""
        if not memories:
            return {"active_hours": [], "frequency": "unknown"}
        
        timestamps = [mem.get("timestamp", 0) for mem in memories if mem.get("timestamp")]
        if not timestamps:
            return {"active_hours": [], "frequency": "unknown"}
        
        # Convert to hours
        hours = [datetime.fromtimestamp(ts).hour for ts in timestamps if ts > 0]
        
        # Find most active hours
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        most_active = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "active_hours": [hour for hour, count in most_active],
            "total_interactions": len(timestamps),
            "frequency": "frequent" if len(timestamps) > 10 else "occasional"
        }
    
    def _identify_topic_clusters(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Identify topic clusters from memories"""
        topics = {}
        
        for memory in memories:
            tags = memory.get("tags", [])
            content = memory.get("content", "").lower()
            
            # Count topic indicators
            for tag in tags:
                if tag not in ["user", "assistant", "positive", "negative", "neutral"]:
                    topics[tag] = topics.get(tag, 0) + 1
            
            # Look for technical topics
            if any(word in content for word in ["python", "code", "programming", "debug"]):
                topics["technical"] = topics.get("technical", 0) + 1
            if any(word in content for word in ["plan", "organize", "schedule"]):
                topics["planning"] = topics.get("planning", 0) + 1
            if any(word in content for word in ["learn", "study", "education"]):
                topics["learning"] = topics.get("learning", 0) + 1
        
        # Return top topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def get_conversation_context(self, conversation_id: str = None) -> Dict[str, Any]:
        """Get context for current conversation"""
        if not self.simple_memory:
            return {}
        
        history = self.simple_memory.get_conversation_history(conversation_id)
        return {
            "conversation_length": len(history),
            "recent_topics": self._extract_recent_topics(history),
            "emotional_trend": self._analyze_emotional_trend(history),
            "active_tags": self._extract_active_tags(history)
        }
    
    def _extract_recent_topics(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract recent topics from conversation history"""
        topics = []
        for entry in history[-10:]:  # Last 10 entries
            content = entry.get("content", "").lower()
            if "python" in content or "code" in content:
                topics.append("technical")
            elif "plan" in content or "organize" in content:
                topics.append("planning")
            elif "learn" in content or "study" in content:
                topics.append("learning")
        
        return list(set(topics))
    
    def _analyze_emotional_trend(self, history: List[Dict[str, Any]]) -> str:
        """Analyze emotional trend in conversation"""
        if not history:
            return "neutral"
        
        emotions = []
        for entry in history[-5:]:  # Last 5 entries
            tags = entry.get("tags", [])
            if "positive" in tags:
                emotions.append(1)
            elif "negative" in tags:
                emotions.append(-1)
            else:
                emotions.append(0)
        
        avg_emotion = sum(emotions) / len(emotions) if emotions else 0
        if avg_emotion > 0.3:
            return "positive"
        elif avg_emotion < -0.3:
            return "negative"
        else:
            return "neutral"
    
    def _extract_active_tags(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract currently active tags"""
        all_tags = []
        for entry in history[-10:]:  # Last 10 entries
            all_tags.extend(entry.get("tags", []))
        
        # Count tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Return top tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:5]]
    
    def cleanup_old_memories(self, days_old: int = 30):
        """Clean up old memories to manage storage"""
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        for conv_file in self.memory_dir.glob("conv_*.json"):
            try:
                # Check if file is old
                if conv_file.stat().st_mtime < cutoff_time:
                    # Archive instead of delete
                    archive_dir = self.memory_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    
                    # Move to archive
                    archive_file = archive_dir / conv_file.name
                    conv_file.rename(archive_file)
                    self.archived_conversations.add(archive_file.name)
                    
                    print(f"📦 Archived old conversation: {conv_file.name}")
            except Exception as e:
                print(f"⚠️  Could not archive {conv_file.name}: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.simple_memory:
            return {"error": "Simple memory not connected"}
        
        simple_stats = self.simple_memory.get_memory_stats()
        
        return {
            "simple_memory_stats": simple_stats,
            "cached_memories": len(self.memory_cache),
            "archived_conversations": len(self.archived_conversations),
            "personal_context_size": len(self.personal_context),
            "connected_simple_memory": self.simple_memory is not None
        }
