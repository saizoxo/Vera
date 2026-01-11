#/storage/emulated/0/Vxt/Vxt/vera_xt/core/memory_handler.py
#!/usr/bin/env python3
"""
Memory Handler Module - Manages 5D memory system (Semantic, Temporal, Affective, Causal, Project)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class MemoryHandler:
    def __init__(self, memory_dir: str = "Memory_Data"):
        self.memory_dir = Path(memory_dir)
        self.workspace_dir = Path("Workspace")
        
        # Create directories if they don't exist
        self.memory_dir.mkdir(exist_ok=True)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Create memory folder structure
        self.conversations_dir = self.memory_dir / "conversations"
        self.knowledge_dir = self.memory_dir / "knowledge"
        self.skills_dir = self.memory_dir / "skills"
        self.contexts_dir = self.memory_dir / "contexts"
        self.personal_dir = self.memory_dir / "personal"
        
        for dir_path in [self.conversations_dir, self.knowledge_dir, self.skills_dir, self.contexts_dir, self.personal_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Human-like memory systems
        self.short_term_memory = []  # Current conversation
        self.long_term_memory = {}   # Persistent memories
        self.context_awareness = {}  # Current situation context
        self.learned_patterns = {}   # Patterns learned from interactions
        
        # Current session tracking
        self.current_conversation_id = None
        self.current_context = {}
        
        print("🧠 Memory Handler initialized with 5D memory system")
        print("📁 Folders created:")
        print(f"   - Conversations: {self.conversations_dir}")
        print(f"   - Knowledge: {self.knowledge_dir}")
        print(f"   - Skills: {self.skills_dir}")
        print(f"   - Contexts: {self.contexts_dir}")
        print(f"   - Personal: {self.personal_dir}")
    
    def create_conversation(self, topic: str = "general") -> str:
        """Create a new conversation memory file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_id = f"conv_{timestamp}_{topic.replace(' ', '_')[:20]}"
        
        # Create conversation file
        conv_file = self.conversations_dir / f"{conv_id}.json"
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation_id": conv_id,
                "topic": topic,
                "created_at": timestamp,
                "entries": [],
                "metadata": {
                    "importance": 5,  # 1-10 scale
                    "tags": [topic],
                    "context": {}
                }
            }, f, indent=2)
        
        self.current_conversation_id = conv_id
        return conv_id
    
    def add_to_memory(self, entry_type: str, content: str, context: Dict[str, Any] = None, category: str = "general"):
        """Add entry to appropriate memory folder with rich 5D context"""
        if context is None:
            context = {}
        
        # Analyze the content to extract rich context
        enhanced_context = self._analyze_content_for_context(content, context)
        
        # Create memory entry with 5D dimensions
        memory_entry = {
            "id": f"mem_{int(time.time())}_{len(content)}",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": entry_type,
            "content": content,
            "category": category,
            "importance": enhanced_context.get("importance", 3),
            "semantic_tags": enhanced_context.get("semantic_tags", []),
            "emotional_tone": enhanced_context.get("emotional_tone", 0.0),
            "causal_links": enhanced_context.get("causal_links", []),
            "related_memories": enhanced_context.get("related_memories", []),
            "project_context": enhanced_context.get("project_context", "general"),
            "context": enhanced_context
        }
        
        # Add to short-term memory
        self.short_term_memory.append(memory_entry)
        
        # Save to appropriate folder based on category
        self._save_to_category_folder(memory_entry, category)
        
        # If in conversation, also save to conversation file
        if self.current_conversation_id:
            self._save_to_conversation(memory_entry)
        
        # Update learned patterns
        self._update_learned_patterns(memory_entry)
    
    def _analyze_content_for_context(self, content: str, existing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content to extract rich contextual information"""
        enhanced_context = existing_context.copy()
        
        # Extract semantic tags (keywords and concepts)
        semantic_tags = self._extract_semantic_tags(content)
        enhanced_context["semantic_tags"] = semantic_tags
        
        # Analyze emotional tone
        emotional_tone = self._analyze_emotional_tone(content)
        enhanced_context["emotional_tone"] = emotional_tone
        
        # Determine importance level
        importance = self._determine_importance(content, emotional_tone)
        enhanced_context["importance"] = importance
        
        # Identify project context
        project_context = self._identify_project_context(content)
        enhanced_context["project_context"] = project_context
        
        # Find related memories
        related_memories = self._find_related_memories(content)
        enhanced_context["related_memories"] = related_memories
        
        # Analyze for causal relationships
        causal_links = self._analyze_causal_relationships(content)
        enhanced_context["causal_links"] = causal_links
        
        return enhanced_context
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags (keywords, concepts, entities)"""
        # Simple keyword extraction - would be enhanced with NLP in real system
        words = content.lower().split()
        tags = []
        
        # Technical terms
        technical_terms = ["python", "code", "programming", "function", "debug", "error", "algorithm", "data", "ai", "ml"]
        for term in technical_terms:
            if term in content.lower():
                tags.append(f"technical:{term}")
        
        # Emotional terms
        emotional_terms = ["happy", "excited", "frustrated", "confused", "good", "bad", "great", "terrible"]
        for term in emotional_terms:
            if term in content.lower():
                tags.append(f"emotional:{term}")
        
        # Time-related terms
        time_terms = ["today", "yesterday", "tomorrow", "week", "month", "year", "morning", "evening"]
        for term in time_terms:
            if term in content.lower():
                tags.append(f"temporal:{term}")
        
        # Add unique words as tags (length > 3)
        unique_words = list(set(word for word in words if len(word) > 3 and word.isalpha()))
        tags.extend([f"keyword:{word}" for word in unique_words[:10]])  # Limit to top 10
        
        return list(set(tags))  # Remove duplicates
    
    def _analyze_emotional_tone(self, content: str) -> float:
        """Analyze emotional tone (-1.0 to 1.0)"""
        positive_words = ["good", "great", "love", "like", "happy", "excited", "wonderful", "amazing", "excellent", "perfect"]
        negative_words = ["bad", "hate", "angry", "frustrated", "terrible", "awful", "hate", "dislike", "worst", "horrible"]
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _determine_importance(self, content: str, emotional_tone: float) -> int:
        """Determine importance level (1-10)"""
        importance = 5  # Base level
        
        # Increase importance for emotional content
        if abs(emotional_tone) > 0.5:
            importance += 2
        
        # Increase importance for longer content (more detailed)
        if len(content) > 100:
            importance += 1
        
        # Increase importance for specific keywords
        important_keywords = ["important", "critical", "urgent", "need", "help", "problem", "issue"]
        if any(keyword in content.lower() for keyword in important_keywords):
            importance += 2
        
        return max(1, min(10, importance))  # Clamp between 1-10
    
    def _identify_project_context(self, content: str) -> str:
        """Identify project context from content"""
        project_indicators = {
            "coding": ["code", "python", "function", "debug", "program", "script", "algorithm"],
            "planning": ["plan", "organize", "schedule", "arrange", "todo", "task"],
            "learning": ["learn", "study", "understand", "explain", "teach", "education"],
            "personal": ["my", "i", "me", "personal", "private", "own"]
        }
        
        content_lower = content.lower()
        for project, indicators in project_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                return project
        
        return "general"
    
    def _find_related_memories(self, content: str) -> List[str]:
        """Find related memories based on content"""
        # This would search through existing memories for related content
        # For now, return empty - would be enhanced with semantic search
        return []
    
    def _analyze_causal_relationships(self, content: str) -> List[str]:
        """Analyze content for causal relationships"""
        causal_indicators = ["because", "therefore", "so", "then", "if", "when", "since", "due to", "caused by"]
        content_lower = content.lower()
        
        causal_links = []
        for indicator in causal_indicators:
            if indicator in content_lower:
                causal_links.append(indicator)
        
        return causal_links
    
    def _update_learned_patterns(self, memory_entry: Dict[str, Any]):
        """Update learned patterns from this memory"""
        # Update word associations based on semantic tags
        for tag in memory_entry.get("semantic_tags", []):
            if ":" in tag:
                category, word = tag.split(":", 1)
                if category not in self.learned_patterns:
                    self.learned_patterns[category] = []
                if word not in self.learned_patterns[category]:
                    self.learned_patterns[category].append(word)
    
    def _save_to_category_folder(self, memory_entry: Dict[str, Any], category: str):
        """Save memory to appropriate category folder"""
        folder_map = {
            "conversation": self.conversations_dir,
            "knowledge": self.knowledge_dir,
            "skill": self.skills_dir,
            "context": self.contexts_dir,
            "personal": self.personal_dir,
            "general": self.conversations_dir  # Default to conversations
        }
        
        target_folder = folder_map.get(category, self.conversations_dir)
        
        # Create a file based on timestamp
        timestamp = datetime.fromtimestamp(memory_entry["timestamp"]).strftime("%Y%m%d_%H")
        filename = f"{category}_{timestamp}.json"
        file_path = target_folder / filename
        
        # Load existing entries or create new
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        else:
            entries = []
        
        # Add new entry
        entries.append(memory_entry)
        
        # Save back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
    
    def _save_to_conversation(self, memory_entry: Dict[str, Any]):
        """Save memory entry to current conversation file"""
        if not self.current_conversation_id:
            return
        
        conv_file = self.conversations_dir / f"{self.current_conversation_id}.json"
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            conversation["entries"].append(memory_entry)
            
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2)
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories from short-term memory"""
        return self.short_term_memory[-limit:]
    
    def retrieve_relevant_memories(self, input_text: str, category: str = None) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to current input from organized folders"""
        relevant_memories = []
        
        # Search in appropriate folders
        search_folders = []
        if category:
            folder_map = {
                "conversation": self.conversations_dir,
                "knowledge": self.knowledge_dir,
                "skill": self.skills_dir,
                "context": self.contexts_dir,
                "personal": self.personal_dir
            }
            if category in folder_map:
                search_folders.append(folder_map[category])
        else:
            # Search all folders
            search_folders = [self.conversations_dir, self.knowledge_dir, self.skills_dir, self.contexts_dir, self.personal_dir]
        
        # Search for relevant memories
        for folder in search_folders:
            for file_path in folder.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # If it's a conversation file
                        if isinstance(data, dict) and "entries" in data:
                            entries = data["entries"]
                        else:
                            entries = data  # If it's a list of entries
                        
                        if isinstance(entries, list):
                            for entry in entries:
                                if isinstance(entry, dict) and "content" in entry:
                                    content = entry["content"].lower()
                                    input_lower = input_text.lower()
                                    
                                    # Enhanced relevance check using semantic tags
                                    content_relevant = any(word in content for word in input_lower.split()[:5])
                                    tag_relevant = any(
                                        any(word in tag for word in input_lower.split()[:3])
                                        for tag in entry.get("semantic_tags", [])
                                    )
                                    
                                    if content_relevant or tag_relevant:
                                        relevant_memories.append(entry)
                except:
                    continue  # Skip files that can't be read
        
        return relevant_memories[-5:]  # Return last 5 relevant memories
    
    def get_memory_stats(self):
        """Get memory system statistics"""
        stats = {
            "short_term_memory_count": len(self.short_term_memory),
            "learned_patterns_count": len(self.learned_patterns),
            "current_conversation": self.current_conversation_id,
            "folder_stats": {}
        }
        
        # Count files in each folder
        for folder_name, folder_path in [
            ("conversations", self.conversations_dir),
            ("knowledge", self.knowledge_dir),
            ("skills", self.skills_dir),
            ("contexts", self.contexts_dir),
            ("personal", self.personal_dir)
        ]:
            file_count = len(list(folder_path.glob("*.json")))
            stats["folder_stats"][folder_name] = file_count
        
        return stats