#/storage/emulated/0/Vxt/Vxt/vera_xt/memory/simple_memory.py
#!/usr/bin/env python3
"""
Simple Memory System - Smart storage without scanning
Each memory knows its own context, no file scanning needed
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class SimpleMemory:
    def __init__(self, memory_dir: str = "Memory_Data"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Current conversation tracking
        self.current_conversation_id = None
        self.conversation_files = {}  # Cache conversation files
        
        # Memory categorization (learned, not hardcoded)
        self.known_categories = set()
        self.category_files = {}  # category -> file mapping
        
        print("🧠 Simple Memory System initialized")
        print("💡 No file scanning - each memory knows its own context")
    
    def create_conversation(self, topic: str = "general") -> str:
        """Create a new conversation file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_id = f"conv_{timestamp}_{topic.replace(' ', '_')[:20]}"
        filename = f"{conv_id}.json"
        
        # Create conversation file
        conv_file = self.memory_dir / filename
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation_id": conv_id,
                "topic": topic,
                "created_at": timestamp,
                "entries": []
            }, f, indent=2)
        
        self.current_conversation_id = conv_id
        self.conversation_files[conv_id] = conv_file
        
        return conv_id
    
    def add_memory(self, content: str, memory_type: str = "conversation", tags: List[str] = None, 
                   context: Dict[str, Any] = None, conversation_id: str = None) -> bool:
        """Add a memory with its own context tags"""
        if tags is None:
            tags = []
        if context is None:
            context = {}
        
        # Add to current conversation or create new one
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            conv_id = self.create_conversation("general")
        
        # Update known categories
        for tag in tags:
            self.known_categories.add(tag)
        
        # Create memory entry with its own context
        memory_entry = {
            "id": f"mem_{int(time.time())}_{len(str(content))}",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": memory_type,
            "content": content,
            "tags": tags,
            "context": context,
            "conversation_id": conv_id
        }
        
        # Save to appropriate conversation file
        conv_file = self.conversation_files.get(conv_id)
        if not conv_file:
            # Find the conversation file
            for f in self.memory_dir.glob(f"conv_{conv_id}*.json"):
                conv_file = f
                break
        
        if conv_file and conv_file.exists():
            # Load existing conversation
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            # Add new entry
            conversation["entries"].append(memory_entry)
            
            # Save back
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2)
            
            return True
        else:
            # Create new conversation file
            return self._create_new_conversation_with_entry(conv_id, memory_entry)
    
    def _create_new_conversation_with_entry(self, conv_id: str, entry: Dict[str, Any]) -> bool:
        """Create a new conversation and add the entry"""
        timestamp = conv_id.split('_')[1] + '_' + conv_id.split('_')[2]  # Extract time part
        topic = '_'.join(conv_id.split('_')[3:]).replace('_', ' ')
        
        conversation = {
            "conversation_id": conv_id,
            "topic": topic,
            "created_at": timestamp,
            "entries": [entry]
        }
        
        filename = f"{conv_id}.json"
        conv_file = self.memory_dir / filename
        
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2)
        
        self.conversation_files[conv_id] = conv_file
        if not self.current_conversation_id:
            self.current_conversation_id = conv_id
        
        return True
    
    def find_memories_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Find memories by tags without scanning all files"""
        results = []
        
        # Look in recent conversation files first
        for conv_file in list(self.conversation_files.values())[-5:]:  # Last 5 conversations
            if conv_file.exists():
                with open(conv_file, 'r', encoding='utf-8') as f:
                    try:
                        conversation = json.load(f)
                        for entry in conversation.get("entries", []):
                            if any(tag in entry.get("tags", []) for tag in tags):
                                results.append(entry)
                                if len(results) >= limit:
                                    return results
                    except:
                        continue
        
        # If not enough results, search other files
        for conv_file in self.memory_dir.glob("conv_*.json"):
            if conv_file not in self.conversation_files.values() and len(results) < limit:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    try:
                        conversation = json.load(f)
                        for entry in conversation.get("entries", []):
                            if any(tag in entry.get("tags", []) for tag in tags):
                                results.append(entry)
                                if len(results) >= limit:
                                    return results
                    except:
                        continue
        
        return results[:limit]
    
    def find_memories_by_content(self, search_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find memories by content without scanning all files"""
        results = []
        search_lower = search_text.lower()
        
        # Search recent conversations first
        for conv_file in list(self.conversation_files.values())[-5:]:
            if conv_file.exists():
                with open(conv_file, 'r', encoding='utf-8') as f:
                    try:
                        conversation = json.load(f)
                        for entry in conversation.get("entries", []):
                            if search_lower in entry.get("content", "").lower():
                                results.append(entry)
                                if len(results) >= limit:
                                    return results
                    except:
                        continue
        
        # Search other files if needed
        for conv_file in self.memory_dir.glob("conv_*.json"):
            if conv_file not in self.conversation_files.values() and len(results) < limit:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    try:
                        conversation = json.load(f)
                        for entry in conversation.get("entries", []):
                            if search_lower in entry.get("content", "").lower():
                                results.append(entry)
                                if len(results) >= limit:
                                    return results
                    except:
                        continue
        
        return results[:limit]
    
    def get_recent_memories(self, hours: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories without scanning all files"""
        results = []
        time_cutoff = time.time() - (hours * 3600)
        
        # Check recent conversation files
        for conv_file in list(self.conversation_files.values())[-10:]:  # Last 10 conversations
            if conv_file.exists():
                with open(conv_file, 'r', encoding='utf-8') as f:
                    try:
                        conversation = json.load(f)
                        for entry in conversation.get("entries", []):
                            if entry["timestamp"] > time_cutoff:
                                results.append(entry)
                                if len(results) >= limit:
                                    return results
                    except:
                        continue
        
        return results[:limit]
    
    def get_conversation_history(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            return []
        
        # Look for conversation file
        conv_file = self.conversation_files.get(conv_id)
        if not conv_file:
            for f in self.memory_dir.glob(f"conv_{conv_id}*.json"):
                conv_file = f
                break
        
        if conv_file and conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                try:
                    conversation = json.load(f)
                    return conversation.get("entries", [])
                except:
                    return []
        
        return []
    
    def categorize_memory(self, memory_entry: Dict[str, Any]) -> str:
        """Dynamically categorize memory based on content and tags"""
        content = memory_entry.get("content", "").lower()
        tags = memory_entry.get("tags", [])
        
        # Use learned categories
        if tags:
            for tag in tags:
                if tag in self.known_categories:
                    return tag
        
        # Default categories based on content
        technical_indicators = ["code", "python", "function", "debug", "error", "program", "script", "algorithm"]
        personal_indicators = ["i", "my", "me", "personal", "private", "own"]
        planning_indicators = ["plan", "organize", "schedule", "arrange", "todo", "task"]
        
        if any(ind in content for ind in technical_indicators):
            category = "technical"
        elif any(ind in content for ind in personal_indicators):
            category = "personal"
        elif any(ind in content for ind in planning_indicators):
            category = "planning"
        else:
            category = "general"
        
        # Learn this category
        self.known_categories.add(category)
        return category
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_files = len(list(self.memory_dir.glob("conv_*.json")))
        total_memories = 0
        
        for conv_file in self.memory_dir.glob("conv_*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                    total_memories += len(conversation.get("entries", []))
            except:
                continue
        
        return {
            "total_conversations": total_files,
            "total_memories": total_memories,
            "known_categories": list(self.known_categories),
            "current_conversation": self.current_conversation_id,
            "cached_conversations": len(self.conversation_files)
        }
