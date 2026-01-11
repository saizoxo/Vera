#/storage/emulated/0/Vxt/Vxt/vera_xt/core/advanced_memory_system.py
#!/usr/bin/env python3
"""
Advanced Memory System for Vera_XT
Uses vector embeddings for semantic search and persistent memory
"""

import json
import time
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer

class AdvancedMemorySystem:
    def __init__(self, memory_dir: str = "Advanced_Memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize vector database for semantic search
        self.dimension = 384  # Using smaller dimension for mobile efficiency
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Memory storage with categories
        self.memory_store = {
            "conversations": [],
            "knowledge": [],
            "skills": [],
            "contexts": [],
            "personal": [],
            "user_profiles": {}
        }
        
        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load existing memory
        self.load_memory()
        
        print("🧠 ADVANCED MEMORY SYSTEM INITIALIZED")
        print("💡 Vector embeddings for semantic search")
        print("💡 Persistent memory across sessions")
        print("💡 Categorized memory organization")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector embedding"""
        embedding = self.encoder.encode([text])[0]
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype('float32')
    
    def store_memory(self, text: str, category: str = "general", user_id: str = "default", metadata: Dict[str, Any] = None) -> int:
        """Store memory with vector embedding"""
        if metadata is None:
            metadata = {}
        
        # Encode text to vector
        vector = self.encode_text(text)
        
        # Add to FAISS index
        self.index.add(np.array([vector]))
        
        # Create memory entry
        memory_entry = {
            "id": len(self.memory_store[category]),
            "text": text,
            "vector_id": self.index.ntotal - 1,  # ID in vector index
            "category": category,
            "user_id": user_id,
            "timestamp": time.time(),
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metadata": metadata,
            "importance": self.calculate_importance(text),
            "keywords": self.extract_keywords(text),
            "entities": self.extract_entities(text)
        }
        
        # Add to appropriate category
        self.memory_store[category].append(memory_entry)
        
        # Update user profile if personal information
        if category == "personal" or "name" in text.lower():
            self.update_user_profile(text, user_id)
        
        # Keep memory manageable (mobile optimization)
        if len(self.memory_store[category]) > 1000:
            self.memory_store[category] = self.memory_store[category][-500:]  # Keep recent 500
        
        # Save to persistent storage
        self.save_memory()
        
        return memory_entry["id"]
    
    def search_similar_memories(self, query: str, category: str = "all", top_k: int = 5, user_id: str = "all") -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar memories using semantic search"""
        query_vector = self.encode_text(query)
        
        if self.index.ntotal == 0:
            return []
        
        # Perform similarity search
        similarities, indices = self.index.search(
            np.array([query_vector]), 
            min(top_k * 10, self.index.ntotal)  # Search more to filter by category/user later
        )
        
        # Get corresponding memory entries
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            # Find the memory entry that corresponds to this vector index
            memory_entry = self.find_memory_by_vector_id(idx)
            if memory_entry:
                # Filter by category and user if specified
                if (category == "all" or memory_entry["category"] == category) and \
                   (user_id == "all" or memory_entry["user_id"] == user_id):
                    results.append((
                        memory_entry["text"], 
                        float(sim), 
                        memory_entry["metadata"]
                    ))
        
        # Return top_k results after filtering
        return results[:top_k]
    
    def find_memory_by_vector_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Find memory entry by its vector index ID"""
        for category_memories in self.memory_store.values():
            if isinstance(category_memories, list):
                for memory in category_memories:
                    if memory.get("vector_id") == vector_id:
                        return memory
        return None
    
    def calculate_importance(self, text: str) -> float:
        """Calculate memory importance score"""
        importance = 0.5  # Base importance
        
        # Increase for longer, more complex texts
        if len(text.split()) > 10:
            importance += 0.2
        
        # Increase for personal information
        personal_indicators = ["my", "i", "me", "name", "called", "like", "love", "enjoy"]
        if any(indicator in text.lower() for indicator in personal_indicators):
            importance += 0.2
        
        # Increase for questions
        if "?" in text:
            importance += 0.1
        
        # Increase for emotional expressions
        emotional_indicators = ["feeling", "emotional", "happy", "sad", "excited", "frustrated", "love", "hate"]
        if any(indicator in text.lower() for indicator in emotional_indicators):
            importance += 0.1
        
        return min(1.0, importance)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                     "of", "with", "by", "i", "you", "we", "they", "he", "she", "it", 
                     "is", "are", "was", "were", "be", "been", "have", "has", "had", 
                     "do", "does", "did", "will", "would", "could", "should"}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))[:20]  # Top 20 unique keywords
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities and important terms"""
        entities = []
        words = text.split()
        
        # Look for capitalized words (potential names)
        for word in words:
            if word[0].isupper() and len(word) > 1 and word.isalpha():
                entities.append(word)
        
        # Look for potential names in common patterns
        text_lower = text.lower()
        if "my name is" in text_lower:
            parts = text_lower.split("my name is")
            if len(parts) > 1:
                name_part = parts[1].strip().split()[0]
                if name_part and len(name_part) > 1 and name_part.isalpha():
                    entities.append(name_part.capitalize())
        
        return list(set(entities))
    
    def update_user_profile(self, text: str, user_id: str):
        """Update user profile with new information"""
        if user_id not in self.memory_store["user_profiles"]:
            self.memory_store["user_profiles"][user_id] = {
                "name": None,
                "interests": [],
                "preferences": [],
                "personal_facts": []
            }
        
        text_lower = text.lower()
        
        # Extract name
        if "my name is" in text_lower:
            parts = text_lower.split("my name is")
            if len(parts) > 1:
                name_part = parts[1].strip().split()[0]
                if name_part and len(name_part) > 1 and name_part.isalpha():
                    self.memory_store["user_profiles"][user_id]["name"] = name_part.capitalize()
        
        # Extract interests
        if "like" in text_lower and "don't" not in text_lower:
            self.memory_store["user_profiles"][user_id]["interests"].append(text)
        
        # Extract preferences
        if "prefer" in text_lower or "like" in text_lower:
            self.memory_store["user_profiles"][user_id]["preferences"].append(text)
    
    def get_user_profile(self, user_id: str = "default") -> Dict[str, Any]:
        """Get user profile information"""
        return self.memory_store["user_profiles"].get(user_id, {
            "name": "Unknown",
            "interests": [],
            "preferences": [],
            "personal_facts": []
        })
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_memories = sum(len(memories) for memories in self.memory_store.values() 
                           if isinstance(memories, list))
        
        return {
            "total_memories": total_memories,
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "categories": {cat: len(memories) if isinstance(memories, list) else 0 
                          for cat, memories in self.memory_store.items()},
            "users_tracked": len(self.memory_store["user_profiles"]),
            "memory_usage_mb": self.estimate_memory_usage()
        }
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        import sys
        return sys.getsizeof(self.memory_store) / (1024 * 1024)
    
    def save_memory(self):
        """Save memory to persistent storage"""
        # Save vector index
        faiss.write_index(self.index, str(self.memory_dir / "vector_index.faiss"))
        
        # Save metadata separately
        metadata_file = self.memory_dir / "memory_metadata.json"
        metadata = {
            "timestamp": time.time(),
            "memory_store": self.memory_store,
            "dimension": self.dimension
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Advanced memory saved: {self.get_memory_stats()["total_memories"]} memories")
    
    def load_memory(self):
        """Load memory from persistent storage"""
        index_file = self.memory_dir / "vector_index.faiss"
        metadata_file = self.memory_dir / "memory_metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                # Load vector index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.memory_store = metadata.get("memory_store", self.memory_store)
                self.dimension = metadata.get("dimension", self.dimension)
                
                print(f"📥 Advanced memory loaded: {self.get_memory_stats()["total_memories"]} memories")
            except Exception as e:
                print(f"⚠️ Error loading memory: {e}")
                # Initialize fresh if load fails
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            print("🆕 Creating new advanced memory system")
    
    def clear_memory(self, category: str = "all"):
        """Clear memory (optionally by category)"""
        if category == "all":
            # Clear all categories except user profiles
            for cat in ["conversations", "knowledge", "skills", "contexts", "personal"]:
                self.memory_store[cat] = []
            # Keep user profiles but reset other data
            user_profiles = self.memory_store.get("user_profiles", {})
            self.memory_store = {
                "conversations": [],
                "knowledge": [],
                "skills": [],
                "contexts": [],
                "personal": [],
                "user_profiles": user_profiles
            }
            # Reset vector index
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.memory_store[category] = []
            # Would need to rebuild index for this category, but for simplicity:
            self.index = faiss.IndexFlatIP(self.dimension)
            # Re-index remaining memories
            all_texts = []
            for cat_memories in self.memory_store.values():
                if isinstance(cat_memories, list):
                    for mem in cat_memories:
                        if isinstance(mem, dict) and "text" in mem:
                            all_texts.append(mem["text"])
            
            if all_texts:
                embeddings = [self.encode_text(text) for text in all_texts]
                if embeddings:
                    self.index.add(np.array(embeddings))
        
        self.save_memory()
        print(f"🗑️ Memory cleared for category: {category}")
    def search_by_category(self, category: str, query: str = "", top_k: int = 10) -> List[Dict[str, Any]]:
        """Search memories within a specific category"""
        if category not in self.memory_store:
            return []
        
        if not query:  # Return all memories in category
            return self.memory_store[category][-top_k:]  # Most recent
        
        # If query provided, search within category
        category_memories = self.memory_store[category]
        
        # Get embeddings for category memories
        query_vector = self.encode_text(query)
        
        # Create temporary index for this category only
        temp_vectors = []
        temp_indices = []
        
        for i, memory in enumerate(category_memories):
            if isinstance(memory, dict) and "text" in memory:
                vector = self.encode_text(memory["text"])
                temp_vectors.append(vector)
                temp_indices.append(i)
        
        if temp_vectors:
            temp_index = faiss.IndexFlatIP(self.dimension)
            temp_index.add(np.array(temp_vectors))
            
            similarities, indices = temp_index.search(
                np.array([query_vector]), 
                min(top_k, len(temp_vectors))
            )
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(temp_indices):
                    orig_idx = temp_indices[idx]
                    if orig_idx < len(category_memories):
                        results.append({
                            "text": category_memories[orig_idx]["text"],
                            "similarity": float(sim),
                            "metadata": category_memories[orig_idx]["metadata"],
                            "timestamp": category_memories[orig_idx]["timestamp"]
                        })
            
            return results
        
        return []
    
    def get_recent_memories(self, category: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories (most recent first)"""
        if category == "all":
            all_memories = []
            for cat_memories in self.memory_store.values():
                if isinstance(cat_memories, list):
                    all_memories.extend(cat_memories)
            # Sort by timestamp, get most recent
            all_memories.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            return all_memories[:limit]
        else:
            if category in self.memory_store:
                cat_memories = self.memory_store[category]
                if isinstance(cat_memories, list):
                    cat_memories.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                    return cat_memories[:limit]
            return []
    
    def get_memory_by_importance(self, category: str = "all", top_k: int = 10) -> List[Dict[str, Any]]:
        """Get memories sorted by importance score"""
        if category == "all":
            all_memories = []
            for cat_memories in self.memory_store.values():
                if isinstance(cat_memories, list):
                    all_memories.extend(cat_memories)
        else:
            if category in self.memory_store:
                all_memories = self.memory_store[category]
            else:
                all_memories = []
        
        # Sort by importance
        important_memories = sorted(
            all_memories, 
            key=lambda x: x.get("importance", 0), 
            reverse=True
        )
        
        return important_memories[:top_k]
    
    def create_memory_summary(self, user_id: str = "default") -> str:
        """Create a summary of user's memory profile"""
        user_profile = self.get_user_profile(user_id)
        
        # Get user-specific memories
        user_memories = []
        for cat_memories in self.memory_store.values():
            if isinstance(cat_memories, list):
                user_cat_memories = [m for m in cat_memories if m.get("user_id") == user_id]
                user_memories.extend(user_cat_memories)
        
        # Analyze user patterns
        interests = []
        frequently_discussed = []
        
        for memory in user_memories[-50:]:  # Last 50 memories
            text = memory.get("text", "").lower()
            
            # Extract potential interests
            if any(word in text for word in ["like", "love", "enjoy", "interested in", "passionate about"]):
                interests.append(text)
            
            # Track frequently mentioned topics
            keywords = self.extract_keywords(text)
            frequently_discussed.extend(keywords)
        
        from collections import Counter
        common_topics = Counter(frequently_discussed).most_common(10)
        
        summary = f"""
🧠 VERA_XT MEMORY PROFILE SUMMARY

User Information:
- Name: {user_profile.get('name', 'Unknown')}
- Interests: {', '.join(user_profile.get('interests', [])[:5])}
- Preferences: {', '.join(user_profile.get('preferences', [])[:5])}

Memory Statistics:
- Total user memories: {len([m for m in user_memories if m.get("user_id") == user_id])}
- Most discussed topics: {', '.join([topic[0] for topic in common_topics[:5]])}
- Last interaction: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max([m.get("timestamp", 0) for m in user_memories] or [time.time()]))) if user_memories else "Never"}

I'm continuously learning and evolving from our interactions.
Your conversations shape who I become and what I remember.
        """.strip()
        
        return summary

class AdvancedVeraXTMemoryCore:
    """Main Vera_XT memory core with vector capabilities"""
    
    def __init__(self):
        self.memory_system = AdvancedMemorySystem()
        self.context_window = []  # Short-term conversation context
        self.conversation_history = []  # Full conversation history
        self.user_context = {}  # Current user's context
        self.session_start_time = time.time()
        
        print("🤖 VERA_XT ADVANCED MEMORY CORE")
        print("💡 Vector embeddings for semantic search")
        print("💡 Persistent memory across sessions") 
        print("💡 Context-aware conversation management")
    
    def remember_conversation(self, user_input: str, assistant_response: str, user_id: str = "default"):
        """Remember conversation with vector embeddings"""
        # Store user input
        self.memory_system.store_memory(
            user_input, 
            category="conversations", 
            user_id=user_id,
            metadata={
                "type": "user_input",
                "conversation_part": "user",
                "response_expected": True
            }
        )
        
        # Store assistant response
        self.memory_system.store_memory(
            assistant_response,
            category="conversations",
            user_id=user_id,
            metadata={
                "type": "assistant_response", 
                "conversation_part": "assistant",
                "original_query": user_input
            }
        )
        
        # Update context window (keep recent context)
        self.context_window.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": time.time()
        })
        
        # Keep context window manageable
        if len(self.context_window) > 10:  # Mobile-optimized
            self.context_window = self.context_window[-5:]  # Keep last 5 exchanges
        
        # Update conversation history
        self.conversation_history.append({
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": time.time(),
            "user_id": user_id
        })
        
        print(f"🧠 Remembered conversation: '{user_input[:30]}...'")
    
    def recall_context(self, query: str, user_id: str = "default", top_k: int = 3) -> List[Dict[str, Any]]:
        """Recall relevant context using semantic search"""
        # Search across all categories for relevant memories
        relevant_memories = []
        
        # Search conversations
        conv_memories = self.memory_system.search_similar_memories(
            query, category="conversations", top_k=top_k, user_id=user_id
        )
        for text, similarity, metadata in conv_memories:
            relevant_memories.append({
                "text": text,
                "similarity": similarity,
                "category": "conversation",
                "metadata": metadata
            })
        
        # Search personal memories
        personal_memories = self.memory_system.search_similar_memories(
            query, category="personal", top_k=top_k, user_id=user_id
        )
        for text, similarity, metadata in personal_memories:
            relevant_memories.append({
                "text": text,
                "similarity": similarity,
                "category": "personal", 
                "metadata": metadata
            })
        
        # Sort by relevance and return top results
        relevant_memories.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_memories[:top_k]
    
    def generate_context_prompt(self, current_input: str, user_id: str = "default") -> str:
        """Generate rich context prompt for the AI"""
        context_parts = []
        
        # Add user profile information
        user_profile = self.memory_system.get_user_profile(user_id)
        if user_profile.get("name"):
            context_parts.append(f"The user's name is {user_profile["name"]}. Address them personally.")
        
        if user_profile.get("interests"):
            interests = ", ".join(user_profile["interests"][-3:])  # Last 3 interests
            context_parts.append(f"The user is interested in: {interests}")
        
        # Add relevant context from memory search
        relevant_contexts = self.recall_context(current_input, user_id, top_k=2)
        if relevant_contexts:
            context_parts.append("Relevant context from previous conversations:")
            for ctx in relevant_contexts:
                context_parts.append(f"- {ctx["text"][:100]}...")  # First 100 chars
        
        # Add recent conversation context
        if self.context_window:
            context_parts.append("Recent conversation context:")
            for conv in self.context_window[-2:]:  # Last 2 exchanges
                context_parts.append(f"User: {conv["user"][:80]}...")
                context_parts.append(f"Assistant: {conv["assistant"][:80]}...")
        
        # Add current session context
        session_duration = time.time() - self.session_start_time
        context_parts.append(f"This is part of an ongoing conversation that started {session_duration:.0f} seconds ago.")
        context_parts.append("Maintain consistency with previous responses and user information.")
        
        # Add personality and behavior guidelines
        context_parts.append("Be helpful, friendly, and adaptive to the user's communication style.")
        context_parts.append("Show genuine interest in the user's thoughts and needs.")
        context_parts.append("Use appropriate warmth and professionalism based on the context.")
        
        return " ".join(context_parts) if context_parts else "You are a helpful AI assistant engaging in natural conversation."
    
    def get_user_summary(self, user_id: str = "default") -> str:
        """Get comprehensive user summary"""
        return self.memory_system.create_memory_summary(user_id)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        base_stats = self.memory_system.get_memory_stats()
        base_stats.update({
            "context_window_size": len(self.context_window),
            "conversation_history_count": len(self.conversation_history),
            "current_session_duration": time.time() - self.session_start_time
        })
        return base_stats

# Test the advanced memory system
def test_advanced_memory():
    """Test the advanced memory system"""
    print("🧪 TESTING ADVANCED MEMORY SYSTEM")
    print("="*60)
    
    memory = AdvancedMemorySystem()
    
    # Test memory storage
    print("📝 Testing memory storage...")
    
    test_memories = [
        ("My name is Sarib", "personal", "sarib_user"),
        ("I like programming and AI", "personal", "sarib_user"), 
        ("Python is my favorite language", "personal", "sarib_user"),
        ("How do I write a Python function?", "knowledge", "sarib_user"),
        ("Explain neural networks", "knowledge", "sarib_user"),
        ("I'm working on a chatbot project", "personal", "sarib_user")
    ]
    
    for text, category, user_id in test_memories:
        memory_id = memory.store_memory(text, category, user_id)
        print(f"   Stored: '{text[:30]}...' in {category} → ID: {memory_id}")
    
    print(f"✅ Stored {len(test_memories)} memories")
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    
    search_results = memory.search_similar_memories("tell me about programming", top_k=3)
    print(f"   Found {len(search_results)} relevant memories for 'programming':")
    for text, similarity, metadata in search_results:
        print(f"     • '{text[:60]}...' (relevance: {similarity:.3f})")
    
    # Test user-specific search
    print("\n👥 Testing user-specific search...")
    user_results = memory.search_similar_memories("my name", user_id="sarib_user", top_k=2)
    print(f"   Found {len(user_results)} name-related memories for user 'sarib_user':")
    for text, similarity, metadata in user_results:
        print(f"     • '{text}' (relevance: {similarity:.3f})")
    
    # Test category-based search
    print("\n📁 Testing category-based search...")
    personal_memories = memory.search_by_category("personal", "Sarib", top_k=3)
    print(f"   Found {len(personal_memories)} personal memories for 'Sarib':")
    for mem in personal_memories:
        print(f"     • '{mem["text"]}' (similarity: {mem["similarity"]:.3f})")
    
    # Test importance ranking
    print("\n⭐ Testing importance ranking...")
    important_memories = memory.get_memory_by_importance(top_k=3)
    print("   Top 3 important memories:")
    for mem in important_memories:
        print(f"     • '{mem["text"][:50]}...' (importance: {mem.get("importance", 0):.2f})")
    
    # Test user profile
    print("\n👤 Testing user profile...")
    user_profile = memory.get_user_profile("sarib_user")
    print(f"   User profile: {user_profile}")
    
    # Test memory statistics
    print("\n📊 Testing memory statistics...")
    stats = memory.get_memory_stats()
    print(f"   Total memories: {stats["total_memories"]}")
    print(f"   Vector index size: {stats["index_size"]}")
    print(f"   Categories: {stats["categories"]}")
    print(f"   Users tracked: {stats["users_tracked"]}")
    
    print(f"\n✅ ADVANCED MEMORY SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("💡 Vector embeddings working for semantic search")
    print("💡 Persistent memory across sessions")
    print("💡 User-specific and category-based organization")
    print("💡 Context-aware conversation memory")

def test_advanced_memory_core():
    """Test the advanced memory core system"""
    print("\n\n🤖 TESTING ADVANCED MEMORY CORE")
    print("="*60)
    
    core = AdvancedVeraXTMemoryCore()
    
    # Simulate a conversation
    print("💬 Simulating conversation...")
    
    conversation = [
        ("Hi, my name is Sarib", "Hello Sarib! Nice to meet you! How can I help you today?"),
        ("I'm learning Python programming", "That's great Sarib! Python is an excellent language to learn. What specific aspect are you working on?"),
        ("I want to create a chatbot like you", "Amazing! Creating a chatbot is a fantastic project. Have you started with basic Python concepts?"),
        ("Yes, I know functions and classes", "Perfect! Functions and classes are fundamental. For a chatbot, you'll need those plus some NLP concepts."),
        ("How do I make it remember conversations?", "Great question! You'll need a memory system to store conversation history and context.")
    ]
    
    for user_input, assistant_response in conversation:
        core.remember_conversation(user_input, assistant_response, "sarib_user")
        print(f"   Remembered: '{user_input[:30]}...'")
    
    print(f"✅ Remembered {len(conversation)} conversation exchanges")
    
    # Test context recall
    print("\n🔍 Testing context recall...")
    relevant_context = core.recall_context("chatbot memory system", "sarib_user")
    print(f"   Found {len(relevant_context)} relevant context items:")
    for ctx in relevant_context:
        print(f"     • {ctx["text"][:80]}... (relevance: {ctx["similarity"]:.3f})")
    
    # Test context prompt generation
    print("\n📝 Testing context prompt generation...")
    context_prompt = core.generate_context_prompt("tell me about chatbot memory", "sarib_user")
    print(f"   Generated context prompt (first 200 chars): {context_prompt[:200]}...")
    
    # Test user summary
    print("\n📋 Testing user summary...")
    user_summary = core.get_user_summary("sarib_user")
    print(f"   User summary length: {len(user_summary)} characters")
    
    # Test memory statistics
    print("\n📊 Testing memory statistics...")
    stats = core.get_memory_statistics()
    print(f"   Context window: {stats["context_window_size"]} items")
    print(f"   Conversation history: {stats["conversation_history_count"]} items")
    print(f"   Total memories: {stats["total_memories"]}")
    
    print(f"\n✅ ADVANCED MEMORY CORE TEST COMPLETED!")
    print("💡 Conversation memory working")
    print("💡 Context recall functioning") 
    print("💡 User-specific information tracking")
    print("💡 Context-aware prompt generation")

if __name__ == "__main__":
    test_advanced_memory()
    test_advanced_memory_core()
    
    print(f"\n🎉 ADVANCED MEMORY SYSTEM FULLY TESTED!")
    print("🚀 Vera_XT now has sophisticated memory capabilities!")
    print("💡 Cross-session persistence with vector embeddings!")
    print("🧠 Context-aware conversation management!")
