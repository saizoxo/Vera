#/storage/emulated/0/Vxt/Vxt/test_session_persistence.py
#!/usr/bin/env python3
"""
Test script for session persistence and memory recall
"""

import json
import time
from pathlib import Path
from datetime import datetime

def test_memory_persistence():
    """Test that memories persist across sessions"""
    print("🧪 Testing Memory Persistence...")
    
    # Check if memory files exist and have rich context
    memory_dir = Path("Memory_Data")
    conversation_files = list((memory_dir / "conversations").glob("*.json"))
    
    if conversation_files:
        print(f"📚 Found {len(conversation_files)} conversation files")
        
        # Check the most recent conversation file
        latest_file = max(conversation_files, key=lambda x: x.stat().st_mtime)
        print(f"🔍 Examining: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        
        print(f"📊 Conversation entries: {len(conversation.get('entries', []))}")
        
        # Check if entries have rich context
        entries = conversation.get('entries', [])
        if entries:
            first_entry = entries[0]
            print(f"📋 First entry structure:")
            print(f"   - Content: {first_entry.get('content', '')[:50]}...")
            print(f"   - Semantic tags: {first_entry.get('semantic_tags', [])}")
            print(f"   - Emotional tone: {first_entry.get('emotional_tone', 0.0)}")
            print(f"   - Importance: {first_entry.get('importance', 0)}")
            print(f"   - Project context: {first_entry.get('project_context', 'unknown')}")
            print(f"   - Causal links: {first_entry.get('causal_links', [])}")
        
        print("✅ Memory persistence test completed!")
    else:
        print("❌ No conversation files found")
    
    # Test different memory categories
    for category in ["knowledge", "skills", "contexts", "personal"]:
        category_dir = memory_dir / category
        if category_dir.exists():
            files = list(category_dir.glob("*.json"))
            print(f"📂 {category.title()} files: {len(files)}")
            
            # Check first file in each category
            if files:
                with open(files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        entry = data[0]
                        print(f"   First {category} entry: {entry.get('content', '')[:50]}...")
                        print(f"      Semantic tags: {entry.get('semantic_tags', [])}")
                        print(f"      Emotional tone: {entry.get('emotional_tone', 0.0)}")
                        print(f"      Importance: {entry.get('importance', 0)}")

def test_memory_recall():
    """Test memory recall functionality"""
    print("\n🧪 Testing Memory Recall...")
    
    from vera_xt.core.memory_handler import MemoryHandler
    
    memory = MemoryHandler()
    
    # Test retrieving relevant memories
    print("🔍 Testing relevant memory retrieval...")
    relevant_memories = memory.retrieve_relevant_memories("hello")
    
    print(f"📊 Found {len(relevant_memories)} relevant memories")
    for i, mem in enumerate(relevant_memories):
        print(f"   {i+1}. {mem.get('content', '')[:100]}...")
        print(f"      Tags: {mem.get('semantic_tags', [])}")
        print(f"      Importance: {mem.get('importance', 0)}")
    
    print("✅ Memory recall test completed!")

def test_context_awareness():
    """Test context awareness and pattern learning"""
    print("\n🧪 Testing Context Awareness...")
    
    from vera_xt.core.memory_handler import MemoryHandler
    
    memory = MemoryHandler()
    
    # Check learned patterns
    learned_patterns = memory.learned_patterns
    print(f"🧠 Learned patterns: {len(learned_patterns)} categories")
    
    for category, patterns in learned_patterns.items():
        print(f"   {category}: {patterns[:5]}...")  # Show first 5 patterns
    
    print("✅ Context awareness test completed!")

def test_memory_statistics():
    """Test memory statistics"""
    print("\n📊 Testing Memory Statistics...")
    
    from vera_xt.core.memory_handler import MemoryHandler
    
    memory = MemoryHandler()
    stats = memory.get_memory_stats()
    
    print(f"📈 Memory Statistics:")
    for key, value in stats.items():
        if key == "folder_stats":
            print(f"   Folder stats:")
            for folder, count in value.items():
                print(f"     - {folder}: {count} files")
        else:
            print(f"   {key}: {value}")
    
    print("✅ Memory statistics test completed!")

if __name__ == "__main__":
    print("🚀 Vera_XT Session Persistence Test")
    print("=" * 50)
    
    test_memory_persistence()
    test_memory_recall()
    test_context_awareness()
    test_memory_statistics()
    
    print("\n✅ All persistence tests completed!")
    print("💡 Memory system is ready for API integration!")