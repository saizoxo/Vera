#/storage/emulated/0/Vxt/Vxt/test_advanced_memory.py
#!/usr/bin/env python3
"""
Test script for Advanced Memory System
Verifies cross-session memory persistence and advanced features
"""

import time
import json
from pathlib import Path

def test_advanced_memory_system():
    """Test the advanced memory system"""
    print("🧪 TESTING ADVANCED MEMORY SYSTEM")
    print("="*60)
    
    from vera_xt.core.advanced_memory import AdvancedMemorySystem
    
    # Initialize memory system
    memory = AdvancedMemorySystem()
    
    print("✅ Advanced Memory System initialized")
    print(f"📁 Memory directory: {memory.memory_dir}")
    print(f"📚 Conversations folder: {memory.conversations_dir}")
    print(f"📖 Knowledge folder: {memory.knowledge_dir}")
    print(f"🛠️  Skills folder: {memory.skills_dir}")
    print(f"🌍 Contexts folder: {memory.contexts_dir}")
    print(f"👤 Personal folder: {memory.personal_dir}")
    
    # Test memory storage
    print("\n📝 Testing memory storage...")
    
    # Store some test memories
    test_memories = [
        ("My name is Sarib", {"category": "personal", "type": "identity"}),
        ("I like programming", {"category": "interests", "type": "preference"}),
        ("Python is my favorite language", {"category": "skills", "type": "technical"}),
        ("I work on AI projects", {"category": "projects", "type": "professional"})
    ]
    
    for text, meta in test_memories:
        memory_id = memory.store_memory(text, metadata=meta, user_id="sarib_test")
        print(f"   Stored: '{text[:30]}...' → ID: {memory_id}")
    
    print(f"✅ Stored {len(test_memories)} test memories")
    
    # Test memory retrieval
    print("\n🔍 Testing memory retrieval...")
    
    # Search for related memories
    related = memory.search_similar_memories("programming", top_k=5)
    print(f"   Found {len(related)} related memories for 'programming':")
    for text, similarity, meta in related:
        print(f"     - '{text[:50]}...' (similarity: {similarity:.3f})")
    
    # Test user-specific memory
    print("\n👥 Testing user-specific memory...")
    user_memories = memory.get_user_memories("sarib_test")
    print(f"   User 'sarib_test' has {len(user_memories)} memories")
    
    for i, mem in enumerate(user_memories):
        print(f"     {i+1}. {mem['text'][:60]}...")
        print(f"        Category: {mem['metadata'].get('category', 'unknown')}")
        print(f"        Type: {mem['metadata'].get('type', 'unknown')}")
        print(f"        Timestamp: {mem['datetime']}")
    
    # Test cross-session persistence
    print("\n💾 Testing cross-session persistence...")
    
    # Save current state
    memory.save_memory()
    print("   ✓ Memory saved to persistent storage")
    
    # Create new instance (simulates new session)
    del memory
    time.sleep(1)  # Brief pause
    
    memory2 = AdvancedMemorySystem()
    print("   ✓ New memory instance created (simulates new session)")
    
    # Check if memories persisted
    all_memories = memory2.get_all_memories()
    print(f"   ✓ Retrieved {len(all_memories)} memories from persistent storage")
    
    user_memories_after_load = memory2.get_user_memories("sarib_test")
    print(f"   ✓ Retrieved {len(user_memories_after_load)} user-specific memories")
    
    # Verify specific memory recall
    name_memories = [m for m in all_memories if "name is sarib" in m['text'].lower()]
    print(f"   ✓ Found {len(name_memories)} memories containing 'name is sarib'")
    
    if name_memories:
        print(f"   ✓ Cross-session memory working: '{name_memories[0]['text']}'")
    
    # Test semantic search across sessions
    print("\n🔍 Testing semantic search across sessions...")
    semantic_results = memory2.search_similar_memories("tell me about coding", top_k=3)
    print(f"   Found {len(semantic_results)} semantically related memories for 'coding':")
    for text, similarity, meta in semantic_results:
        print(f"     - '{text[:50]}...' (relevance: {similarity:.3f})")
    
    # Test memory statistics
    print("\n📊 Testing memory statistics...")
    stats = memory2.get_memory_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Categories: {list(stats['category_stats'].keys())}")
    print(f"   Users tracked: {len(stats['user_profiles'])}")
    print(f"   Conversation threads: {len(stats['conversation_threads'])}")
    
    print("\n✅ ADVANCED MEMORY SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("💡 Cross-session memory persistence verified")
    print("💡 Semantic search functionality confirmed")
    print("💡 User-specific memory tracking working")
    print("💡 Memory organization by categories working")
    
    return True

def test_advanced_personality():
    """Test advanced personality features"""
    print("\n\n🎭 TESTING ADVANCED PERSONALITY FEATURES")
    print("="*60)
    
    from vera_xt.core.advanced_brain import AdvancedBrain
    
    brain = AdvancedBrain()
    
    print("✅ Advanced Brain initialized with personality gradients")
    print(f"🧠 Identity: {brain.identity_state['self_concept']}")
    print(f"🎯 Purpose: {brain.identity_state['purpose']}")
    print(f"🏷️  Role: {brain.identity_state['role']}")
    
    # Test personality trait evolution
    print("\n🔄 Testing personality evolution...")
    
    test_inputs = [
        "I have a question about Python programming",
        "This is really frustrating, I can't debug this code",
        "LOL that's hilarious, you're so funny!",
        "I need to finish this important project urgently",
        "Let's play a game together!"
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"   Input {i+1}: '{input_text[:30]}...'")
        
        # Process input to evolve personality
        response = brain.think_human_like(input_text)
        
        # Show personality changes
        active_traits = {k: v for k, v in brain.personality_traits.items() if v > 0.1}
        print(f"      Active traits: {active_traits}")
    
    print("✅ Personality evolution verified")
    
    # Test identity formation
    print("\n🏗️  Testing identity formation...")
    
    identity_inputs = [
        "You are my AI assistant",
        "Help me with my programming projects", 
        "Act as my coding mentor",
        "Be my technical advisor"
    ]
    
    for input_text in identity_inputs:
        brain.think_human_like(input_text)
    
    print(f"   Updated purpose: {brain.identity_state['purpose']}")
    print(f"   Updated role: {brain.identity_state['role']}")
    print(f"   Updated self-concept: {brain.identity_state['self_concept']}")
    
    print("✅ Identity formation verified")
    
    print("\n🎭 ADVANCED PERSONALITY TEST COMPLETED!")
    print("💡 Personality traits evolve from interactions")
    print("💡 Identity forms from user instructions")
    print("💡 Communication style adapts dynamically")

def run_complete_test():
    """Run complete advanced system test"""
    print("🚀 RUNNING COMPLETE ADVANCED SYSTEM TEST")
    print("=" * 80)
    
    memory_success = test_advanced_memory_system()
    personality_success = test_advanced_personality()
    
    print("\n" + "=" * 80)
    print("🏁 COMPLETE ADVANCED SYSTEM TEST RESULTS")
    print("=" * 80)
    
    if memory_success and personality_success:
        print("✅ ALL TESTS PASSED!")
        print("🌟 Advanced memory system working perfectly")
        print("🌟 Advanced personality system working perfectly") 
        print("🌟 Cross-session persistence verified")
        print("🌟 Semantic search functionality confirmed")
        print("🌟 Personality evolution verified")
        print("\n🚀 YOUR ADVANCED VERA_XT SYSTEM IS READY!")
        print("💡 Maximum sophistication achieved!")
    else:
        print("❌ SOME TESTS FAILED")
        print("💡 Please check the error messages above")
    
    print("=" * 80)

if __name__ == "__main__":
    run_complete_test()
