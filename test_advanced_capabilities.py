#/storage/emulated/0/Vxt/Vxt/test_advanced_capabilities.py
#!/usr/bin/env python3
"""
Test script for advanced server capabilities and 5D memory integration
"""

import time
import json
from pathlib import Path

def test_server_capabilities():
    """Test the advanced server integration capabilities"""
    print("🧪 Testing Advanced Server Capabilities...")
    
    from vera_xt.core.config_manager import config_manager
    from vera_xt.core.advanced_server_integration import AdvancedServerIntegration
    
    # Initialize server integration
    server = AdvancedServerIntegration(config_manager)
    
    # Check if server is available
    print(f"🌐 Server URL: {server.server_url}")
    
    # Test server capabilities
    capabilities = server.get_server_capabilities()
    print(f"📋 Server capabilities: {capabilities}")
    
    print("✅ Server capabilities test completed!")

def test_5d_memory_enhancement():
    """Test enhanced memory with server-generated content"""
    print("\n🧪 Testing 5D Memory Enhancement...")
    
    from vera_xt.core.memory_handler import MemoryHandler
    
    memory = MemoryHandler()
    
    # Test enhanced memory features
    print("Adding enhanced memory entries...")
    
    # Add a memory with rich context
    memory.add_to_memory(
        "enhanced_content", 
        "Advanced machine learning concepts including neural networks, backpropagation, and gradient descent",
        category="knowledge",
        context={
            "importance": 8,
            "semantic_tags": ["technical:ml", "technical:neural_networks", "keyword:backpropagation", "keyword:gradient"],
            "emotional_tone": 0.3,  # Slightly positive (educational)
            "project_context": "learning",
            "causal_links": ["neural_networks", "deep_learning"]
        }
    )
    
    # Add another memory with different context
    memory.add_to_memory(
        "personal_plan", 
        "Planning to learn advanced Python programming concepts this month",
        category="personal",
        context={
            "importance": 7,
            "semantic_tags": ["personal:learning", "technical:python", "planning:monthly"],
            "emotional_tone": 0.8,  # Positive (motivated)
            "project_context": "skill_development",
            "causal_links": ["skill_building", "career_advancement"]
        }
    )
    
    print("📊 Testing memory retrieval...")
    relevant_memories = memory.retrieve_relevant_memories("python")
    print(f"Found {len(relevant_memories)} relevant memories for 'python'")
    
    for i, mem in enumerate(relevant_memories):
        print(f"   {i+1}. Content: {mem.get('content', '')[:100]}...")
        print(f"      Tags: {mem.get('semantic_tags', [])}")
        print(f"      Importance: {mem.get('importance', 0)}")
        print(f"      Emotional tone: {mem.get('emotional_tone', 0.0)}")
    
    stats = memory.get_memory_stats()
    print(f"📈 Memory stats: {stats}")
    
    print("✅ 5D memory enhancement test completed!")

def test_enhanced_training_system():
    """Test the enhanced training system"""
    print("\n🧪 Testing Enhanced Training System...")
    
    from vera_xt.core.config_manager import config_manager  # Fixed: Added missing import
    from vera_xt.core.advanced_server_integration import EnhancedTrainingSystem, AdvancedServerIntegration
    from vera_xt.core.memory_handler import MemoryHandler
    
    # Initialize components
    memory = MemoryHandler()
    server = AdvancedServerIntegration(config_manager)
    server.set_memory_handler(memory)
    
    # Initialize enhanced trainer
    trainer = EnhancedTrainingSystem(server, memory)
    
    # Test training data generation (without actual server connection)
    print("Testing training data generation...")
    
    # Create sample training data manually to simulate server response
    sample_training_data = [
        {
            "id": f"server_train_{int(time.time())}_0",
            "topic": "python_programming",
            "prompt": "Explain Python decorators in detail with examples.",
            "response": "Python decorators are a powerful feature that allows you to modify the behavior of functions or classes without permanently modifying their code. They are essentially functions that take another function as an argument and extend its behavior.",
            "timestamp": time.time(),
            "quality_score": 0.85,
            "semantic_tags": ["technical:python", "technical:decorators", "educational:programming"]
        }
    ]
    
    # Add to memory like the trainer would
    for entry in sample_training_data:
        memory.add_to_memory(
            "server_training_data",
            f"Prompt: {entry['prompt']}\nResponse: {entry['response']}",
            category="knowledge",
            context={
                "importance": 8,
                "training_data": True,
                "topic": entry["topic"],
                "quality_score": entry["quality_score"],
                "semantic_tags": entry["semantic_tags"]
            }
        )
    
    print(f"📊 Added {len(sample_training_data)} training entries to memory")
    
    # Test insights
    insights = trainer.get_training_insights()
    print(f"📈 Training insights: {insights}")
    
    print("✅ Enhanced training system test completed!")

def test_memory_integration():
    """Test integration between server capabilities and 5D memory"""
    print("\n🧪 Testing Server-Memory Integration...")
    
    from vera_xt.core.basic_brain import BasicBrain
    
    brain = BasicBrain()
    
    # Test basic functionality
    print("Testing basic brain functionality...")
    response = brain.think_human_like("Hello, how are you?")
    print(f"🧠 Basic response: {response[:100]}...")
    
    # Test with enhanced memory
    brain.memory_handler.add_to_memory(
        "test_context", 
        "User is interested in learning Python programming",
        category="personal",
        context={
            "importance": 9,
            "semantic_tags": ["personal:interest", "technical:python", "learning:programming"],
            "emotional_tone": 0.7,
            "project_context": "education",
            "causal_links": ["skill_development"]
        }
    )
    
    # Test memory retrieval
    relevant = brain.memory_handler.retrieve_relevant_memories("python")
    print(f"🔍 Found {len(relevant)} relevant memories for 'python'")
    
    if relevant:
        mem = relevant[0]
        print(f"📋 Memory content: {mem.get('content', '')[:100]}...")
        print(f"   Tags: {mem.get('semantic_tags', [])}")
        print(f"   Importance: {mem.get('importance', 0)}")
        print(f"   Emotional tone: {mem.get('emotional_tone', 0.0)}")
    
    print("✅ Server-memory integration test completed!")

def test_performance_tracking():
    """Test performance tracking capabilities"""
    print("\n🧪 Testing Performance Tracking...")
    
    from vera_xt.core.config_manager import config_manager  # Fixed: Added missing import
    from vera_xt.core.advanced_server_integration import AdvancedServerIntegration
    
    server = AdvancedServerIntegration(config_manager)
    
    # Simulate some performance logging
    server._log_performance("test_method", 100, 200, time.time())
    server._log_performance("another_method", 150, 300, time.time() + 1)
    
    performance = server.benchmark_performance()
    print(f"📊 Performance metrics: {performance}")
    
    print("✅ Performance tracking test completed!")

def test_complete_workflow():
    """Test complete workflow: server + memory + training"""
    print("\n🧪 Testing Complete Workflow...")
    
    from vera_xt.core.basic_brain import BasicBrain
    from vera_xt.core.config_manager import config_manager  # Fixed: Added missing import
    from vera_xt.core.advanced_server_integration import AdvancedServerIntegration, EnhancedTrainingSystem
    
    # Initialize brain
    brain = BasicBrain()
    
    # Initialize server integration
    server = AdvancedServerIntegration(config_manager)
    server.set_memory_handler(brain.memory_handler)
    
    # Initialize enhanced trainer
    trainer = EnhancedTrainingSystem(server, brain.memory_handler)
    
    # Simulate a complete interaction cycle
    user_input = "I want to learn about machine learning"
    
    print(f"👤 User input: {user_input}")
    
    # Add to memory (simulating conversation)
    brain.memory_handler.add_to_memory(
        "user_request", 
        user_input,
        category="conversation",
        context={
            "importance": 7,
            "semantic_tags": ["request:learning", "technical:machine_learning"],
            "emotional_tone": 0.6,
            "project_context": "education"
        }
    )
    
    # Simulate server response (without actual server)
    server_response = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to 'learn' from data."
    
    # Add server response to memory
    brain.memory_handler.add_to_memory(
        "system_response", 
        server_response,
        category="conversation",
        context={
            "importance": 6,
            "semantic_tags": ["response:ml", "educational:ai"],
            "emotional_tone": 0.4,
            "project_context": "education"
        }
    )
    
    # Test memory retrieval for context
    relevant_memories = brain.memory_handler.retrieve_relevant_memories("machine learning")
    print(f"🔗 Found {len(relevant_memories)} relevant memories for follow-up")
    
    # Test enhanced training (simulated)
    if hasattr(trainer, 'get_training_insights'):
        insights = trainer.get_training_insights()
        print(f"🎓 Training insights: {insights}")
    
    print("✅ Complete workflow test completed!")

if __name__ == "__main__":
    print("🚀 Vera_XT Advanced Capabilities Test Suite")
    print("=" * 50)
    
    test_server_capabilities()
    test_5d_memory_enhancement()
    test_enhanced_training_system()
    test_memory_integration()
    test_performance_tracking()
    test_complete_workflow()
    
    print("\n✅ All advanced capability tests completed!")
    print("💡 System is ready for training integration and benchmarking!")