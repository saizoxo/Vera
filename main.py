#/storage/emulated/0/Vxt/Vxt/main.py
#!/usr/bin/env python3
"""
Vera_XT - Your Always-Available AI Partner
Phase 1: Basic Foundation
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 Starting Vera_XT - AI Partner")
    print("Loading basic foundation system...")
    
    try:
        # Import all components
        from vera_xt.interfaces.cli_interface import CLIInterface
        from vera_xt.core.basic_brain import BasicBrain
        from vera_xt.memory.simple_memory import SimpleMemory
        from vera_xt.memory.memory_manager import MemoryManager
        from vera_xt.security.basic_protection import BasicProtection
        
        print("✅ All modules imported successfully!")
        
        # Initialize the brain (core intelligence)
        print("🧠 Initializing Basic Brain...")
        brain = BasicBrain()
        
        # Initialize simple memory system
        print("🧠 Initializing Simple Memory...")
        simple_memory = SimpleMemory()
        
        # Initialize memory manager
        print("🧠 Initializing Memory Manager...")
        memory_manager = MemoryManager()
        memory_manager.set_simple_memory(simple_memory)
        
        # Initialize security system
        print("🛡️  Initializing Security System...")
        security_system = BasicProtection()
        
        # Connect all systems together
        print("🔗 Connecting systems...")
        brain.simple_memory = simple_memory
        brain.memory_manager = memory_manager
        brain.security_system = security_system
        
        memory_manager.simple_memory = simple_memory
        
        print("✅ All systems connected!")
        
        # Create the interface
        print("🖥️  Creating CLI Interface...")
        interface = CLIInterface(brain)
        
        print("\n" + "="*60)
        print("🎉 Vera_XT Basic Foundation Loaded Successfully!")
        print("💡 Your AI Partner is ready to help")
        print("✨ Features: Human-like thinking, Smart memory, Security")
        print("="*60)
        print("Type 'help' for commands or start chatting!")
        print("-" * 50)
        
        # Start the interface
        interface.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all required files are created in the correct locations")
        print("Expected structure:")
        print("  Vxt/Vxt/")
        print("  ├── vera_xt/")
        print("  │   ├── core/basic_brain.py")
        print("  │   ├── memory/simple_memory.py") 
        print("  │   ├── memory/memory_manager.py")
        print("  │   ├── security/basic_protection.py")
        print("  │   └── interfaces/cli_interface.py")
        print("  ├── Workspace/")
        print("  ├── Models/")
        print("  ├── Memory_Data/")
        print("  └── main.py")
        
    except Exception as e:
        print(f"❌ Error starting Vera_XT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
