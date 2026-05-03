#!/usr/bin/env python3
"""
Simple test script to verify the chat handler works correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from src.bot.handlers.chat_handler import build_text_chat_handler, cmd_start
        from src.bot.application import create_bot
        from src.services.ai_services import chat
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ai_service():
    """Test that the AI service works."""
    try:
        from src.services.ai_services import chat
        
        # Test with a simple question
        response = chat(
            query="Halo, apa itu KKP?",
            session_id="test_session"
        )
        
        print(f"✅ AI Service test successful!")
        print(f"Response keys: {list(response.keys())}")
        print(f"Answer preview: {response.get('answer', '')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ AI Service error: {e}")
        return False

def main():
    print("🧪 Testing Chat Handler Integration")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        return False
    
    # Test AI service
    if not test_ai_service():
        return False
    
    print("\n🎉 All tests passed! Chat handler is ready to use.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)