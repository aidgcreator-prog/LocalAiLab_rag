#!/usr/bin/env python
"""Test basic agent functionality without subagents."""

import asyncio
import os
from pathlib import Path
from langchain.chat_models import init_chat_model

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

async def test_model():
    """Test if the model is accessible."""
    model_name = os.getenv("DEEPAGENT_MODEL", "ollama:gemma4:26b")
    print(f"Testing model: {model_name}")
    
    try:
        model = init_chat_model(model=model_name, temperature=0)
        print("[OK] Model initialized")
        
        # Try a simple invoke
        result = model.invoke([{"role": "user", "content": "Say 'test passed'"}])
        print(f"[OK] Model response: {result.content[:100]}")
        
    except Exception as e:
        print(f"[ERR] Model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_model())
