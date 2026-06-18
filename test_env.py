#!/usr/bin/env python
"""Test that .env file loads correctly."""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    print("[OK] python-dotenv installed")
    print(f"LANGSMITH_API_KEY: {'*' * 10 + os.getenv('LANGSMITH_API_KEY', '')[-20:] if os.getenv('LANGSMITH_API_KEY') else 'NOT SET'}")
    print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
    print(f"LANGCHAIN_ENABLE_CACHE: {os.getenv('LANGCHAIN_ENABLE_CACHE')}")
except ImportError:
    print("[ERR] python-dotenv not installed")
