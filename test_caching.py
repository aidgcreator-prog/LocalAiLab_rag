#!/usr/bin/env python
"""Quick test to find correct caching import."""

import sys

# Test 1: langchain.cache
try:
    from langchain.cache import InMemoryCache
    print('[OK] InMemoryCache found in langchain.cache')
    sys.exit(0)
except ImportError as e:
    print(f'[ERR] langchain.cache failed: {e}')

# Test 2: langchain_core.cache  
try:
    from langchain_core.cache import InMemoryCache
    print('[OK] InMemoryCache found in langchain_core.cache')
    sys.exit(0)
except ImportError as e:
    print(f'[ERR] langchain_core.cache failed: {e}')

# Test 3: Check what's available
print('\n Searching for cache-related modules...')
try:
    import langchain
    print('langchain contents:', [m for m in dir(langchain) if 'cache' in m.lower()])
except Exception as e:
    print(f'Error: {e}')

try:
    import langchain_community
    print('langchain_community contents:', [m for m in dir(langchain_community) if 'cache' in m.lower()])
except Exception as e:
    print(f'Error: {e}')
