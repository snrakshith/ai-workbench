#!/usr/bin/env python3
"""Quick test of HyDE enhancer"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("🧪 Testing HyDE enhancer...")

try:
    from src.enhancement.hyde import create_hyde_enhancer
    print("✅ HyDE enhancer imported successfully")
    
    # Create the enhancer
    hyde_enhancer = create_hyde_enhancer()
    print("✅ HyDE enhancer created successfully")
    
    # Test with a simple query
    test_query = "What are LangChain agents?"
    print(f"\n🔍 Testing with query: {test_query}")
    
    result = hyde_enhancer.enhance_query(test_query, domain_context="LangChain")
    
    if result.get("success"):
        print("✅ HyDE enhancement successful!")
        print(f"📄 Generated {len(result.get('hypothetical_documents', []))} hypothetical documents")
        print(f"⏱️  Total time: {result.get('performance', {}).get('total_time_ms', 0):.1f}ms")
    else:
        print(f"❌ HyDE enhancement failed: {result.get('error', 'Unknown error')}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()