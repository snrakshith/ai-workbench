#!/usr/bin/env python3
"""
Google Gemini API Test Script
A utility to test and verify Google Gemini API connectivity and functionality.

Usage:
    python test_gemini_api.py
    python test_gemini_api.py --verbose
    python test_gemini_api.py --model gemini-2.0-flash-exp
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    import logging
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )

def load_api_key() -> Optional[str]:
    """Load Google API key from environment or .env file."""
    # Try environment variable first
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        # Try loading from .env file
        env_file = project_root / '.env'
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('GOOGLE_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"\'')
                            break
            except Exception as e:
                print(f"⚠️  Warning: Could not read .env file: {e}")
    
    return api_key

def test_basic_connection(api_key: str, model_name: str = "gemini-2.0-flash", verbose: bool = False) -> Dict[str, Any]:
    """Test basic API connection and authentication."""
    print(f"🔧 Testing basic connection to {model_name}...")
    
    try:
        import google.genai as genai
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        if verbose:
            print(f"   ✅ Client initialized successfully")
        
        # Test simple generation
        start_time = time.time()
        
        response = client.models.generate_content(
            model=model_name,
            contents=[{"parts": [{"text": "Hello! Please respond with 'API test successful' to confirm connectivity."}]}],
            config={
                "max_output_tokens": 50,
                "temperature": 0.1
            }
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Parse response
        response_text = ""
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'content'):
                if hasattr(response.candidates[0].content, 'parts'):
                    response_text = response.candidates[0].content.parts[0].text
        elif isinstance(response, dict):
            if 'candidates' in response and response['candidates']:
                candidate = response['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    response_text = candidate['content']['parts'][0]['text']
        
        if response_text:
            print(f"   ✅ Basic generation test: SUCCESS")
            if verbose:
                print(f"   📝 Response: {response_text.strip()}")
                print(f"   ⏱️  Response time: {response_time:.1f}ms")
            
            return {
                "success": True,
                "response": response_text.strip(),
                "response_time_ms": response_time,
                "model": model_name
            }
        else:
            print(f"   ❌ Basic generation test: FAILED - Empty response")
            return {
                "success": False,
                "error": "Empty response from API",
                "model": model_name
            }
            
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print(f"   💡 Install google-genai: uv pip install google-genai")
        return {"success": False, "error": f"Import error: {e}"}
    except Exception as e:
        print(f"   ❌ Basic generation test: FAILED")
        print(f"   🐛 Error: {str(e)}")
        
        # Check for common errors
        error_str = str(e).lower()
        if "api_key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            print(f"   💡 Likely cause: Invalid or missing API key")
        elif "quota" in error_str or "limit" in error_str:
            print(f"   💡 Likely cause: API quota exceeded or rate limit")
        elif "model" in error_str or "not found" in error_str:
            print(f"   💡 Likely cause: Model '{model_name}' not available")
        elif "network" in error_str or "connection" in error_str:
            print(f"   💡 Likely cause: Network connectivity issue")
        
        return {"success": False, "error": str(e), "model": model_name}

def test_embedding_generation(api_key: str, verbose: bool = False) -> Dict[str, Any]:
    """Test embedding generation capability."""
    print(f"🔧 Testing embedding generation...")
    
    try:
        import google.genai as genai
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        start_time = time.time()
        
        # Test embedding generation
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=["This is a test document for embedding generation."]
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Check response
        if hasattr(response, 'embeddings') and response.embeddings:
            embedding = response.embeddings[0]
            if hasattr(embedding, 'values') and embedding.values:
                embedding_dim = len(embedding.values)
                print(f"   ✅ Embedding generation: SUCCESS")
                if verbose:
                    print(f"   📐 Embedding dimension: {embedding_dim}")
                    print(f"   ⏱️  Response time: {response_time:.1f}ms")
                
                return {
                    "success": True,
                    "embedding_dimension": embedding_dim,
                    "response_time_ms": response_time
                }
        
        print(f"   ❌ Embedding generation: FAILED - Invalid response structure")
        return {"success": False, "error": "Invalid response structure"}
        
    except Exception as e:
        print(f"   ❌ Embedding generation: FAILED")
        print(f"   🐛 Error: {str(e)}")
        return {"success": False, "error": str(e)}

def test_advanced_generation(api_key: str, model_name: str = "gemini-2.0-flash", verbose: bool = False) -> Dict[str, Any]:
    """Test advanced generation with longer content and specific parameters."""
    print(f"🔧 Testing advanced generation capabilities...")
    
    try:
        import google.genai as genai
        
        client = genai.Client(api_key=api_key)
        
        # Complex prompt for testing
        complex_prompt = """Please analyze the following scenario and provide a structured response:

Scenario: A company wants to implement a RAG (Retrieval-Augmented Generation) system for their customer support.

Please provide:
1. Three key benefits of using RAG
2. Two potential challenges they might face
3. One recommended approach for implementation

Format your response with clear sections and bullet points."""

        start_time = time.time()
        
        response = client.models.generate_content(
            model=model_name,
            contents=[{"parts": [{"text": complex_prompt}]}],
            config={
                "max_output_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Parse response
        response_text = ""
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'content'):
                if hasattr(response.candidates[0].content, 'parts'):
                    response_text = response.candidates[0].content.parts[0].text
        elif isinstance(response, dict):
            if 'candidates' in response and response['candidates']:
                candidate = response['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    response_text = candidate['content']['parts'][0]['text']
        
        if response_text and len(response_text) > 100:
            print(f"   ✅ Advanced generation: SUCCESS")
            if verbose:
                print(f"   📝 Response length: {len(response_text)} characters")
                print(f"   ⏱️  Response time: {response_time:.1f}ms")
                print(f"   📄 Response preview: {response_text[:200]}...")
            
            return {
                "success": True,
                "response_length": len(response_text),
                "response_time_ms": response_time,
                "response_preview": response_text[:200]
            }
        else:
            print(f"   ❌ Advanced generation: FAILED - Response too short or empty")
            return {"success": False, "error": "Response too short or empty"}
            
    except Exception as e:
        print(f"   ❌ Advanced generation: FAILED")
        print(f"   🐛 Error: {str(e)}")
        return {"success": False, "error": str(e)}

def test_rate_limits(api_key: str, model_name: str = "gemini-2.0-flash", verbose: bool = False) -> Dict[str, Any]:
    """Test API rate limits by making multiple rapid requests."""
    print(f"🔧 Testing rate limits (5 rapid requests)...")
    
    try:
        import google.genai as genai
        
        client = genai.Client(api_key=api_key)
        
        successful_requests = 0
        failed_requests = 0
        total_time = 0
        errors = []
        
        for i in range(5):
            try:
                start_time = time.time()
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[{"parts": [{"text": f"Test request {i+1}: Please respond with a short acknowledgment."}]}],
                    config={
                        "max_output_tokens": 20,
                        "temperature": 0.1
                    }
                )
                
                request_time = (time.time() - start_time) * 1000
                total_time += request_time
                
                # Check if response is valid
                response_text = ""
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    if hasattr(response.candidates[0], 'content'):
                        if hasattr(response.candidates[0].content, 'parts'):
                            response_text = response.candidates[0].content.parts[0].text
                elif isinstance(response, dict):
                    if 'candidates' in response and response['candidates']:
                        candidate = response['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            response_text = candidate['content']['parts'][0]['text']
                
                if response_text:
                    successful_requests += 1
                    if verbose:
                        print(f"   ✅ Request {i+1}: {request_time:.0f}ms")
                else:
                    failed_requests += 1
                    errors.append(f"Request {i+1}: Empty response")
                
            except Exception as e:
                failed_requests += 1
                errors.append(f"Request {i+1}: {str(e)}")
                if verbose:
                    print(f"   ❌ Request {i+1}: {str(e)}")
        
        avg_time = total_time / successful_requests if successful_requests > 0 else 0
        
        if successful_requests >= 4:  # Allow 1 failure
            print(f"   ✅ Rate limit test: SUCCESS ({successful_requests}/5 requests)")
            if verbose:
                print(f"   ⏱️  Average response time: {avg_time:.1f}ms")
        else:
            print(f"   ⚠️  Rate limit test: PARTIAL ({successful_requests}/5 requests)")
        
        return {
            "success": successful_requests >= 4,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "average_response_time_ms": avg_time,
            "errors": errors
        }
        
    except Exception as e:
        print(f"   ❌ Rate limit test: FAILED")
        print(f"   🐛 Error: {str(e)}")
        return {"success": False, "error": str(e)}

def print_summary(results: Dict[str, Any]):
    """Print a comprehensive summary of all tests."""
    print("\n" + "="*80)
    print("📋 GOOGLE GEMINI API TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for test in results.values() if test.get('success', False))
    
    print(f"📊 Overall Status: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - Google Gemini API is fully functional!")
    elif passed_tests >= total_tests * 0.75:
        print("⚠️  MOSTLY WORKING - Some minor issues detected")
    elif passed_tests >= total_tests * 0.5:
        print("🔧 PARTIAL FUNCTIONALITY - Significant issues need attention")
    else:
        print("❌ MAJOR ISSUES - API may not be working properly")
    
    print("\n📝 Test Details:")
    
    test_names = {
        'basic': '🔌 Basic Connection',
        'embedding': '📐 Embedding Generation',
        'advanced': '🧠 Advanced Generation',
        'rate_limit': '⚡ Rate Limits'
    }
    
    for test_key, result in results.items():
        test_name = test_names.get(test_key, test_key.title())
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"   {test_name}: {status}")
        
        if not result.get('success', False) and 'error' in result:
            print(f"      Error: {result['error']}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if results.get('basic', {}).get('success', False):
        print(f"   ✅ Basic API connectivity is working")
    else:
        print(f"   🔧 Check your API key and network connectivity")
    
    if results.get('embedding', {}).get('success', False):
        print(f"   ✅ Embedding generation is working - RAG system ready")
    else:
        print(f"   ⚠️  Embedding generation issues may affect RAG functionality")
    
    if results.get('advanced', {}).get('success', False):
        print(f"   ✅ Advanced generation is working - Full LLM capabilities available")
    else:
        print(f"   ⚠️  Advanced generation issues may limit response quality")
    
    rate_result = results.get('rate_limit', {})
    if rate_result.get('success', False):
        print(f"   ✅ Rate limits are reasonable for normal usage")
    else:
        successful = rate_result.get('successful_requests', 0)
        if successful > 0:
            print(f"   ⚠️  Some rate limiting detected ({successful}/5 requests succeeded)")
        else:
            print(f"   ❌ Severe rate limiting or API issues")

def main():
    """Main function for the API test script."""
    parser = argparse.ArgumentParser(
        description="Test Google Gemini API connectivity and functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_gemini_api.py
  python test_gemini_api.py --verbose
  python test_gemini_api.py --model gemini-2.0-flash-exp
  python test_gemini_api.py --skip-embedding --skip-rate-limit
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output and response content"
    )
    
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Model to test (default: gemini-2.0-flash)"
    )
    
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding generation tests"
    )
    
    parser.add_argument(
        "--skip-advanced",
        action="store_true",
        help="Skip advanced generation tests"
    )
    
    parser.add_argument(
        "--skip-rate-limit",
        action="store_true",
        help="Skip rate limit tests"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    print("🧪 Google Gemini API Test Suite")
    print("="*50)
    
    # Load API key
    print("🔑 Loading API key...")
    api_key = load_api_key()
    
    if not api_key:
        print("❌ No Google API key found!")
        print("💡 Please set GOOGLE_API_KEY environment variable or add it to .env file")
        print("   export GOOGLE_API_KEY='your-api-key-here'")
        print("   # or add to .env file: GOOGLE_API_KEY=your-api-key-here")
        sys.exit(1)
    
    print(f"✅ API key loaded (length: {len(api_key)} characters)")
    print(f"🤖 Testing model: {args.model}")
    print()
    
    # Run tests
    results = {}
    
    # Basic connection test
    results['basic'] = test_basic_connection(api_key, args.model, args.verbose)
    
    # Embedding test
    if not args.skip_embedding:
        results['embedding'] = test_embedding_generation(api_key, args.verbose)
    
    # Advanced generation test
    if not args.skip_advanced:
        results['advanced'] = test_advanced_generation(api_key, args.model, args.verbose)
    
    # Rate limit test
    if not args.skip_rate_limit:
        results['rate_limit'] = test_rate_limits(api_key, args.model, args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    total_tests = len(results)
    passed_tests = sum(1 for test in results.values() if test.get('success', False))
    
    if passed_tests == total_tests:
        print(f"\n🎉 All tests passed! Google Gemini API is ready for use.")
        sys.exit(0)
    elif passed_tests >= total_tests * 0.75:
        print(f"\n⚠️  Most tests passed. API should work but monitor for issues.")
        sys.exit(0)
    else:
        print(f"\n❌ Multiple test failures. Please check your API key and connectivity.")
        sys.exit(1)

if __name__ == "__main__":
    main()