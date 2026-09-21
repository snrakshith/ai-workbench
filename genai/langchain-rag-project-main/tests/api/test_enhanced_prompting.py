#!/usr/bin/env python3
"""
Test script for the enhanced prompting system.
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict


def test_enhanced_prompting():
    """Test the enhanced prompting system with various query types."""
    
    base_url = "http://localhost:8000"
    
    # Test queries for different content types
    test_queries = {
        "API Documentation": [
            "What are the parameters for the ChatOpenAI class in LangChain?",
            "How do I configure the retrieval settings in LangChain?",
            "What methods are available in the Document class?"
        ],
        "Tutorial/Guide": [
            "How do I get started with LangChain agents?",
            "What's the step-by-step process to create a RAG application?",
            "How do I build a simple chatbot with LangChain?"
        ],
        "Code Examples": [
            "Show me example code for creating a vector store in LangChain",
            "How do I implement a custom document loader?",
            "What's an example of using LangGraph for agent workflows?"
        ],
        "Configuration": [
            "How do I set up environment variables for LangChain?",
            "What's the proper way to configure embedding models?",
            "How do I setup LangSmith tracing?"
        ],
        "Non-LangChain (Should fallback)": [
            "Write Python code to generate Fibonacci series up to n terms",
            "How do I create a tkinter GUI application?",
            "What's the quicksort algorithm implementation?"
        ]
    }
    
    print("🧪 Testing Enhanced Prompting System")
    print("=" * 80)
    
    results = []
    
    for category, queries in test_queries.items():
        print(f"\n📋 Testing Category: {category}")
        print("-" * 50)
        
        for query in queries:
            print(f"\n🔍 Query: {query[:60]}...")
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/api/v1/rag/query",
                    json={
                        "question": query,
                        "conversation_id": f"test-enhanced-{datetime.now().strftime('%H%M%S')}"
                    },
                    timeout=30
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract metadata for analysis
                    metadata = result.get("metadata", {})
                    context_analysis = result.get("context_analysis", {})
                    response_structure = result.get("response_structure", {})
                    
                    retrieval_strategy = metadata.get("retrieval_strategy", "unknown")
                    content_type = context_analysis.get("primary_content_type", "unknown")
                    adaptive_used = context_analysis.get("adaptive_prompting_used", False)
                    
                    # Display results
                    if retrieval_strategy == "direct_llm_fallback":
                        print("   ✅ FALLBACK - Used direct LLM (expected for non-LangChain queries)")
                    else:
                        print(f"   📚 RAG - Retrieved {result.get('retrieved_documents', 0)} docs")
                        print(f"   🎯 Content Type: {content_type}")
                        print(f"   🤖 Adaptive Prompting: {'Yes' if adaptive_used else 'No'}")
                    
                    print(f"   ⏱️ Response Time: {response_time:.0f}ms")
                    print(f"   📊 Confidence: {result.get('confidence_score', 0):.2f}")
                    
                    # Check response structure
                    if response_structure:
                        structure_quality = []
                        if response_structure.get("has_code_blocks"):
                            structure_quality.append("Code")
                        if response_structure.get("has_bullet_points") or response_structure.get("has_numbered_lists"):
                            structure_quality.append("Lists")
                        if response_structure.get("has_headers"):
                            structure_quality.append("Headers")
                        
                        if structure_quality:
                            print(f"   📝 Structure: {', '.join(structure_quality)}")
                        
                        if response_structure.get("follows_structured_format"):
                            print("   ✅ Follows structured format")
                    
                    # Store result for summary
                    results.append({
                        "category": category,
                        "query": query,
                        "success": True,
                        "retrieval_strategy": retrieval_strategy,
                        "content_type": content_type,
                        "adaptive_used": adaptive_used,
                        "response_time": response_time,
                        "confidence": result.get("confidence_score", 0),
                        "structured_format": response_structure.get("follows_structured_format", False),
                        "word_count": response_structure.get("word_count", 0)
                    })
                    
                    print(f"   📄 Answer Preview: {result['answer'][:100]}...")
                    
                else:
                    print(f"   ❌ API Error: {response.status_code}")
                    results.append({
                        "category": category,
                        "query": query,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "category": category,
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("📊 ENHANCED PROMPTING TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = [r for r in results if r.get("success")]
    failed_tests = [r for r in results if not r.get("success")]
    
    print(f"✅ Successful Tests: {len(successful_tests)}")
    print(f"❌ Failed Tests: {len(failed_tests)}")
    print(f"📈 Success Rate: {len(successful_tests)/(len(results))*100:.1f}%")
    
    if successful_tests:
        # Analyze adaptive prompting usage
        adaptive_used_count = sum(1 for r in successful_tests if r.get("adaptive_used"))
        fallback_count = sum(1 for r in successful_tests if r.get("retrieval_strategy") == "direct_llm_fallback")
        rag_count = len(successful_tests) - fallback_count
        
        print(f"🤖 Adaptive Prompting Used: {adaptive_used_count}/{len(successful_tests)} ({adaptive_used_count/len(successful_tests)*100:.1f}%)")
        print(f"📚 RAG Responses: {rag_count}")
        print(f"💡 Direct LLM Fallbacks: {fallback_count}")
        
        # Response quality metrics
        avg_confidence = sum(r.get("confidence", 0) for r in successful_tests) / len(successful_tests)
        avg_response_time = sum(r.get("response_time", 0) for r in successful_tests) / len(successful_tests)
        structured_responses = sum(1 for r in successful_tests if r.get("structured_format"))
        avg_word_count = sum(r.get("word_count", 0) for r in successful_tests) / len(successful_tests)
        
        print(f"📊 Average Confidence: {avg_confidence:.3f}")
        print(f"⏱️ Average Response Time: {avg_response_time:.0f}ms")
        print(f"📝 Structured Responses: {structured_responses}/{len(successful_tests)} ({structured_responses/len(successful_tests)*100:.1f}%)")
        print(f"📄 Average Word Count: {avg_word_count:.0f}")
        
        # Content type distribution
        content_types = {}
        for result in successful_tests:
            if result.get("retrieval_strategy") != "direct_llm_fallback":
                content_type = result.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1
        
        if content_types:
            print("\n🎯 Content Type Distribution:")
            for content_type, count in content_types.items():
                print(f"   {content_type}: {count}")
    
    if failed_tests:
        print("\n❌ Failed Tests:")
        for result in failed_tests:
            print(f"   {result['category']}: {result['query'][:50]}... - {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("✅ Enhanced prompting system testing completed!")
    
    # Export results to JSON for further analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_prompting_test_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests)/(len(results))*100 if results else 0,
                "adaptive_prompting_usage": adaptive_used_count/len(successful_tests)*100 if successful_tests else 0,
                "average_confidence": avg_confidence if successful_tests else 0,
                "average_response_time_ms": avg_response_time if successful_tests else 0,
                "structured_response_rate": structured_responses/len(successful_tests)*100 if successful_tests else 0
            },
            "detailed_results": results
        }, indent=2)
    
    print(f"📁 Detailed results exported to: {results_file}")


if __name__ == "__main__":
    test_enhanced_prompting()