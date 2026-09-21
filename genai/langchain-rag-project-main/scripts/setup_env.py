"""Setup script for initializing the development environment."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if the environment is properly set up."""
    project_root = Path(__file__).parent.parent
    
    # Load environment variables
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Environment variables loaded from .env")
    else:
        print("❌ .env file not found")
        print("📝 Creating .env from template...")
        
        # Copy .env.example to .env
        example_path = project_root / ".env.example"
        if example_path.exists():
            with open(example_path, 'r') as src, open(env_path, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created from template")
            print("⚠️  Please edit .env file with your actual API keys")
        else:
            print("❌ .env.example not found")
            return False
    
    # Check required environment variables
    required_vars = [
        'GOOGLE_API_KEY',
        'QDRANT_URL',
        'QDRANT_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your_'):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing or placeholder values for: {', '.join(missing_vars)}")
        print("Please update your .env file with actual API keys")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_api_connections():
    """Test connections to external APIs."""
    print("\n🔌 Testing API connections...")
    
    # Test Google Gemini API
    try:
        import google.genai as genai
        
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            
            # Quick test
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content("Hello")
            print("✅ Google Gemini API connection successful")
        else:
            print("❌ Google API key not found")
    except Exception as e:
        print(f"❌ Google Gemini API connection failed: {str(e)}")
    
    # Test Qdrant connection
    try:
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if qdrant_url and qdrant_api_key:
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            collections = client.get_collections()
            print("✅ Qdrant connection successful")
        else:
            print("❌ Qdrant credentials not found")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {str(e)}")

def main():
    """Main setup function."""
    print("🚀 Setting up LangChain RAG System environment...\n")
    
    # Check Python version
    if sys.version_info < (3, 12):
        print("❌ Python 3.12 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check environment
    env_ok = check_environment()
    
    if env_ok:
        test_api_connections()
        
        print("\n🎉 Environment setup complete!")
        print("\nNext steps:")
        print("1. Run: jupyter notebook notebooks/01_setup.ipynb")
        print("2. Or start FastAPI: python app/main.py")
        return True
    else:
        print("\n⚠️  Please fix the environment issues above and run again")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)