#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment variables from {env_file}")
else:
    print(f"⚠️  No .env file found at {env_file}")

# Set up paths but stay in project root for proper imports
app_dir = Path(__file__).parent
project_root = Path(__file__).parent.parent

# Add project root to Python path for proper imports
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)

print("🚀 Direct Ultimate RAG Server Start")
print("=" * 40)
print(f"📂 Directory: {os.getcwd()}")
print(f"🌐 URL: http://localhost:8000/docs")
print("🛑 Press Ctrl+C to stop")
print("=" * 40)

# Use uvicorn directly via command line to avoid import issues
# Change to app directory for uvicorn to find main.py
os.chdir(app_dir)
try:
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "127.0.0.1", 
        "--port", "8000",
        "--log-level", "warning"
    ], env=dict(os.environ, PYTHONPATH=str(project_root)))
except KeyboardInterrupt:
    print("\n👋 Server stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")