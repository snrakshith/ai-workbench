# 📚 LangChain Notebooks - Complete RAG Course

A comprehensive, hands-on course for learning **Retrieval-Augmented Generation (RAG)** with **LangChain 1.0.5+** - November 2025.

Perfect for **mixed-level classes** with beginners and intermediate students!

## 🎯 What You'll Learn

This course teaches you how to build production-ready RAG applications from scratch:

1. ✅ **LangChain Fundamentals** - Architecture, LCEL, core concepts
2. ✅ **Document Loading** - PDF, CSV, JSON, HTML, and more
3. ✅ **Text Splitting** - Optimal chunking strategies
4. ✅ **Embeddings** - OpenAI, Google Gemini, comparisons
5. ✅ **Vector Stores** - FAISS, Chroma, InMemory
6. ✅ **Retrieval Strategies** - Similarity, MMR, custom retrievers
7. ✅ **Complete RAG Pipeline** - Production-ready implementation

---

## 📖 Course Structure

### 7 Progressive Notebooks

| Notebook | Topic | Level | Duration |
|----------|-------|-------|----------|
| [01_Introduction_and_Fundamentals.ipynb](./01_Introduction_and_Fundamentals.ipynb) | LangChain basics, LCEL, first LLM call | 🔰 Beginner | 45 min |
| [02_Document_Loaders.ipynb](./02_Document_Loaders.ipynb) | Load PDF, CSV, JSON, HTML, batch loading | 🔰 Beginner | 60 min |
| [03_Text_Splitting_Strategies.ipynb](./03_Text_Splitting_Strategies.ipynb) | Recursive, HTML, JSON, token splitters | 🔰→🎓 | 45 min |
| [04_Embeddings_and_Vector_Representations.ipynb](./04_Embeddings_and_Vector_Representations.ipynb) | OpenAI, Gemini embeddings, similarity | 🔰→🎓 | 45 min |
| [05_Vector_Stores.ipynb](./05_Vector_Stores.ipynb) | InMemory, FAISS, Chroma comparison | 🎓 Intermediate | 60 min |
| [06_Retrieval_Strategies.ipynb](./06_Retrieval_Strategies.ipynb) | Similarity, MMR, custom retrievers | 🎓 Intermediate | 45 min |
| [07_Complete_RAG_Pipeline.ipynb](./07_Complete_RAG_Pipeline.ipynb) | End-to-end RAG, best practices | 🎓 Intermediate | 90 min |

**Total Course Time:** ~6 hours of hands-on learning

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9, 3.10, or 3.11 (3.12+ may have compatibility issues)
- OpenAI API key (required)
- Google API key (optional, for Notebook 04)
- Basic Python knowledge

### 1. Install Dependencies

```bash
# Clone or download this repository
cd simple-rag-langchain

# Install required packages
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required
OPENAI_API_KEY=sk-proj-your-openai-key-here

# Optional (for Notebook 04 - Google Gemini)
GOOGLE_API_KEY=your-google-api-key-here
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Google Gemini: https://makersuite.google.com/app/apikey

### 3. Start Jupyter

```bash
jupyter notebook
```

### 4. Begin with Notebook 01

Open `01_Introduction_and_Fundamentals.ipynb` and work through the notebooks in order.

---

## 📁 Project Structure

```
simple-rag-langchain/
├── 01_Introduction_and_Fundamentals.ipynb    # Start here!
├── 02_Document_Loaders.ipynb
├── 03_Text_Splitting_Strategies.ipynb
├── 04_Embeddings_and_Vector_Representations.ipynb
├── 05_Vector_Stores.ipynb
├── 06_Retrieval_Strategies.ipynb
├── 07_Complete_RAG_Pipeline.ipynb
│
├── sample_data/                              # Example files for learning
│   ├── products.csv                          # Product catalog data
│   ├── api_response.json                     # API response example
│   ├── blog_post.html                        # HTML blog post
│   └── notes.txt                             # Study notes
│
├── requirements.txt                          # Python dependencies
├── .env.example                              # Environment template
├── .env                                      # Your API keys (create this)
├── README.md                                 # This file
│
└── Generated during course:
    ├── faiss_index/                          # Persisted FAISS vector store
    ├── chroma_db/                            # Persisted Chroma database
    └── rag_vectorstore/                      # RAG pipeline vector store
```

---

## 🎓 Learning Path

### For Beginners (New to LangChain/RAG)

1. **Start with Notebook 01** - Understand LangChain basics
2. **Work through 02-03** - Learn data loading and processing
3. **Practice exercises** in each notebook before moving forward
4. **Skip advanced sections** (marked 🎓 INTERMEDIATE) on first pass
5. **Complete Notebook 07** - Build your first RAG app
6. **Return to advanced sections** later

### For Intermediate Students (Have LLM experience)

1. **Skim Notebook 01** - Review LCEL syntax
2. **Focus on Notebooks 04-07** - Advanced concepts
3. **Complete all 🎓 INTERMEDIATE sections**
4. **Try practice exercises** at the end of each notebook
5. **Build a custom RAG app** with your own data

---

## 📚 Notebook Highlights

### Notebook 01: Introduction & Fundamentals
- What is LangChain and why use it?
- LCEL (LangChain Expression Language) explained
- First LLM call with prompt templates
- Comparison: LangChain vs traditional ML pipelines

### Notebook 02: Document Loaders
- **PDF**: PyPDFLoader for research papers
- **CSV**: Product catalogs and structured data
- **JSON**: API responses with jq queries
- **HTML**: Web scraping with BeautifulSoup
- **Batch**: DirectoryLoader for multiple files

### Notebook 03: Text Splitting
- RecursiveCharacterTextSplitter (recommended default)
- Chunk size optimization (500 vs 1000 vs 2000)
- Overlap strategies (10% vs 20% vs 30%)
- HTML and JSON structure-aware splitting

### Notebook 04: Embeddings
- OpenAI text-embedding-3-small (1536 dimensions)
- Google Gemini embedding-001 (768 dimensions)
- Cosine similarity calculations
- Model comparison and selection guide

### Notebook 05: Vector Stores
- InMemoryVectorStore (testing)
- FAISS (production, speed)
- Chroma (persistent, metadata filtering)
- When to use which store

### Notebook 06: Retrieval Strategies
- Similarity search (default)
- MMR for diverse results
- Custom retrievers with @chain decorator
- Retrieval with scores for debugging

### Notebook 07: Complete RAG Pipeline
- End-to-end implementation
- LCEL chain building
- Error handling patterns
- Streaming responses
- Production best practices checklist

---

## 🎯 What Makes This Course Special

✅ **Mixed-Level Approach**
- Clear markers for BEGINNER (🔰) and INTERMEDIATE (🎓) sections
- Detailed comments explain every line
- Multiple examples per concept

✅ **Latest LangChain 1.0.5+ Syntax**
- Uses LCEL (pipe operator `|`)
- Modern `.invoke()` methods
- Proper package imports

✅ **Multiple File Formats**
- PDF, CSV, JSON, HTML examples
- Real sample data files included
- Practical, real-world scenarios

✅ **Hands-On Learning**
- Practice exercises at end of each notebook
- Incremental complexity
- Build complete projects

✅ **Production-Ready Code**
- Error handling
- Best practices
- Cost optimization tips
- Persistence strategies

---

## 💡 Sample Data Included

The course includes ready-to-use sample data:

- **products.csv**: 15 products with descriptions (for CSVLoader)
- **api_response.json**: 5 AI articles (for JSONLoader)
- **blog_post.html**: Complete blog post about RAG (for WebBaseLoader)
- **notes.txt**: LangChain study notes (for TextLoader)

Students can immediately start learning without hunting for data files!

---

## 🔧 Customization Guide

### Use Your Own Documents

```python
# In any notebook, replace sample files:
pdf_path = "your_document.pdf"
csv_path = "your_data.csv"
```

### Change Embedding Model

```python
# Cost-effective (Notebook 04):
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Higher quality:
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Free/local:
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
```

### Adjust Chunk Size

```python
# Notebook 03 - Test different sizes:
for size in [500, 1000, 1500]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.2)  # 20% overlap
    )
```

### Select LLM Model

```python
# Fast & cheap (testing):
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Best quality (production):
llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09")
```

---

## 💰 Cost Estimates

### Per Notebook (approximate):

| Notebook | OpenAI API Cost | Notes |
|----------|----------------|-------|
| 01 | $0.01-0.05 | LLM calls only |
| 02 | $0.00 | Document loading (no API) |
| 03 | $0.00 | Text splitting (no API) |
| 04 | $0.05-0.10 | Embedding examples |
| 05 | $0.10-0.20 | Vector store creation |
| 06 | $0.05-0.10 | Retrieval testing |
| 07 | $0.20-0.50 | Complete RAG pipeline |

**Total Course Cost:** ~$0.50-1.00 with sample data

**With your own data (1000 pages):**
- Embeddings: ~$0.50
- Testing queries: ~$1.00-2.00
- Total: ~$1.50-2.50

**💡 Cost-Saving Tips:**
1. Use GPT-3.5-Turbo for learning ($10x cheaper than GPT-4)
2. Persist vector stores to avoid re-embedding
3. Test with small datasets first
4. Use HuggingFace embeddings (free) for experimentation

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. "Module not found" errors

```bash
# Ensure all packages are installed:
pip install --upgrade -r requirements.txt

# Verify LangChain version:
python -c "import langchain; print(langchain.__version__)"
# Should be 1.0.5
```

#### 2. "OPENAI_API_KEY not found"

```bash
# Check .env file exists:
ls -la .env

# Verify content:
cat .env

# Ensure load_dotenv() is called in notebook
```

#### 3. "allow_dangerous_deserialization" error

This is normal when loading FAISS indices. The notebooks include the required parameter:

```python
vectorstore = FAISS.load_local(
    path,
    embeddings,
    allow_dangerous_deserialization=True  # Required!
)
```

#### 4. Out of memory errors

```python
# Use lazy loading for large files:
for page in loader.lazy_load():
    process_page(page)

# Reduce chunk size:
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
```

#### 5. Slow notebook execution

```python
# 1. Reuse persisted vector stores
if Path("./faiss_index").exists():
    vectorstore = FAISS.load_local(...)  # Fast!

# 2. Use smaller k for retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Use GPT-3.5-Turbo instead of GPT-4
llm = ChatOpenAI(model="gpt-3.5-turbo")
```

#### 6. Python version issues

```bash
# Check Python version:
python --version

# Recommended: 3.9, 3.10, or 3.11
# Python 3.12+ may have package compatibility issues
```

---

## 🎯 Practice Exercises

Each notebook includes exercises:

### Beginner Exercises
- Load different file types
- Experiment with chunk sizes
- Test similarity searches
- Build simple chains

### Intermediate Exercises
- Multi-format loaders
- Custom retrievers
- Hybrid search strategies
- Production error handling

### Advanced Projects
- Build RAG for your domain
- Implement caching
- Add conversation memory
- Deploy to production

---

## 📚 Additional Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)
- [OpenAI API Docs](https://platform.openai.com/docs)

### Learning Resources
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [r/LangChain on Reddit](https://www.reddit.com/r/LangChain/)
- Stack Overflow: Tag `langchain`

---

## 🚀 Next Steps After Course

1. ✅ **Build with your own data** - Apply to your domain
2. ✅ **Add advanced features**:
   - Conversation memory
   - Hybrid search (vector + keyword)
   - Re-ranking models
   - Query transformation
3. ✅ **Deploy to production**:
   - Web interface (Streamlit/Gradio)
   - API with FastAPI
   - Cloud deployment (AWS/GCP/Azure)
4. ✅ **Optimize performance**:
   - Caching strategies
   - Batch processing
   - Cost monitoring
5. ✅ **Explore advanced topics**:
   - Agents and tools
   - Multi-modal RAG (images, audio)
   - Fine-tuning embeddings
   - Evaluation frameworks (RAGAS)

---

## 📝 Course Updates

**Version:** 1.0.0 (November 2025)

**LangChain Version:** 1.0.5+

**What's Included:**
- ✅ 7 comprehensive notebooks
- ✅ 4 sample data files
- ✅ Complete requirements.txt
- ✅ Production-ready code examples
- ✅ Mixed-level teaching approach
- ✅ Practice exercises



---


## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - Framework
- [OpenAI](https://openai.com/) - LLMs and Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector Search
- [Chroma](https://www.trychroma.com/) - Vector Database

---

## 📧 Support

- **Issues**: Found a bug? [Open an issue](https://github.com/your-repo/issues)
- **Questions**: Use GitHub Discussions
- **Contributions**: Pull requests welcome!

---

**Ready to master RAG with LangChain? Start with Notebook 01! 🚀**

Last Updated: November 2025
