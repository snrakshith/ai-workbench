# Natural Language Processing for Machine Learning
## Comprehensive Tutorial Series

This repository contains a complete, hands-on tutorial series covering Natural Language Processing (NLP) fundamentals for Machine Learning applications.

## 📚 Course Structure

### Foundational NLP Topics
1. **00_Introduction_to_NLP_for_ML.ipynb** - Overview of NLP in ML and practical use cases
2. **01_Text_Preprocessing.ipynb** - Tokenization, stemming, lemmatization, stopwords
3. **02_Text_Representation.ipynb** - One-Hot Encoding, N-Grams, Bag of Words, TF-IDF
4. **03_POS_Tagging.ipynb** - Part of Speech tagging with NLTK
5. **04_Named_Entity_Recognition.ipynb** - NER concepts and implementation
6. **05_Word_Embeddings.ipynb** - Introduction to word embeddings and their benefits

### Advanced Topics (Module 13)
7. **06_Word2Vec.ipynb** - Word2Vec intuition, Skip-gram, and CBOW architectures

## 🚀 Setup Instructions

### 1. Clone or Download This Repository
```bash
cd /path/to/NLPML
```

### 2. Create and Activate Virtual Environment
```bash
# The virtual environment is already created
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

### 5. Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 6. Launch Jupyter Notebook
```bash
jupyter notebook
```

## 📖 How to Use These Notebooks

Each notebook is self-contained and includes:
- **Detailed Theory**: Step-by-step explanations with markdown
- **Code Examples**: Heavily commented code for clarity
- **Simple Examples**: Toy examples to understand concepts
- **Real-World Applications**: Practical use cases with actual data
- **Visualizations**: Basic plots to understand patterns
- **Practice Exercises**: End-of-section exercises (optional)

**Recommended Order**: Start with notebook 00 and proceed sequentially through 06.

## 🛠️ Technologies Used

- **NLTK**: Core NLP operations and classical techniques
- **spaCy**: Modern NLP processing and NER
- **Gensim**: Word embeddings and Word2Vec
- **scikit-learn**: Text vectorization (TF-IDF, BoW)
- **Matplotlib/Seaborn**: Data visualization
- **Pandas/NumPy**: Data manipulation

## 📝 Topics Covered

### Text Preprocessing
- Tokenization (word, sentence, custom)
- Basic NLP terminology
- Stemming algorithms (Porter, Snowball)
- Lemmatization
- Stopwords removal

### Text Representation
- One-Hot Encoding
- N-Grams (unigram, bigram, trigram)
- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)

### Advanced NLP Techniques
- Part of Speech (POS) Tagging
- Named Entity Recognition (NER)
- Word Embeddings concepts
- Word2Vec (Skip-gram and CBOW)

## 🎯 Learning Objectives

By completing this tutorial series, you will:
- Understand fundamental NLP concepts and terminology
- Master text preprocessing techniques
- Learn different text representation methods
- Implement POS tagging and NER systems
- Understand and train word embedding models
- Build practical NLP pipelines for ML applications

## 📧 Support

If you encounter any issues:
1. Ensure all dependencies are installed correctly
2. Verify NLTK data is downloaded
3. Check that spaCy language model is installed
4. Make sure you're using Python 3.8+

## 📄 License

This educational material is provided for learning purposes.

---

**Happy Learning! 🚀**
