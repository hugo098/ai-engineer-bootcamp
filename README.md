# AI Engineer Bootcamp

A comprehensive learning repository for AI and Natural Language Processing (NLP) concepts, featuring hands-on Jupyter notebooks and practical projects.

## 📚 Repository Structure

### NLP Module
Progressive learning path covering fundamental to advanced NLP techniques:

#### Text Preprocessing Fundamentals
1. **[Lowercasing](NLP/1-Lowercasing.ipynb)** - Text normalization basics
2. **[Stop Words](NLP/2-StopWords.ipynb)** - Removing common words
3. **[Regular Expressions](NLP/3-RegularExpressions.ipynb)** - Pattern matching and text manipulation
4. **[Tokenization](NLP/4-Tokenization.ipynb)** - Breaking text into tokens
5. **[N-Grams](NLP/5-NGrams.ipynb)** - Multi-word sequences and context
6. **[Text Preprocessing Hands-On](NLP/6-TextPreprocessingHandsOn.ipynb)** - Practical preprocessing exercises

#### Advanced NLP Techniques
7. **[Parts of Speech](NLP/7-PartsOfSpeech.ipynb)** - POS tagging and grammatical analysis
8. **[Named Entity Recognition](NLP/8-NameEntityRecognition.ipynb)** - Identifying entities in text
9. **[POS & NER Practical Task](NLP/9-PosNerPracticalTask.ipynb)** - Applied exercises

#### Sentiment Analysis & Classification
10. **[Rule-Based Sentiment Analysis](NLP/10-RuleBasedSentimentAnalysis.ipynb)** - Traditional sentiment approaches
11. **[Pre-Trained Transformers](NLP/11-PreTrainedTransformer.ipynb)** - Using modern transformer models
12. **[Sentiment Analysis Hands-On](NLP/12-SentimentAnalysisHandsOn.ipynb)** - Practical sentiment analysis

#### Text Vectorization & Modeling
13. **[Text Vectorization](NLP/13-TextVectorization.ipynb)** - Converting text to numerical representations
14. **[Topic Modeling](NLP/14-TopicModelling.ipynb)** - Discovering themes in documents
15. **[Logistic Regression Text Classifiers](NLP/15-LogisticRegressionTextClassifiers.ipynb)** - ML-based text classification

#### Projects
- **[Fake News Classifier](NLP/FakeNews/FakeNewsClassifier.ipynb)** - Complete ML project for detecting fake news

### LLM Module
Modern Large Language Models and transformer architectures:

#### Foundation Models & APIs
1. **[GPT Models](LLM/1-GptModels.ipynb)** - Introduction to GPT models and OpenAI API usage
2. **[LangChain Introduction](LLM/2-LangChainIntro.ipynb)** - LangChain framework fundamentals and chains
3. **[Hugging Face Transformers](LLM/3-HuggingFaceTransformers.ipynb)** - Working with the transformers library
4. **[BERT](LLM/4-BERT.ipynb)** - BERT model implementation and fine-tuning

#### Projects
- **[Text Classification with XLNet](LLM/TextClassificationXLNet/TextClassificationXLNet.ipynb)** - Complete pipeline for emotion classification
  - Fine-tuning XLNet-base-cased model
  - 4-class emotion detection (anger, fear, joy, sadness)
  - Training on labeled tweet dataset
  - Model evaluation and inference

### LangChain Module
Comprehensive framework for building LLM applications:

#### 1. Model Input/Output
Learn to work with LLM inputs and outputs:
- **[OpenAI API Setup](LangChain/1-ModelInput/0-OpenApi.ipynb)** - Configuration and basic API usage
- **[Model I/O Basics](LangChain/1-ModelInput/1-ModelInputOutput%20copy.ipynb)** - Input/output handling
- **[System & Human Messages](LangChain/1-ModelInput/2-SystemAndHumanMessages%20copy.ipynb)** - Chat message types
- **[AI Messages](LangChain/1-ModelInput/3-AIMessages.ipynb)** - Working with AI responses
- **[Prompt Templates](LangChain/1-ModelInput/4-PromptTemplates.ipynb)** - Creating reusable prompts
- **[Chat Prompt Templates](LangChain/1-ModelInput/5-ChatPromtTemplates.ipynb)** - Chat-specific templates
- **[Few-Shot Prompt Templates](LangChain/1-ModelInput/6-FewShotPromptTemplate.ipynb)** - Few-shot learning examples

#### 2. Output Parsers
Structured output handling:
- **[Output Parsers](LangChain/2-ModelOutput/1-OutputParsers.ipynb)** - Parsing and validating LLM outputs

#### 3. LangChain Expression Language (LCEL)
Build complex LLM chains with composable components:
- **[Piping](LangChain/3-LangChainExpressionLanguage/1-Piping.ipynb)** - Chain components together
- **[Batching](LangChain/3-LangChainExpressionLanguage/2-Batching.ipynb)** - Batch processing with chains
- **[Streaming](LangChain/3-LangChainExpressionLanguage/3-Streaming.ipynb)** - Real-time streaming responses
- **[Runnable & Runnable Sequence](LangChain/3-LangChainExpressionLanguage/4-RunnableAndRunnableSequence%20copy.ipynb)** - Core runnable components
- **[Runnable Passthrough](LangChain/3-LangChainExpressionLanguage/5-RunnablePassthrough.ipynb)** - Pass data through chains
- **[Runnable Parallel](LangChain/3-LangChainExpressionLanguage/6-RunnableParallel.ipynb)** - Parallel execution
- **[Runnable Passthrough Conserve Input](LangChain/3-LangChainExpressionLanguage/7-RunnablePassthroughConserveInput.ipynb)** - Preserve input data
- **[Runnable Lambda](LangChain/3-LangChainExpressionLanguage/8-RunnableLambda.ipynb)** - Custom functions in chains
- **[Chain Decorator](LangChain/3-LangChainExpressionLanguage/9-ChainDecorator.ipynb)** - Simplify chain creation

#### 4. Retrieval-Augmented Generation (RAG)
Build context-aware AI applications with document retrieval:
- **[Document Indexing](LangChain/4-RAG/1-Indexing.ipynb)** - Loading and indexing documents
- **[Document Splitting](LangChain/4-RAG/3-IndexingDocumentSplitting.ipynb)** - Chunking strategies
- **[Markdown Header Text Splitter](LangChain/4-RAG/4-IndexingMarkdownHeaderTextSplitter.ipynb)** - Structure-aware splitting
- **[Vector Store Management](LangChain/4-RAG/7-IndexingManagingDocsVectorstore.ipynb)** - Working with Chroma vector stores
- **[Maximal Marginal Relevance (MMR) Search](LangChain/4-RAG/8-RetrievalMarginalRelevanceSearch.ipynb)** - Diverse retrieval strategies
- **[Document Generation with Stuffing](LangChain/4-RAG/11-GenerationStuffingDocuments.ipynb)** - Context-aware response generation

### Python Introduction
- **[Python Variables](PythonIntro/PythonVariables.ipynb)** - Python programming basics

## 📊 Datasets

The repository includes several real-world datasets for practice:

- `bbc_news.csv` - BBC news articles
- `book_reviews_sample.csv` - Book reviews dataset
- `news_articles.csv` - General news articles
- `tripadvisor_hotel_reviews.csv` - Hotel reviews for sentiment analysis
- `fake_news_data.csv` - Labeled fake/real news dataset (in FakeNews folder)

## 🛠️ Technologies & Libraries

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization

### NLP Libraries
- **spaCy** - Industrial-strength NLP
- **NLTK** - Natural Language Toolkit
- **Gensim** - Topic modeling and document similarity
- **vaderSentiment** - Sentiment analysis

### LLM & Deep Learning
- **transformers** - Hugging Face transformers library
- **torch** - PyTorch deep learning framework
- **datasets** - Hugging Face datasets library
- **evaluate** - Model evaluation metrics
- **LangChain** - Framework for LLM applications
  - **langchain-openai** - OpenAI integration
  - **langchain-community** - Community integrations
  - **langchain-chroma** - Chroma vector store
  - **langchain-core** - Core LangChain components
- **OpenAI** - GPT models API
- **chromadb** - Vector database for embeddings
- **pypdf** - PDF document processing
- **docx2txt** - Word document processing
- **python-dotenv** - Environment variable management

### Machine Learning
- **scikit-learn** - ML algorithms and tools
  - TF-IDF & Count Vectorizers
  - Logistic Regression
  - SGD Classifier
  - Model evaluation metrics
  - Label encoding and preprocessing

### Deep Learning (Transformers)
- XLNet for sequence classification
- BERT and variants
- Pre-trained transformer models for advanced NLP tasks
- Model fine-tuning and transfer learning

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn
pip install spacy nltk gensim vaderSentiment
pip install scikit-learn transformers torch datasets evaluate
pip install langchain openai python-dotenv
pip install langchain-openai langchain-community langchain-chroma
pip install pypdf docx2txt chromadb
pip install jupyter notebook

# Download NLTK data
python -m nltk.downloader stopwords wordnet punkt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running Notebooks
```bash
# Navigate to the project directory
cd /path/to/AI

# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

## 📖 Learning Path

**Recommended order for beginners:**
1. Start with Python Introduction if new to Python
2. Follow NLP notebooks in numerical order (1-15)
3. Complete hands-on exercises (6, 9, 12)
4. Apply knowledge to the Fake News Classifier project
5. Explore LLM notebooks (1-4) for modern transformer models
6. Work on the XLNet emotion classification project
7. **Learn LangChain framework:**
   - Start with Model Input/Output (1-ModelInput)
   - Understand Output Parsers (2-ModelOutput)
   - Master LCEL components (3-LangChainExpressionLanguage)
   - Build RAG applications (4-RAG)

## 🎯 Key Concepts Covered

### NLP Fundamentals
- Text preprocessing and cleaning
- Tokenization strategies
- Feature extraction (TF-IDF, Count Vectorization)
- Sentiment analysis (rule-based and ML-based)
- Named Entity Recognition
- Part-of-Speech tagging
- Topic modeling
- Text classification with machine learning

### Large Language Models
- GPT models and OpenAI API integration
- LangChain framework for LLM applications
  - Prompt engineering and templates
  - Few-shot learning
  - Chain composition with LCEL
  - Streaming and batching
  - Custom functions with Runnable Lambda
- Hugging Face transformers ecosystem
- BERT architecture and applications
- XLNet for sequence classification
- Model fine-tuning and transfer learning
- Text preprocessing for transformer models
- Emotion classification with deep learning

### Retrieval-Augmented Generation (RAG)
- Document loading and indexing (PDF, DOCX)
- Text splitting strategies (character-based, markdown headers)
- Vector embeddings with OpenAI
- Vector stores (Chroma)
- Document retrieval (similarity search, MMR)
- Context-aware response generation
- Metadata preservation and filtering

## 📝 Project: Fake News Classifier

A complete end-to-end machine learning project demonstrating:
- Data loading and exploration
- Text preprocessing pipeline
- Feature engineering
- Model training (Logistic Regression, SGD)
- Model evaluation and metrics
- Real-world application of NLP techniques

## 📝 Project: XLNet Emotion Classification

A deep learning project using transformers:
- Fine-tuning XLNet-base-cased on emotion dataset
- Multi-class classification (anger, fear, joy, sadness)
- Working with Hugging Face datasets and trainers
- Text cleaning and tokenization for transformers
- Model evaluation and inference pipeline
- Saving and loading fine-tuned models

## 🤝 Contributing

This is a learning repository. Feel free to:
- Add new notebooks
- Improve existing examples
- Fix bugs or typos
- Add more datasets

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🎓**
