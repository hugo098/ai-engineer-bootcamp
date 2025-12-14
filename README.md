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

### Machine Learning
- **scikit-learn** - ML algorithms and tools
  - TF-IDF & Count Vectorizers
  - Logistic Regression
  - SGD Classifier
  - Model evaluation metrics

### Deep Learning (Transformers)
- Pre-trained transformer models for advanced NLP tasks

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn
pip install spacy nltk gensim vaderSentiment
pip install scikit-learn transformers
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

## 🎯 Key Concepts Covered

- Text preprocessing and cleaning
- Tokenization strategies
- Feature extraction (TF-IDF, Count Vectorization)
- Sentiment analysis (rule-based and ML-based)
- Named Entity Recognition
- Part-of-Speech tagging
- Topic modeling
- Text classification with machine learning
- Working with pre-trained transformer models

## 📝 Project: Fake News Classifier

A complete end-to-end machine learning project demonstrating:
- Data loading and exploration
- Text preprocessing pipeline
- Feature engineering
- Model training (Logistic Regression, SGD)
- Model evaluation and metrics
- Real-world application of NLP techniques

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
