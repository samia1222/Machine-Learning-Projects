# Leveraging Machine Learning Algorithms to Optimize Sentiment Analysis and Movie Recommendation Systems
A collection of Jupyter notebooks demonstrating both supervised and unsupervised machine learning techniques. It includes supervised learning algorithms like regression and classification, as well as unsupervised learning methods such as clustering and dimensionality reduction for analyzing unlabeled data.
# Machine Learning Projects Collection

A collection of machine learning projects demonstrating supervised and unsupervised learning techniques for NLP and recommendation systems.

## 📚 Projects Overview

### 1. Supervised Learning - IMDB Sentiment Analysis
Binary classification of movie reviews using Natural Language Processing (NLP) techniques.

### 2. Unsupervised Learning - Movie Recommendation System
Collaborative filtering-based movie recommendation system using clustering algorithms.

---

## 🎯 Project 1: IMDB Sentiment Analysis (Supervised Learning)

### Overview
A sentiment analysis system that classifies IMDB movie reviews as positive or negative using machine learning algorithms and NLP preprocessing techniques.

### Dataset
- **Source**: IMDB Dataset (50,000 movie reviews)
- **Features**: Review text
- **Target**: Sentiment (Positive/Negative)
- **Format**: CSV file

### Key Features
- **Text Preprocessing**: Cleaning, tokenization, stopword removal, lemmatization
- **Feature Extraction**: TF-IDF Vectorization
- **Dimensionality Reduction**: TruncatedSVD
- **Visualization**: Word clouds, distribution analysis
- **Multiple ML Models**: Classification algorithms comparison

### Preprocessing Pipeline
1. Text cleaning (removing HTML, special characters)
2. Lowercasing
3. Stopword removal
4. Lemmatization using WordNet
5. TF-IDF vectorization
6. Dimensionality reduction

### Technologies Used
- **Python Libraries**:
  - pandas, numpy - Data manipulation
  - NLTK - Natural Language Processing
  - scikit-learn - Machine Learning
  - matplotlib, seaborn - Visualization
  - WordCloud - Text visualization

### Exploratory Data Analysis
- Review length distribution
- Word frequency analysis
- Word clouds for positive/negative sentiments
- Class balance analysis

### Models (Likely included)
- Logistic Regression
- Naive Bayes
- Support Vector Machines (SVM)
- Random Forest
- Decision Trees

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🎬 Project 2: Movie Recommendation System (Unsupervised Learning)

### Overview
A collaborative filtering-based recommendation system that suggests movies to users based on rating patterns and user similarities.

### Dataset
- **Source**: MovieLens Dataset
- **Files**:
  - `ratings.csv` - User ratings for movies
  - `movies.csv` - Movie metadata (title, genres)
- **Size**: ~60 million entries (after genre explosion)
- **Sampling**: 5% sample used for computational efficiency

### Key Features
- **User-Item Matrix**: Sparse matrix of user ratings
- **Collaborative Filtering**: User-based and item-based approaches
- **Clustering**: Grouping similar users/movies
- **Genre Analysis**: Multi-label genre processing
- **Scalable Design**: Sampling strategy for large datasets

### Preprocessing Steps
1. Data loading and inspection
2. Missing value handling
3. Genre column explosion (one-hot encoding)
4. Strategic sampling (5% of dataset)
5. User-item matrix construction
6. Normalization

### Technologies Used
- **Python Libraries**:
  - pandas, numpy - Data manipulation
  - scikit-learn - Clustering and preprocessing
  - matplotlib, seaborn - Visualization
  - scipy - Sparse matrix operations

### Clustering Algorithms (Likely)
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- User/Movie similarity computation

### Recommendation Approach
- Collaborative filtering based on user ratings
- Genre-based filtering
- Hybrid recommendations

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/ml-projects.git
   cd ml-projects
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
   ```

3. **Download NLTK data** (for supervised learning)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

4. **Download datasets**
   - **IMDB Dataset**: Place `IMDB Dataset.csv` in project root
   - **MovieLens Dataset**: Place `ratings.csv` and `movies.csv` in project root

### Running the Projects

#### Supervised Learning (Sentiment Analysis)
```bash
jupyter notebook supervised.ipynb
```

#### Unsupervised Learning (Recommendation System)
```bash
jupyter notebook unsupervised_learning.ipynb
```

---

## 📊 Project Structure

```
├── supervised.ipynb              # IMDB sentiment analysis notebook
├── unsupervised_learning.ipynb   # Movie recommendation notebook
├── IMDB Dataset.csv              # IMDB reviews dataset (not included)
├── ratings.csv                   # MovieLens ratings (not included)
├── movies.csv                    # MovieLens movies (not included)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📈 Results & Insights

### Supervised Learning Results
- Successfully classifies movie reviews into positive/negative sentiments
- Achieves high accuracy through proper text preprocessing
- TF-IDF vectorization effectively captures important features
- Word clouds reveal key sentiment indicators

### Unsupervised Learning Results
- Efficient sampling strategy handles large-scale data
- Clustering reveals user preference patterns
- Genre explosion provides rich feature space
- Collaborative filtering generates personalized recommendations

---

## 🛠️ Technical Highlights

### Data Preprocessing
- **Text Cleaning**: Robust preprocessing pipeline for NLP
- **Dimensionality Reduction**: TruncatedSVD for sparse matrices
- **Sampling Strategy**: Efficient handling of large datasets
- **Feature Engineering**: TF-IDF, genre encoding

### Machine Learning Techniques
- **Supervised**: Classification algorithms with cross-validation
- **Unsupervised**: Clustering and similarity-based recommendations
- **Evaluation**: Comprehensive metrics and visualization

### Best Practices
- Warning suppression for clean output
- Modular code structure
- Clear documentation
- Reproducible results

---

## 📚 Key Learnings

1. **Text Preprocessing** is crucial for NLP tasks
2. **TF-IDF** effectively captures word importance
3. **Sampling** is necessary for large datasets
4. **Collaborative Filtering** works well for recommendations
5. **Visualization** aids in understanding data patterns

---

## 🔮 Future Enhancements

### Supervised Learning
- [ ] Deep learning models (LSTM, BERT)
- [ ] Hyperparameter tuning with GridSearch
- [ ] Multi-class sentiment analysis (1-5 stars)
- [ ] Real-time sentiment prediction API
- [ ] Aspect-based sentiment analysis

### Unsupervised Learning
- [ ] Matrix factorization (SVD, NMF)
- [ ] Deep learning recommendations (Neural CF)
- [ ] Hybrid recommendation system
- [ ] Content-based filtering
- [ ] Real-time recommendation engine
- [ ] A/B testing framework

---

## 📝 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
jupyter>=1.0.0
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Machine Learning projects for NLP and Recommendation Systems

---

## 🔗 Resources

### Datasets
- [IMDB Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

### Documentation
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)
- [pandas Documentation](https://pandas.pydata.org/)

### Tutorials
- [Sentiment Analysis Tutorial](https://realpython.com/sentiment-analysis-python/)
- [Recommendation Systems Guide](https://realpython.com/build-recommendation-engine-collaborative-filtering/)

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: The datasets (`IMDB Dataset.csv`, `ratings.csv`, `movies.csv`) are not included in this repository due to size constraints. Please download them from the sources mentioned above.
