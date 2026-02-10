# Supervised Learning - IMDB Sentiment Analysis

## Project Overview

This notebook implements a complete sentiment analysis pipeline for IMDB movie reviews, classifying them as positive or negative using machine learning and Natural Language Processing techniques.

## Dataset Information

- **Name**: IMDB Dataset of 50K Movie Reviews
- **Size**: 50,000 reviews
- **Distribution**: Balanced (25,000 positive, 25,000 negative)
- **Format**: CSV with 'review' and 'sentiment' columns
- **Source**: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Pipeline Stages

### 1. Data Loading & Exploration
- Load IMDB dataset
- Initial data inspection
- Check for missing values
- Analyze class distribution

### 2. Exploratory Data Analysis (EDA)
- Review length distribution
- Word frequency analysis
- Sentiment distribution
- Common words in positive vs negative reviews
- Word cloud visualization

### 3. Text Preprocessing

#### Cleaning Steps:
1. **HTML Tag Removal**: Strip HTML tags using regex
2. **Special Character Removal**: Keep only letters and spaces
3. **Lowercasing**: Convert all text to lowercase
4. **Tokenization**: Split text into words
5. **Stopword Removal**: Remove common English words (using NLTK)
6. **Lemmatization**: Reduce words to base form (using WordNetLemmatizer)

#### Feature Extraction:
- **TF-IDF Vectorization**: Convert text to numerical features
  - Captures word importance
  - Considers document frequency
  - Creates sparse matrix representation

#### Dimensionality Reduction:
- **TruncatedSVD**: Reduce feature space
  - Improves computational efficiency
  - Reduces noise
  - Prevents overfitting

### 4. Model Training

Typical models used:
- **Logistic Regression**: Baseline linear model
- **Naive Bayes**: Probabilistic classifier
- **Support Vector Machine (SVM)**: Margin-based classifier
- **Random Forest**: Ensemble method
- **Decision Tree**: Tree-based classifier

### 5. Model Evaluation

Metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed prediction breakdown

## Key Libraries Used

```python
import pandas as pd                    # Data manipulation
import numpy as np                     # Numerical operations
import nltk                           # NLP toolkit
import matplotlib.pyplot as plt       # Plotting
import seaborn as sns                 # Statistical visualization
from wordcloud import WordCloud       # Word cloud generation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
```

## Running the Notebook

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
```

### NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Dataset
Place `IMDB Dataset.csv` in the same directory as the notebook.

### Execution
```bash
jupyter notebook supervised.ipynb
```

## Expected Results

- **Training Accuracy**: ~85-95%
- **Test Accuracy**: ~80-90%
- **F1-Score**: ~0.80-0.90

Results may vary based on:
- Preprocessing parameters
- Vectorization settings
- Model hyperparameters
- Train/test split ratio

## Preprocessing Example

```python
# Before preprocessing:
"This movie was <b>AMAZING</b>!!! I loved it so much!!!"

# After preprocessing:
"movie amazing love much"
```

## Insights

### Common Positive Words:
- excellent, great, amazing, wonderful
- love, perfect, best, brilliant

### Common Negative Words:
- bad, worst, terrible, awful
- boring, waste, poor, disappointing

## Troubleshooting

### Issue: NLTK Data Not Found
**Solution**: Run the NLTK download commands

### Issue: Out of Memory
**Solution**: 
- Reduce max_features in TfidfVectorizer
- Use smaller n-gram range
- Sample the dataset

### Issue: Low Accuracy
**Solution**:
- Improve preprocessing
- Try different vectorization parameters
- Experiment with different models
- Adjust hyperparameters

## Performance Optimization

1. **Vectorization**:
   - `max_features=5000-10000` for faster training
   - `ngram_range=(1,2)` for better context

2. **Dimensionality Reduction**:
   - Use TruncatedSVD with n_components=100-300

3. **Model Selection**:
   - Logistic Regression: Fast and effective
   - SVM: Best accuracy but slower
   - Naive Bayes: Fast but lower accuracy

## Extensions & Improvements

- Deep learning models (LSTM, GRU, BERT)
- Ensemble methods
- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Feature engineering (POS tags, sentiment lexicons)
- Multi-class classification (1-5 star ratings)

## References

- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
