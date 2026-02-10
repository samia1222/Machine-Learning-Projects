# Unsupervised Learning - Movie Recommendation System

## Project Overview

This notebook implements a collaborative filtering-based movie recommendation system using unsupervised learning techniques on the MovieLens dataset.

## Dataset Information

- **Name**: MovieLens Dataset
- **Files**:
  - `ratings.csv`: User ratings for movies (userId, movieId, rating, timestamp)
  - `movies.csv`: Movie metadata (movieId, title, genres)
- **Original Size**: ~60 million entries (after genre explosion)
- **Sampled Size**: 5% of original dataset (for computational efficiency)
- **Source**: [MovieLens by GroupLens](https://grouplens.org/datasets/movielens/)

## Why Sampling?

The original dataset expands to approximately 60 million entries after exploding the genres column (multi-label to single-label conversion). To manage computational resources while maintaining statistical validity, a 5% stratified sample is used.

## Pipeline Stages

### 1. Data Loading & Inspection
- Load ratings and movies datasets
- Inspect data structure
- Check data types and missing values
- Understand the schema

### 2. Data Preprocessing

#### Steps:
1. **Genre Explosion**: Convert pipe-separated genres into individual rows
   - Before: "Action|Adventure|Sci-Fi"
   - After: Three separate rows (one per genre)

2. **Missing Value Handling**: Remove or impute missing data

3. **Data Merging**: Combine ratings with movie metadata

4. **Strategic Sampling**: 
   - Random sample of 5% of data
   - Maintains distribution
   - Enables faster experimentation

5. **User-Item Matrix Construction**:
   - Rows: Users
   - Columns: Movies
   - Values: Ratings (sparse matrix)

### 3. Exploratory Data Analysis (EDA)

Analysis includes:
- **Rating Distribution**: Histogram of rating values
- **User Activity**: Number of ratings per user
- **Movie Popularity**: Number of ratings per movie
- **Genre Analysis**: Most popular genres
- **Temporal Patterns**: Rating trends over time
- **Sparsity Analysis**: Matrix density calculation

### 4. Feature Engineering

- **Normalization**: Center ratings by user mean
- **Genre Encoding**: One-hot encoding of genres
- **User Features**: 
  - Average rating per user
  - Rating count per user
  - Favorite genres

- **Movie Features**:
  - Average rating per movie
  - Rating count per movie
  - Genre combinations

### 5. Clustering & Similarity

Unsupervised learning techniques:

#### User-Based Clustering:
- Group similar users based on rating patterns
- Algorithms: K-Means, Hierarchical Clustering
- Features: User rating vectors

#### Item-Based Clustering:
- Group similar movies
- Algorithms: K-Means, DBSCAN
- Features: Movie rating vectors, genres

#### Similarity Metrics:
- **Cosine Similarity**: Angle between rating vectors
- **Pearson Correlation**: Linear correlation
- **Euclidean Distance**: Geometric distance

### 6. Recommendation Generation

#### Collaborative Filtering:
1. **User-Based**: Find similar users, recommend their liked movies
2. **Item-Based**: Find similar movies to user's liked movies

#### Steps:
1. Identify user's rated movies
2. Find similar users/items
3. Predict ratings for unrated movies
4. Rank and recommend top-N movies

## Key Libraries Used

```python
import pandas as pd                    # Data manipulation
import numpy as np                     # Numerical operations
import matplotlib.pyplot as plt       # Plotting
import seaborn as sns                 # Statistical visualization
from sklearn.cluster import KMeans    # Clustering
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix   # Sparse matrix operations
from sklearn.metrics import silhouette_score
```

## Running the Notebook

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Dataset
Download and place these files in the same directory:
- `ratings.csv`
- `movies.csv`

### Execution
```bash
jupyter notebook unsupervised_learning.ipynb
```

## Expected Outputs

### Clustering Results:
- **Optimal K**: Determined via elbow method or silhouette analysis
- **User Clusters**: 5-10 distinct user groups
- **Movie Clusters**: 10-20 genre-based clusters

### Recommendations:
- Top-N movies for each user
- Similar movies for each movie
- Cluster-based recommendations

## Performance Metrics

### Clustering Evaluation:
- **Silhouette Score**: Cluster cohesion (0 to 1, higher is better)
- **Inertia**: Within-cluster sum of squares
- **Davies-Bouldin Index**: Cluster separation

### Recommendation Quality:
- **Coverage**: % of items that can be recommended
- **Diversity**: Variety in recommendations
- **Novelty**: Recommendation of less-known items

## Example Use Cases

### 1. User-Based Recommendation
```
User ID: 123
Liked Movies: ["The Matrix", "Inception", "Interstellar"]
Recommended: ["Blade Runner 2049", "Tenet", "The Prestige"]
```

### 2. Item-Based Recommendation
```
Movie: "The Shawshank Redemption"
Similar Movies:
1. The Green Mile
2. Forrest Gump
3. The Godfather
```

### 3. Cluster-Based Insight
```
Cluster 1: Action/Sci-Fi Enthusiasts
- 15% of users
- Prefer: High-budget blockbusters
- Average rating: 4.2/5
```

## Data Insights

### Common Patterns:
- **Popular Genres**: Drama, Comedy, Action
- **Rating Distribution**: Most ratings are 3-4 stars
- **Power Users**: 20% of users contribute 80% of ratings
- **Long Tail**: Many movies have few ratings

### Sparsity Challenge:
- **Matrix Density**: <1% (highly sparse)
- **Implication**: Most user-movie pairs have no rating
- **Solution**: Collaborative filtering handles sparsity well

## Troubleshooting

### Issue: Memory Error
**Solution**: 
- Reduce sample size further (3% or 1%)
- Use sparse matrix representations
- Process in batches

### Issue: Slow Clustering
**Solution**:
- Reduce number of features
- Use MiniBatchKMeans instead of KMeans
- Sample users/movies for clustering

### Issue: Poor Recommendations
**Solution**:
- Improve feature engineering
- Adjust similarity threshold
- Try hybrid approaches
- Include content-based features

## Optimization Strategies

1. **Data Sampling**:
   - Stratified sampling maintains distribution
   - 5% sample balances speed and accuracy

2. **Sparse Matrix Usage**:
   - Reduce memory footprint
   - Faster computations
   - Essential for large datasets

3. **Dimensionality Reduction**:
   - PCA or SVD on rating matrix
   - Reduces noise
   - Improves clustering

## Extensions & Improvements

- **Matrix Factorization**: SVD, NMF, ALS
- **Deep Learning**: Neural Collaborative Filtering
- **Hybrid Systems**: Combine collaborative + content-based
- **Context-Aware**: Include time, location, device
- **Implicit Feedback**: Use viewing history, not just ratings
- **Cold Start Solutions**: For new users/movies
- **Real-Time System**: Stream processing for live recommendations
- **A/B Testing**: Evaluate recommendation quality in production

## Advanced Topics

### Matrix Factorization (SVD)
- Decompose rating matrix into user and item latent factors
- Captures hidden patterns
- Better handles sparsity

### Neural Collaborative Filtering
- Deep neural networks for user-item interaction
- Learns complex non-linear patterns
- State-of-the-art performance

### Hybrid Recommendations
- Combine collaborative filtering with content-based
- Use movie metadata (genres, actors, directors)
- Improves recommendation quality

## References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering Guide](https://developers.google.com/machine-learning/recommendation)
- [Recommendation Systems Handbook](https://www.springer.com/gp/book/9781489976369)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

## Evaluation Considerations

### Online Metrics (Production):
- Click-through rate (CTR)
- Conversion rate
- User engagement time
- Diversity of consumption

### Offline Metrics (Development):
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)

---

**Note**: This project demonstrates foundational concepts. Production systems require additional considerations like scalability, real-time processing, and continuous learning.
