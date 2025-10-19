# Clustering Analysis

This directory contains comprehensive implementations and analysis of unsupervised learning algorithms, specifically focusing on clustering techniques. The implementation demonstrates various clustering methods, evaluation metrics, and practical applications in customer segmentation.

## 📋 Overview

Clustering is a fundamental unsupervised learning technique that groups similar data points together without prior knowledge of group labels. This implementation provides:

- **Multiple clustering algorithms** including K-Means and DBSCAN
- **Comprehensive evaluation metrics** for cluster quality assessment
- **Advanced visualization techniques** including dimensionality reduction
- **Real-world application** in customer segmentation analysis
- **Comparative analysis** of different clustering approaches

## 🎯 Clustering Algorithms

### K-Means Clustering
A centroid-based clustering algorithm that partitions data into k clusters.

#### Algorithm Steps
1. **Initialize**: Randomly place k centroids
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as cluster means
4. **Iterate**: Repeat until convergence

#### Implementation Features
- **Multiple Initialization Methods**: K-means++, random initialization
- **Convergence Criteria**: Tolerance-based stopping conditions
- **Optimal K Selection**: Elbow method and silhouette analysis
- **Scalability**: Efficient implementation for large datasets

```python
class KMeansCluster:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
    
    def fit(self, X):
        # Initialize centroids
        self.centroids = self._init_centroids(X)
        
        for i in range(self.max_iters):
            # Assign points to clusters
            distances = self._calculate_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = self._update_centroids(X)
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids, rtol=self.tol):
                break
                
            self.centroids = new_centroids
        
        return self
```

### DBSCAN (Density-Based Spatial Clustering)
A density-based clustering algorithm that can find arbitrarily shaped clusters and identify outliers.

#### Algorithm Concepts
- **Core Points**: Points with sufficient neighbors within epsilon radius
- **Border Points**: Non-core points within epsilon of core points
- **Noise Points**: Points that are neither core nor border points
- **Density Connectivity**: Clusters formed by density-connected core points

#### Key Parameters
- **Epsilon (ε)**: Maximum distance between points in the same neighborhood
- **MinPts**: Minimum number of points required to form a dense region
- **Distance Metric**: Euclidean, Manhattan, or custom distance functions

```python
class DBSCANCluster:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
    
    def fit(self, X):
        self.labels = np.full(len(X), -1)  # Initialize as noise
        cluster_id = 0
        
        for i, point in enumerate(X):
            if self.labels[i] != -1:  # Already processed
                continue
                
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        
        return self
```

## 📊 Dataset: Customer Segmentation

### Dataset Overview
The clustering analysis uses a comprehensive customer dataset for market segmentation:

#### Customer Features
- **Demographics**: Age, education, marital status, income
- **Purchasing Behavior**: Spending on different product categories
- **Engagement Metrics**: Website visits, campaign responses
- **Temporal Patterns**: Recency of purchases, customer tenure
- **Geographic Information**: Location-based features

#### Dataset Characteristics
- **Size**: Thousands of customer records
- **Features**: 20+ customer attributes
- **Data Types**: Mixed numerical and categorical variables
- **Business Context**: Real-world retail/e-commerce scenario
- **Challenges**: Missing values, outliers, feature scaling needs

### Data Preprocessing Pipeline
```python
def preprocess_customer_data(df):
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Feature engineering
    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    
    df['Customer_Age'] = 2024 - df['Year_Birth']
    df['Customer_Tenure'] = df['Dt_Customer'].apply(lambda x: (pd.Timestamp.now() - pd.to_datetime(x)).days)
    
    # Categorical encoding
    df = pd.get_dummies(df, columns=['Education', 'Marital_Status'])
    
    # Feature scaling
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df
```

## 🔍 Evaluation Metrics

### Internal Validation Metrics
These metrics evaluate cluster quality without external labels:

#### Silhouette Score
Measures how similar points are to their own cluster compared to other clusters.
```python
def silhouette_analysis(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    return {
        'average_score': silhouette_avg,
        'sample_scores': sample_silhouette_values,
        'interpretation': 'Higher is better (range: -1 to 1)'
    }
```

#### Within-Cluster Sum of Squares (WCSS)
Measures compactness of clusters (used in elbow method).
```python
def calculate_wcss(X, centroids, labels):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss
```

#### Davies-Bouldin Index
Measures average similarity between clusters (lower is better).
```python
def davies_bouldin_score(X, labels):
    n_clusters = len(np.unique(labels))
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    
    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                ratio = (intra_cluster_distance(X, labels, i) + 
                        intra_cluster_distance(X, labels, j)) / \
                       inter_cluster_distance(centroids, i, j)
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    return db_index / n_clusters
```

### Cluster Validation Techniques

#### Elbow Method
Determines optimal number of clusters by finding the "elbow" in WCSS curve.
```python
def elbow_method(X, max_k=10):
    wcss_values = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss_values, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    
    return wcss_values
```

#### Silhouette Analysis
Comprehensive analysis of cluster separation quality.
```python
def plot_silhouette_analysis(X, range_n_clusters):
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fit clustering algorithm
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        # Plot silhouette plot
        y_lower = 10
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            y_lower = y_upper + 10
        
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_title(f'Silhouette Plot for {n_clusters} Clusters')
        
        # Plot clusters
        colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, c=colors, alpha=0.6)
        ax2.set_title(f'Cluster Visualization for {n_clusters} Clusters')
        
        plt.tight_layout()
        plt.show()
```

## 📈 Dimensionality Reduction and Visualization

### UMAP (Uniform Manifold Approximation and Projection)
Advanced dimensionality reduction technique for cluster visualization.

```python
def umap_visualization(X, labels, n_components=2):
    # Apply UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(X)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('UMAP Visualization of Clusters')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Add cluster centers if available
    if hasattr(labels, 'cluster_centers_'):
        centers_embedded = reducer.transform(labels.cluster_centers_)
        plt.scatter(centers_embedded[:, 0], centers_embedded[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
    
    plt.show()
    
    return embedding
```

### t-SNE Visualization
Alternative dimensionality reduction for cluster analysis.
```python
def tsne_visualization(X, labels, perplexity=30):
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
    
    return embedding
```

## 💼 Business Applications

### Customer Segmentation Analysis
```python
def analyze_customer_segments(df, cluster_labels):
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Segment profiles
    segment_profiles = df_clustered.groupby('Cluster').agg({
        'Customer_Age': ['mean', 'std'],
        'Income': ['mean', 'std'],
        'Total_Spending': ['mean', 'std'],
        'Recency': ['mean', 'std'],
        'NumWebPurchases': ['mean', 'std']
    }).round(2)
    
    # Segment sizes
    segment_sizes = df_clustered['Cluster'].value_counts().sort_index()
    
    # Business insights
    insights = {
        'High-Value Customers': 'Cluster with highest spending and income',
        'Price-Sensitive': 'Cluster with low spending but high purchase frequency',
        'Occasional Buyers': 'Cluster with infrequent but substantial purchases',
        'New Customers': 'Cluster with recent acquisition and growing engagement'
    }
    
    return {
        'profiles': segment_profiles,
        'sizes': segment_sizes,
        'insights': insights
    }
```

### Marketing Strategy Recommendations
```python
def generate_marketing_strategies(cluster_analysis):
    strategies = {}
    
    for cluster_id, profile in cluster_analysis['profiles'].iterrows():
        if profile[('Total_Spending', 'mean')] > 1000:
            strategies[cluster_id] = {
                'segment_name': 'Premium Customers',
                'strategy': 'VIP programs, exclusive offers, premium products',
                'channels': ['Email', 'Direct mail', 'Personal consultation'],
                'budget_allocation': 'High'
            }
        elif profile[('NumWebPurchases', 'mean')] > 5:
            strategies[cluster_id] = {
                'segment_name': 'Digital Natives',
                'strategy': 'Online promotions, app-based offers, social media',
                'channels': ['Social media', 'Mobile app', 'Online ads'],
                'budget_allocation': 'Medium'
            }
        else:
            strategies[cluster_id] = {
                'segment_name': 'Value Seekers',
                'strategy': 'Discount campaigns, loyalty programs, bulk offers',
                'channels': ['Email', 'SMS', 'Print ads'],
                'budget_allocation': 'Low-Medium'
            }
    
    return strategies
```

## 🎓 Learning Objectives

After working through this clustering implementation, you will understand:

1. **Unsupervised Learning**: Learning patterns without labeled data
2. **Clustering Algorithms**: Different approaches to grouping similar data
3. **Distance Metrics**: Various ways to measure similarity between points
4. **Cluster Validation**: Methods to evaluate clustering quality
5. **Dimensionality Reduction**: Visualization of high-dimensional clusters
6. **Business Applications**: Practical use cases in customer analytics
7. **Parameter Tuning**: Optimizing clustering algorithm parameters

## 📁 Files Description

- `clustering.ipynb`: Complete clustering analysis and customer segmentation
- `data/customers.csv`: Customer dataset for segmentation analysis
- Includes:
  - Data preprocessing and exploration
  - Multiple clustering algorithm implementations
  - Comprehensive evaluation metrics
  - Advanced visualization techniques
  - Business insights and recommendations

## 💻 Usage Examples

### Basic Clustering Analysis
```python
# Load and preprocess data
df = pd.read_csv('data/customers.csv')
X_processed = preprocess_customer_data(df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_processed)

# Evaluate clustering quality
silhouette_avg = silhouette_score(X_processed, cluster_labels)
print(f"Average Silhouette Score: {silhouette_avg:.3f}")

# Visualize results
umap_embedding = umap_visualization(X_processed, cluster_labels)
```

### Comprehensive Clustering Pipeline
```python
# Determine optimal number of clusters
wcss_values = elbow_method(X_processed, max_k=10)
plot_silhouette_analysis(X_processed, range(2, 8))

# Compare multiple algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=4)
}

results = {}
for name, algorithm in algorithms.items():
    labels = algorithm.fit_predict(X_processed)
    if len(np.unique(labels)) > 1:  # Valid clustering
        results[name] = {
            'labels': labels,
            'silhouette_score': silhouette_score(X_processed, labels),
            'n_clusters': len(np.unique(labels))
        }

# Business analysis
best_algorithm = max(results.keys(), key=lambda k: results[k]['silhouette_score'])
best_labels = results[best_algorithm]['labels']
business_insights = analyze_customer_segments(df, best_labels)
marketing_strategies = generate_marketing_strategies(business_insights)
```

## 🚀 Getting Started

1. **Open the Jupyter notebook**: `clustering.ipynb`
2. **Load the customer dataset** from the data directory
3. **Follow the preprocessing steps** for data preparation
4. **Run clustering algorithms** with different parameters
5. **Evaluate cluster quality** using multiple metrics
6. **Visualize results** with dimensionality reduction
7. **Interpret business insights** and develop strategies

## 🔬 Advanced Extensions

### Algorithm Enhancements
- **Fuzzy C-Means**: Soft clustering with membership probabilities
- **Gaussian Mixture Models**: Probabilistic clustering approach
- **Spectral Clustering**: Graph-based clustering for complex shapes
- **Hierarchical Clustering**: Tree-based clustering with dendrograms

### Evaluation Improvements
- **Consensus Clustering**: Combine multiple clustering results
- **Stability Analysis**: Assess clustering robustness across samples
- **External Validation**: Compare with ground truth when available
- **Statistical Significance**: Test clustering quality significance

### Business Applications
- **Dynamic Segmentation**: Time-based customer segment evolution
- **Predictive Clustering**: Forecast future segment membership
- **Multi-dimensional Segmentation**: Combine multiple data sources
- **Real-time Clustering**: Streaming data clustering for live insights

---

This clustering implementation provides a comprehensive foundation for understanding unsupervised learning while demonstrating practical applications in customer analytics and business intelligence.