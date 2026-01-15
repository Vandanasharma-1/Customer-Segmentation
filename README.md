# Customer Segmentation Analysis

This project performs customer segmentation using various unsupervised machine learning algorithms on an online retail dataset. The goal is to identify distinct groups of customers based on their purchasing behavior to help businesses tailor their marketing strategies.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Clustering Algorithms](#clustering-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)

## Project Overview
The notebook implements a full data science pipeline for clustering:
1. **Data Loading & Cleaning**: Handling missing values and resetting indices.
2. **Feature Engineering**: Encoding categorical variables.
3. **Scaling**: Standardizing data for model performance.
4. **Dimensionality Reduction**: Using PCA to visualize high-dimensional data.
5. **Clustering**: Applying multiple algorithms to find the best customer segments.

## Dataset
The analysis is performed on the **Online Retail** dataset, which includes transactions for a UK-based non-store online retail.
- **Key Features**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

## Technologies Used
- **Language**: Python
- **Libraries**:
    - **Data Manipulation**: `pandas`, `numpy`
    - **Visualization**: `matplotlib`, `seaborn`
    - **Machine Learning**: `scikit-learn` (Clustering, Preprocessing, Decomposition, Metrics)

## Data Preprocessing
- **Missing Values**: All rows with missing values are removed to ensure data quality.
- **Label Encoding**: Categorical columns like `Country` and `Description` are converted into numerical values using `LabelEncoder`.
- **Scaling**: Features are scaled using `StandardScaler` or `MinMaxScaler` to prevent features with larger ranges from dominating calculations.

## Clustering Algorithms
The project explores and compares several clustering techniques:
- **K-Means**: Partitions data into K distinct clusters.
- **MeanShift**: A centroid-based algorithm that shifts points toward the highest density of data points.
- **DBSCAN**: Groups points based on density and identifies outliers as noise.
- **Gaussian Mixture Model (GMM)**: A probabilistic model that assumes data is generated from a mixture of Gaussian distributions.

## Evaluation Metrics
To determine the quality of the clusters, the following metrics are used:
- **Silhouette Score**: Measures how similar a point is to its own cluster compared to others.
- **Calinski-Harabasz Index**: The ratio of the sum of between-clusters dispersion and within-cluster dispersion.
- **Davies-Bouldin Index**: Measures the average similarity between clusters.

## Visualizations
- **Principal Component Analysis (PCA)**: Used to reduce the dataset's dimensionality for visual inspection.
- **Cluster Plots**: Scatter plots showing individual data points colored by their assigned cluster and their respective centroids (e.g., Invoice Number vs. Unit Price).
