#  Clustering

This folder contains implementations of major unsupervised clustering algorithms used for discovering hidden patterns and structure in data.

##  K-Means Clustering
- Applied centroid-based clustering.
- Determined optimal number of clusters using evaluation methods.
- Assigned cluster labels to each data point.
- Used cluster labels as new features for further analysis.
- Compared clustering results with different values of K.

##  Hierarchical Clustering
- Implemented agglomerative clustering approach.
- Built dendrograms to visualize cluster merging.
- Analyzed cluster structure at different levels.
- Selected appropriate number of clusters based on linkage patterns.
- Compared results with K-Means.

##  DBSCAN (Density-Based Clustering)
- Applied density-based clustering algorithm.
- Identified core points, border points, and noise.
- Handled arbitrary-shaped clusters.
- Detected outliers automatically.
- Compared performance with centroid-based methods.

---

##  Workflow Followed
- Data preprocessing and scaling
- Applying clustering algorithm
- Evaluating cluster quality
- Visualizing clusters (where applicable)
- Analyzing differences between clustering methods

---

##  Objective
To understand different clustering approaches (centroid-based, hierarchical, and density-based) and analyze how each method discovers structure in data.