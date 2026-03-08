# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect the dataset Obtain the customer dataset containing independent variables (features such as age, annual income, spending score, purchase frequency, etc.). This is an unsupervised learning task — no labeled output variable is required.
2. Identify variables Let X be the set of independent variables (customer attributes). Let K be the number of clusters to form. Define the feature space in which similarity between customers will be measured.
3. Preprocess the data Handle missing values if any. Encode categorical variables into numerical form. Normalize or standardize features (e.g., using Min-Max or Z-score normalization) to ensure equal contribution from each feature.
4. Choose the number of clusters K Select K using the Elbow Method: compute the Within-Cluster Sum of Squares (WCSS) for different values of K and pick the point where the rate of decrease sharply levels off. WCSS formula:
WCSS = ΣK  Σ  ||xᵢ − μₖ||2
5. Initialize cluster centroids Randomly select K data points from the dataset as initial centroids, or use K-Means++ initialization for improved results. K-Means++ selects each subsequent centroid proportional to its squared distance from the nearest existing centroid.
6. Assign each data point to the nearest centroid For each data point xᵢ, calculate the Euclidean distance to all K centroids. Assign xᵢ to the cluster Cₖ whose centroid μₖ is closest. Distance formula:
d(xᵢ, μₖ) = √Σ(xᵢj − μₖj)2
7. Update cluster centroids Recompute the centroid of each cluster by taking the mean of all data points currently assigned to that cluster. The new centroid μₖ becomes the arithmetic mean of all points in cluster Cₖ.
8. Apply stopping conditions Stop iterating when: Centroids no longer change (convergence is reached), or Data points no longer switch clusters between iterations, or The maximum number of iterations is reached.
9. Assign cluster labels Label each cluster with a meaningful customer segment name based on the centroid characteristics (e.g., High-Value Customers, Budget Shoppers, Occasional Buyers).
10. Train the model Construct the final K-Means model by running the algorithm on the full training dataset with the optimal K and best initialization.
11. Test the model Use the testing dataset (or new customer data) to predict cluster membership by assigning each new point to the nearest centroid.
12. Evaluate model performance Assess cluster quality using: Silhouette Score (measures cohesion vs. separation), Davies-Bouldin Index (lower is better), Inertia/WCSS (lower means tighter clusters), and visual inspection via scatter plots.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SUDARSAN.A
RegisterNumber:  212224220111
*/
```
```python
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Deepak S
RegisterNumber: 212224230053
*/

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++",n_init=10)
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters=5, n_init=10)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster4")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster5")
plt.legend()
plt.title("Customer Segments")
```

## Output:
### DATA.HEAD()
<img width="830" height="292" alt="Screenshot 2026-03-08 173723" src="https://github.com/user-attachments/assets/ad5585b3-292c-4d22-ad7f-f315dd567831" />

### DATA.INFO()
<img width="633" height="319" alt="Screenshot 2026-03-08 173740" src="https://github.com/user-attachments/assets/d4d7a4c1-e45e-4193-8905-2b56444c9aac" />

### DATA.ISNULL().SUM()
<img width="346" height="165" alt="Screenshot 2026-03-08 173805" src="https://github.com/user-attachments/assets/08d825a3-8e12-4ddc-a01c-588868722d2b" />

### ELBOW GRAPH:
<img width="1015" height="714" alt="Screenshot 2026-03-08 173940" src="https://github.com/user-attachments/assets/9189457b-5123-4d70-9592-9e31cb02426c" />

### Y_PREDICTION:
<img width="837" height="271" alt="Screenshot 2026-03-08 174009" src="https://github.com/user-attachments/assets/4accd833-38f1-48f1-92ba-a3267841d414" />

### CLUSTERS:
<img width="983" height="705" alt="Screenshot 2026-03-08 174027" src="https://github.com/user-attachments/assets/504f6514-0176-4edb-a81a-369154bf4048" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
