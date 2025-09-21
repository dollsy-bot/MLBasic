# ===============================
# K-Means Clustering for Mall Customers
# Ready-to-run script for VS Code
# ===============================

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Fix MKL KMeans warning on Windows

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load dataset if exists, else create sample
# -------------------------------
try:
    df = pd.read_csv("Mall_Customers.csv")
    print("Dataset loaded from file.")
except FileNotFoundError:
    print("Mall_Customers.csv not found. Using sample dataset.")
    data = {
        "CustomerID": range(1, 21),
        "Gender": ["Male", "Female", "Female", "Male", "Female",
                   "Male", "Female", "Male", "Female", "Male",
                   "Female", "Male", "Female", "Male", "Female",
                   "Male", "Female", "Male", "Female", "Male"],
        "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30,
                40, 42, 36, 65, 38, 48, 50, 37, 67, 18],
        "Annual Income (k$)": [15, 16, 17, 18, 19, 20, 21, 22, 40, 42,
                               60, 62, 65, 66, 68, 70, 75, 78, 85, 90],
        "Spending Score (1-100)": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                                   47, 52, 60, 15, 32, 60, 50, 85, 20, 90]
    }
    df = pd.DataFrame(data)

# -------------------------------
# Select features and scale
# -------------------------------
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Use Elbow Method to find optimal k
# -------------------------------
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# -------------------------------
# Train KMeans with chosen k (5 clusters)
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Print first few rows with cluster labels
print(df.head())

# -------------------------------
# Plot clusters
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="rainbow", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color="black", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# -------------------------------
# Save clustered dataset
# -------------------------------
df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("Clustered dataset saved as Mall_Customers_Clustered.csv")