
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Dict

def load_vectors(file_path: str) -> List[Dict]:
    """Loads vectors from a jsonl file."""
    vectors = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "vector" in data:
                vectors.append(data)
    return vectors

def main():
    """Main function to run the analysis."""
    # Load the data
    data_path = "outputs/diffusion_train.jsonl"
    print(f"Loading data from {data_path}...")
    papers = load_vectors(data_path)
    vectors = np.array([p["vector"] for p in papers])
    print(f"Loaded {len(vectors)} vectors.")

    if len(vectors) == 0:
        print("No vectors found. Exiting.")
        return

    # Dimensionality Reduction with PCA
    print("Performing PCA to reduce dimensions to 2...")
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Clustering with K-Means
    # We'll start with a reasonable guess for the number of clusters, e.g., 8
    n_clusters = 8
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors_2d)

    # Visualization
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    plt.title("Paper Embeddings Clustering (PCA + K-Means)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(n_clusters)])
    
    # Save the plot
    output_path = "outputs/clusters.png"
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
