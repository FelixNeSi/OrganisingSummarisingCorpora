from bert import create_transformer_model, encode_corpus_with_model
import umap
import hdbscan


import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def create_umap_embeddings(data, n_neighbors=15, n_components=5):
    print("START UMAP")
    # model = create_transformer_model()
    # embeddings = encode_corpus_with_model(data, model)
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(data, show_progress_bar=True)
    print("creating umap")
    # umap_embeddings = umap.UMAP().fit_transform(embeddings)
    uma = umap.UMAP(n_neighbors=5,
                                n_components=5,
                                metric='cosine').fit(embeddings)
    # print("transforming")
    print("TRANSFORMING")
    umap_embeddings = uma.transform(embeddings)
    print("complete UMAP")
    return umap_embeddings


def hdbscan_cluster(umap_embeddings, min_cluster_size=15, metric='euclidean', cluster_selection_method='eom'):
    print("START HDB")
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              metric=metric,
                              cluster_selection_method=cluster_selection_method).fit(umap_embeddings)
    return cluster


def visualise_clusters(cluster, embeddings):
    # Prepare data
    print("START VIS")
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()

from sklearn.datasets import fetch_20newsgroups
# data = fetch_20newsgroups(subset='all')['data']
# data = data[:100]
# data = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes",
#        "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
#     , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"]
# maps = create_umap_embeddings(data, 2)
# clust = hdbscan_cluster(maps)
# visualise_clusters(clust, maps)