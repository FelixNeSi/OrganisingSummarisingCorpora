from preprocessing import *
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np


def create_transformer_model(model_name='distilbert-base-nli-mean-tokens'):
    model = SentenceTransformer(model_name)
    return model


def encode_corpus_with_model(data, model):
    embeddings = model.encode(data, show_progress_bar=True)
    return embeddings


def kmeans_cluster(corpus_embeddings, num_clusters):
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def aglo_cluster(corpus_embeddings, n_clusters=None, distance_threshold=1.5):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def print_clusters(cluster_assignment, corpus):
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i + 1)
        print(cluster)
        print("")


def normalise_embeddings(embeddings):
    embeddings = embeddings /  np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


data = ["This is a 1 TEST sentence that contains", "ALL OF THE STOPWORDS !@!@!@@", "Good TESTS go For EdGe CaSes"]
model = create_transformer_model('distilbert-base-nli-mean-tokens')
embeddings = encode_corpus_with_model(data, model)
embeddings = normalise_embeddings(embeddings)
assignments = kmeans_cluster(embeddings, 2)
print_clusters(assignments, data)

# print(embeddings)