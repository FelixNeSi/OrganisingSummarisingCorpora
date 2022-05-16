import pandas as pd
from sentence_transformers import util
from sklearn.cluster import KMeans, AgglomerativeClustering

from bert import create_transformer_model, encode_corpus_with_model, do_bert_keyword_extraction
from topic_modelling import do_get_topic_model
from keyword_extraction_yake import create_yake, get_yake_keywords, do_get_expand_sfpd_phrases
from sklearn import metrics
import numpy as np
import pickle
from sklearn.decomposition import PCA

def cluster_doc_representation(doc_representations, method, num_clusters=10, min_community_size=1, threshold=0.5,
                               dist_threshold=1.5):
    if method == 'kmeans':
        clusters = kmeans_cluster(doc_representations, num_clusters)
    elif method == 'fast':
        clusters = fast_cluster(doc_representations, min_community_size, threshold)
    elif method == 'aglo':
        clusters = aglo_cluster(doc_representations, num_clusters)
    return clusters


def group_clustered_documents(cluster_labels, data):
    clustered_docs = {}
    for i, label in enumerate(cluster_labels):
        docs_in_cluster = clustered_docs.get(label, [])
        docs_in_cluster.append(data[i])
        clustered_docs[label] = docs_in_cluster
    return clustered_docs


def kmeans_cluster(corpus_embeddings, num_clusters=5):
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def aglo_cluster(corpus_embeddings, n_clusters=5, distance_threshold=None):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters,
                                               distance_threshold=distance_threshold)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def fast_cluster(corpus_embeddings, min_community_size=5, threshold=0.25):
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold,
                                        init_max_size=len(corpus_embeddings))
    assignments = [None for x in range(len(corpus_embeddings))]
    for i, cluster in enumerate(clusters):
        for doc in cluster:
            assignments[doc] = i
    # print(assignments)
    return assignments