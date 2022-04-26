import pandas as pd

from bert import create_transformer_model, encode_corpus_with_model, kmeans_cluster, fast_cluster, aglo_cluster, \
    do_bert_keyword_extraction
from topic_modelling import do_get_topic_model
from keyword_extraction_yake import create_yake, get_yake_keywords, do_get_expand_sfpd_phrases
from sklearn import metrics
import numpy as np
import pickle
from sklearn.decomposition import PCA


def cluster_doc_representation(doc_representations, method, num_clusters=10, min_community_size=5, threshold=0.75,
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