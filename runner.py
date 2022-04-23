import pandas as pd

from bert import create_transformer_model, encode_corpus_with_model, kmeans_cluster, fast_cluster, aglo_cluster, \
    do_bert_keyword_extraction
from topic_modelling import do_get_topic_model
from keyword_extraction_yake import create_yake, get_yake_keywords, do_get_expand_sfpd_phrases
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import numpy as np
import pickle
from sklearn.decomposition import PCA


def get_doc_representation(data, method, model_name='distilbert-base-nli-mean-tokens'):
    if method == 'bert':
        model = create_transformer_model(model_name)
        # doc_representations = [encode_corpus_with_model(d, model) for d in data]
        doc_representations = encode_corpus_with_model(data, model)
    elif method == 'topic':
        doc_representations = do_get_topic_model(data)
        # doc_representations = [do_get_topic_model(d) for d in data]
    return doc_representations


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


def extract_keywords_from_cluster(clustered_docs, method):
    background = ["Dont use this document!", "No testing here"]
    if method == 'yake':
        model = create_yake()
    keywords_from_cluster = {}
    for k, v in clustered_docs.items():
        if method == 'bert':
            keywords_from_cluster[k] = do_bert_keyword_extraction(v)
        elif method == 'yake':
            keywords_from_cluster[k] = [get_yake_keywords(model, doc) for doc in v]
        elif method == 'sfpd':
            keywords_from_cluster[k] = do_get_expand_sfpd_phrases(v, background)
    return keywords_from_cluster


def organsise_summarise_corpus(data):
    docs = get_doc_representation(data, "bert")
    print(docs)
    clust = cluster_doc_representation(docs, 'fast', num_clusters=3, min_community_size=1)
    print(clust)
    # for i, cluster in enumerate(clust):
    #     print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    #     for sentence_id in cluster[0:3]:
    #         print("\t", data[sentence_id])
    #     print("\t", "...")
    #     for sentence_id in cluster[-3:]:
    #         print("\t", data[sentence_id])
    groups = group_clustered_documents(clust, data)
    print(groups)
    kwords = extract_keywords_from_cluster(groups, 'yake')
    print(kwords)
    return kwords


# dat = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes",
#        "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
#     , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"
def get_save_doc_representation(file_name, representation_method, model_name="distilbert-base-nli-mean-tokens", n_topics=10, save_doc_representation=True):
    with open('{}.pickle'.format(file_name), 'rb') as handle:
        m_data = pickle.load(handle)

    m_text = m_data[1]
    # print(m_text)

    if representation_method == 'bert':
        pickle_name = model_name.replace('/', '-')
    else:
        pickle_name = 'lda_{}'.format(n_topics)

    if save_doc_representation:
        docs = get_doc_representation(m_text, method=representation_method, model_name=model_name)
        with open('{}_{}.pickle'.format(file_name, pickle_name), 'wb') as handle:
            pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('{}_{}.pickle'.format(file_name, pickle_name), 'rb') as handle:
            docs = pickle.load(handle)

    return docs

# def cluster_experiment(file_name, save_name, representation_method='bert', model_name='distilbert-base-nli-mean-tokens', save_doc_representation=True, n_topics=10):
def cluster_experiment(docs, save_name, cluster_method):
    # with open('{}.pickle'.format(file_name), 'rb') as handle:
    #     m_data = pickle.load(handle)
    #
    # m_text = m_data[1]
    # # print(m_text)
    #
    # if representation_method == 'bert':
    #     pickle_name = model_name
    # else:
    #     pickle_name = 'lda_{}'.format(n_topics)
    #
    # if save_doc_representation:
    #     docs = get_doc_representation(m_text, method=representation_method)
    #     with open('{}_{}.pickle'.format(file_name, pickle_name), 'wb') as handle:
    #         pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     with open('{}_{}.pickle'.format(file_name, pickle_name), 'rb') as handle:
    #         docs = pickle.load(handle)

    # pca = PCA(n_components=50)
    # new_docs = pca.fit_transform(docs)

    #cluster_nums = [5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    cluster_nums = [2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    # , 450]
        # , 500, 550, 600, 650, 700, 750,
        #             800, 850, 900, 950, 1000, 1100, 1200]
        #, 500, 550, 600, 650, 700]
    all_silhouette = []
    for cl in cluster_nums:
        clusters = cluster_doc_representation(docs, method=cluster_method, num_clusters=cl)
        silhouette = metrics.silhouette_score(docs, clusters, metric='euclidean')
        print('Cluster number: {}, score: {}'.format(cl, silhouette))
        all_silhouette.append(silhouette)
        # print(metrics.silhouette_score(docs, clusters, metric='euclidean'))
    # organsise_summarise_corpus(dat)
    df = pd.DataFrame(list(zip(cluster_nums, all_silhouette)), columns=['n_cluster', 'silhouette score'])
    df.to_csv('{}.csv'.format(save_name))

#cluster_experiment("500n-KPCrowd", "500n-KPCrowd-aglo-cluster-all-MiniLM-L6-v2", model_name='all-MiniLM-L6-v2', save_doc_representation=False)
# cluster_experiment("500n-KPCrowd", "500n-KPCrowd-kmeans-cluster-100-lda", representation_method='topic', n_topics=100)
#cluster_experiment("Kravpivin2009", "Kravpivin2009-kmeans-cluster-all-MiniLM", 'all-MiniLM-L6-v2', False)

# KPCrowd_distil = get_save_doc_representation("500n-KPCrowd", "bert", "distilbert-base-nli-mean-tokens")
# cluster_experiment(KPCrowd_distil, "500n-KPCrowd-kmeans-distil-no-pre", 'kmeans')
# cluster_experiment(KPCrowd_distil, "500n-KPCrowd-aglo-distil-no-pre", 'aglo')
# KPCrowd_MiniLM = get_save_doc_representation("500n-KPCrowd", "bert", "all-MiniLM-L6-v2")
# cluster_experiment(KPCrowd_MiniLM, "500n-KPCrowd-kmeans-MiniLM", 'kmeans')
# cluster_experiment(KPCrowd_MiniLM, "500n-KPCrowd-aglo-MiniLM", 'aglo')

# marujo_distil = get_save_doc_representation("marujo", "bert", "distilbert-base-nli-mean-tokens")
# print('{} ----- {}'.format("1", len(marujo_distil)))
# # with open("Kravpivin2009_distilbert-base-nli-mean-tokens.pickle", 'rb') as handle:
# #     Kravpivin_distil = pickle.load(handle)
# cluster_experiment(marujo_distil, "marujo-kmeans-distil-no-pre", 'kmeans')
# cluster_experiment(marujo_distil, "marujo-aglo-distil-no-pre", 'aglo')
# marujo_MiniLM = get_save_doc_representation("marujo", "bert", "all-MiniLM-L6-v2")
# print('{} ----- {}'.format("1", len(marujo_MiniLM)))
# # with open("Kravpivin2009_all-MiniLM-L6-v2.pickle", 'rb') as handle:
# #     Kravpivin_MiniLM = pickle.load(handle)
# cluster_experiment(marujo_MiniLM, "marujo-kmeans-MiniLM", 'kmeans')
# cluster_experiment(marujo_MiniLM, "marujo-aglo-MiniLM", 'aglo')
#
# marujo_SciBert = get_save_doc_representation("marujo", "bert", "allenai/scibert_scivocab_uncased")
# print('{} ----- {}'.format("1", len(marujo_SciBert)))
# cluster_experiment(marujo_SciBert, "marujo-kmeans-scibert-no-pre", 'kmeans')
# cluster_experiment(marujo_SciBert, "marujo-aglo-scibert-no-pre", 'aglo')
#
# topic_nums = [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 250, 300]
# for t in topic_nums:
#     marujo_topic = get_save_doc_representation("marujo", "topic", n_topics=t)
#     cluster_experiment(marujo_topic, "marujo-kmeans-no-pre-{}-lda".format(t), 'kmeans')
#     cluster_experiment(marujo_topic, "marujo-aglo-no-pre-{}-lda".format(t), 'aglo')



# KPCrowd_sci = get_save_doc_representation("500n-KPCrowd", "bert", "scibert_scivocab_cased")
# cluster_experiment(KPCrowd_sci, "500n-KPCrowd-kmeans-scibert-no-pre", 'kmeans')
# cluster_experiment(KPCrowd_sci, "500n-KPCrowd-aglo-scibert-no-pre", 'aglo')
#
# Kravpivin_distil = get_save_doc_representation("Kravpivin2009", "bert", "distilbert-base-nli-mean-tokens")
# # with open("Kravpivin2009_distilbert-base-nli-mean-tokens.pickle", 'rb') as handle:
# #     Kravpivin_distil = pickle.load(handle)
# cluster_experiment(Kravpivin_distil, "Kravpivin-kmeans-distil-no-pre", 'kmeans')
# cluster_experiment(Kravpivin_distil, "Kravpivin-aglo-distil-no-pre", 'aglo')
# Kravpivin_MiniLM = get_save_doc_representation("Kravpivin2009", "bert", "all-MiniLM-L6-v2")
# # with open("Kravpivin2009_all-MiniLM-L6-v2.pickle", 'rb') as handle:
# #     Kravpivin_MiniLM = pickle.load(handle)
# cluster_experiment(Kravpivin_MiniLM, "Kravpivin-kmeans-MiniLM", 'kmeans')
# cluster_experiment(Kravpivin_MiniLM, "Kravpivin-aglo-MiniLM", 'aglo')
#
# Kravpivin_SciBert = get_save_doc_representation("Kravpivin2009", "bert", "allenai/scibert_scivocab_uncased")
# cluster_experiment(Kravpivin_SciBert, "Kravpivin-kmeans-scibert-no-pre", 'kmeans')
# cluster_experiment(Kravpivin_SciBert, "Kravpivin-aglo-scibert-no-pre", 'aglo')
#
# topic_nums = [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 250, 300]
# for t in topic_nums:
#     Kravpivin_topic = get_save_doc_representation("Kravpivin2009", "topic", n_topics=t)
#     cluster_experiment(Kravpivin_topic, "Kravpivin-kmeans-no-pre-{}-lda".format(t), 'kmeans')
#     cluster_experiment(Kravpivin_topic, "Kravpivin-aglo-no-pre-{}-lda".format(t), 'aglo')