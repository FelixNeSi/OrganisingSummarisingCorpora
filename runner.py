from bert import create_transformer_model, encode_corpus_with_model, kmeans_cluster, fast_cluster, aglo_cluster, \
    do_bert_keyword_extraction
from topic_modelling import do_get_topic_model
from keyword_extraction_yake import create_yake, get_yake_keywords, do_get_expand_sfpd_phrases


def get_doc_representation(data, method):
    if method == 'bert':
        model = create_transformer_model()
        # doc_representations = [encode_corpus_with_model(d, model) for d in data]
        doc_representations = encode_corpus_with_model(data, model)
    elif method == 'topic':
        doc_representations = [do_get_topic_model(d) for d in data]
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


dat = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes",
       "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
    , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"]

organsise_summarise_corpus(dat)
