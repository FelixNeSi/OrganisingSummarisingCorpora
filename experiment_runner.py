from clustering_runner import cluster_doc_representation
from all_keyword_extraction import group_clustered_documents, group_keywords_from_file, do_large_doc_approach, \
    do_aggregate_approach, do_baseline_approach
from sklearn import metrics
import pandas as pd
import pickle
from bert import create_transformer_model, encode_corpus_with_model
from topic_modelling import do_get_topic_model
from utils import calculate_mean_average_precision
from preprocessing import clean_corpus

def get_doc_representation(data, method, model_name='distilbert-base-nli-mean-tokens'):
    if method == 'bert':
        model = create_transformer_model(model_name)
        # doc_representations = [encode_corpus_with_model(d, model) for d in data]
        doc_representations = encode_corpus_with_model(data, model)
    elif method == 'topic':
        doc_representations = do_get_topic_model(data)
        # doc_representations = [do_get_topic_model(d) for d in data]
    return doc_representations


def get_save_doc_representation(file_name, representation_method, model_name="distilbert-base-nli-mean-tokens",
                                n_topics=10, save_doc_representation=True, preprocessing='no-pre'):
    with open('{}.pickle'.format(file_name), 'rb') as handle:
        m_data = pickle.load(handle)

    m_text = m_data[1]
    # print(m_text)

    if representation_method == 'bert':
        pickle_name = model_name.replace('/', '-')
    else:
        pickle_name = 'lda_{}_{}'.format(n_topics, preprocessing)

    if save_doc_representation:
        if preprocessing == 'clean':
            m_text = clean_corpus(m_text, [], lemmatize=True)
        elif preprocessing == 'clean-no-lemma':
            m_text = clean_corpus(m_text, [], lemmatize=False)
        docs = get_doc_representation(m_text, method=representation_method, model_name=model_name)
        with open('{}_{}.pickle'.format(file_name, pickle_name), 'wb') as handle:
            pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('{}_{}.pickle'.format(file_name, pickle_name), 'rb') as handle:
            docs = pickle.load(handle)

    return docs


def cluster_experiment(docs, save_name, cluster_method, cluster_nums=None, **kwargs):
    if cluster_nums is None:
        cluster_nums = [2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    # , 450]
    # , 500, 550, 600, 650, 700, 750,
    #             800, 850, 900, 950, 1000, 1100, 1200]
    # , 500, 550, 600, 650, 700]
    all_silhouette, all_davies_bouldin, all_calinksi = [], [], []
    for cl in cluster_nums:
        clusters = cluster_doc_representation(docs, method=cluster_method, num_clusters=cl)
        # for x in clusters:
        #     print(x)
        # print(clusters)
        # silhouette = metrics.silhouette_score(docs, clusters, metric='euclidean')
        # davies = metrics.davies_bouldin_score(docs, clusters)
        # calinksi = metrics.calinski_harabasz_score(docs, clusters)
        try:
            silhouette = metrics.silhouette_score(docs, clusters, metric='euclidean')
            davies = metrics.davies_bouldin_score(docs, clusters)
            calinksi = metrics.calinski_harabasz_score(docs, clusters)
        except ValueError:
            silhouette = 0
            davies = 0
            calinksi = 0
            continue
        print('Cluster number: {}, score: {}'.format(cl, silhouette))
        all_silhouette.append(silhouette)
        all_davies_bouldin.append(davies)
        all_calinksi.append(calinksi)
        # print(metrics.silhouette_score(docs, clusters, metric='euclidean'))
    # organsise_summarise_corpus(dat)
    df = pd.DataFrame(list(zip(cluster_nums, all_silhouette, all_davies_bouldin, all_calinksi)),
                      columns=['n_cluster', 'silhouette score', 'Davies-Bouldin', 'Calinski-Harabasz'])
    df.to_csv('{}.csv'.format(save_name))


def full_clustering_experiment(corpus_data_file_name, preprocessing='no-pre'):
    # bert_models = ['allenai/scibert_scivocab_uncased', 'distilbert-base-nli-mean-tokens', 'all-MiniLM-L6-v2']
    bert_models = []
    clustering_methods = ['kmeans', 'aglo', 'fast']
    for bert_model in bert_models:
        # print(bert_model)
        doc_representation = get_save_doc_representation(corpus_data_file_name, "bert", bert_model,
                                                         save_doc_representation=False)
        for clustering_method in clustering_methods:
            print("Currently at {} WITH {}".format(bert_model, clustering_method))
            save_name = 'experiment/{}/{}-{}-{}-{}'.format(corpus_data_file_name, corpus_data_file_name,
                                                           clustering_method, bert_model.replace('/', '-'),
                                                           preprocessing)
            print(doc_representation)
            print(save_name, clustering_method)
            cluster_experiment(doc_representation, save_name, clustering_method)
    topic_nums = [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 250, 300]
    for topic_num in topic_nums:
        print("Currently at TOPIC NUM  {}".format(topic_num))
        topic_doc_representation = get_save_doc_representation(corpus_data_file_name, "topic", n_topics=topic_num, preprocessing=preprocessing)
        current_method_name = 'lda-topic-nums-{}'.format(topic_num)
        for clustering_method in clustering_methods:
            print("Currently at {} WITH {}".format(topic_num, clustering_method))
            save_name = 'experiment/{}/{}-{}-{}-{}'.format(corpus_data_file_name, corpus_data_file_name,
                                                           clustering_method, current_method_name,
                                                           preprocessing)
            cluster_experiment(topic_doc_representation, save_name, clustering_method)


def keyword_extraction_experiment(corpus_data_file_name, corpus_embeddings, cluster_labels):
    keyword_extraction_methods = ['yake', 'bert', 'sfpd']
    with open('{}.pickle'.format(corpus_data_file_name), 'rb') as handle:
        corpus_data = pickle.load(handle)
    corpus_documents = corpus_data[1]
    corpus_true_keywords = corpus_data[2]
    num_clusters = len(set(cluster_labels))

    grouped_true_keywords = group_keywords_from_file(cluster_labels, corpus_true_keywords, num_clusters)
    large_doc_keywords, aggregate_keywords, baseline_keywords, large_approach, aggregate_approach, baseline_approach, large_doc_scores, aggregate_scores, baseline_scores = [], [], [], [], [], [], [], [], []
    all_keywords, all_approach, all_scores = [], [], []
    for keyword_extraction_method in keyword_extraction_methods:
        temp_keywords_large_doc_approach = do_large_doc_approach(cluster_labels, corpus_documents, num_clusters,
                                                                 method=keyword_extraction_method)
        temp_keywords_aggregate_approach = do_aggregate_approach(cluster_labels, corpus_documents, corpus_embeddings,
                                                                 num_clusters, method=keyword_extraction_method)
        temp_keywords_baseline = do_baseline_approach(corpus_documents, method=keyword_extraction_method, precomputed_embeddings=corpus_embeddings)

        large_doc_keywords.append(temp_keywords_large_doc_approach)
        aggregate_keywords.append(temp_keywords_aggregate_approach)
        baseline_keywords.append(temp_keywords_baseline)

        large_approach.append('large-doc-{}'.format(keyword_extraction_method))
        aggregate_approach.append('aggregate-doc-{}'.format(keyword_extraction_method))
        baseline_approach.append('baseline-{}'.format(keyword_extraction_method))

        large_doc_scores.append(
            calculate_mean_average_precision(temp_keywords_large_doc_approach, grouped_true_keywords))
        aggregate_scores.append(
            calculate_mean_average_precision(temp_keywords_aggregate_approach, grouped_true_keywords))
        baseline_scores.append(
            calculate_mean_average_precision(temp_keywords_baseline, corpus_true_keywords)
        )

        all_keywords.append(temp_keywords_large_doc_approach)
        all_keywords.append(temp_keywords_aggregate_approach)
        all_keywords.append(temp_keywords_baseline)

        all_approach.append('large-doc-{}'.format(keyword_extraction_method))
        all_approach.append('aggregate-doc-{}'.format(keyword_extraction_method))
        all_approach.append('baseline-{}'.format(keyword_extraction_method))

        all_scores.append(
            calculate_mean_average_precision(temp_keywords_large_doc_approach, grouped_true_keywords))
        all_scores.append(
            calculate_mean_average_precision(temp_keywords_aggregate_approach, grouped_true_keywords))
        all_scores.append(
            calculate_mean_average_precision(temp_keywords_baseline, corpus_true_keywords)
        )

    # TODO SAVE THE DATA INTO PICKLE AND CSV, FIGURE OUT THE BETTER DATA FORMAT

    df_large = pd.DataFrame(list(zip(large_approach, large_doc_scores, large_doc_keywords)), columns=['approach', 'scores', 'keywords'])
    df_large.to_csv("kw_extraction_large.csv")
    df_aggregate = pd.DataFrame(list(zip(aggregate_approach, aggregate_scores, aggregate_keywords)), columns=['approach', 'scores', 'keywords'])
    df_aggregate.to_csv("kw_extraction_aggregate.csv")
    df_baseline = pd.DataFrame(list(zip(baseline_approach, baseline_scores, baseline_keywords)),
                                columns=['approach', 'scores', 'keywords'])
    df_baseline.to_csv("kw_extraction_baseline.csv")
    df_all = pd.DataFrame(list(zip(all_approach, all_scores, all_keywords)) , columns=['approach', 'scores', 'keywords'])
    df_all.to_csv("kw_extraction_all.csv")

    # grouped_true_kw = group_keywords_from_file(clusters, true_kw, 15)
    # large_kw = do_large_doc_approach(clusters, docs, 15)
    # aggregate_kw = do_aggregate_approach(clusters, docs, Kravpivin_miniLM, 15)

corpora = ['marujo', 'Kravpivin2009', '500n-KPCrowd',  'kdd-science']
# corpora = ['Kravpivin2009', 'kdd-science']
for corpus in corpora:
    full_clustering_experiment(corpus, preprocessing='clean-no-lemma')


# with open("Kravpivin2009_all-MiniLM-L6-v2.pickle", 'rb') as handle:
#     Kravpivin_miniLM = pickle.load(handle)
# # TODO FIGURE OUT HOW TO REUSE THE PREGENERATED EMBEDDINGS
# with open('{}.pickle'.format('Kravpivin2009'), 'rb') as handle:
#     ccorpus_data = pickle.load(handle)
# ccorpus_documents = ccorpus_data[1]
#
# clusterss = cluster_doc_representation(Kravpivin_miniLM, method='aglo', num_clusters=20)
# keyword_extraction_experiment('Kravpivin2009', Kravpivin_miniLM, clusterss)

# /Users/felixnesi/PycharmProjects/OrganisingSummarisingCorpus/scibert_scivocab_uncased
# /Users/felixnesi/PycharmProjects/OrganisingSummarisingCorpus/distilbert-base-nli-mean-tokens
# /Users/felixnesi/PycharmProjects/OrganisingSummarisingCorpus/all-MiniLM-L6-v2
