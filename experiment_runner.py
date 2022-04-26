from clustering_runner import cluster_doc_representation
from sklearn import metrics
import pandas as pd
import pickle
from bert import create_transformer_model, encode_corpus_with_model
from topic_modelling import do_get_topic_model


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
                                n_topics=10, save_doc_representation=True):
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


def cluster_experiment(docs, save_name, cluster_method, cluster_nums=None):
    if cluster_nums is None:
        cluster_nums = [2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    # , 450]
    # , 500, 550, 600, 650, 700, 750,
    #             800, 850, 900, 950, 1000, 1100, 1200]
    # , 500, 550, 600, 650, 700]
    all_silhouette, all_davies_bouldin, all_calinksi = [], [], []
    for cl in cluster_nums:
        clusters = cluster_doc_representation(docs, method=cluster_method, num_clusters=cl)
        silhouette = metrics.silhouette_score(docs, clusters, metric='euclidean')
        davies = metrics.davies_bouldin_score(docs, clusters)
        calinksi = metrics.calinski_harabasz_score(docs, clusters)
        print('Cluster number: {}, score: {}'.format(cl, silhouette))
        all_silhouette.append(silhouette)
        all_davies_bouldin.append(davies)
        all_calinksi.append(calinksi)
        # print(metrics.silhouette_score(docs, clusters, metric='euclidean'))
    # organsise_summarise_corpus(dat)
    df = pd.DataFrame(list(zip(cluster_nums, all_silhouette, all_davies_bouldin, all_calinksi)),
                      columns=['n_cluster', 'silhouette score', 'Davies-Bouldin', 'Calinski-Harabasz'])
    df.to_csv('{}.csv'.format(save_name))
