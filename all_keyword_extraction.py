import pandas as pd
import torch

from keyword_extraction_yake import create_yake, get_yake_keywords, do_get_expand_sfpd_phrases
from bert import create_transformer_model, get_most_cosine_similar, \
    do_bert_keyword_extraction, max_sum_sim, mmr
import pickle
from clustering_runner import group_clustered_documents
from utils import calculate_mean_average_precision


def large_doc_extract(grouped_documents, num_clusters, method='yake', bert_model_name='all-MiniLM-L6-v2', precomputed_embeddings=None):
    available_methods = ['yake', 'bert', 'sfpd']
    if method not in available_methods:
        print("Please choose an availble method from: {}".format(available_methods))
        return None
    if method == 'yake':
        yake_model = create_yake(num_of_keywords=50)
    if method == 'bert':
        model = create_transformer_model(bert_model_name)
    # grouped_large_kw = ["" for i in range(num_clusters)]
    combined_docs = []
    for i in range(num_clusters):
        print('STARTING ON CLUSTER {}'.format(str(i)))
        # temp_combined_doc = ''
        # for doc in grouped_documents[i]:
        #     temp_combined_doc = temp_combined_doc + ' ' + doc
        temp_combined_doc = ' '.join(grouped_documents[i])
        combined_docs.append(temp_combined_doc)

    keywords = []
    for i, combined_doc in enumerate(combined_docs):
        print('STARTING KEYWORD ON CLUSTER {}'.format(str(i)))
        if method == 'yake':
            # CODE FOR YAKE KEYWORDS, MUST!!! STRIP THE KEYWORDS AFTER
            keywords.append(get_yake_keywords(yake_model, combined_doc))
        elif method == 'bert':
            # BERT APPROACH SHOULD WORK HERE, LOOK AT HYPERPARAMETERS FOR NGRAM RANGE
            keywords.append(do_bert_keyword_extraction([combined_doc], model, precomputed_embeddings=precomputed_embeddings)[0])
        elif method == 'sfpd':
            # SFPD SHOULD WORK HERE, LOOK INTO PRE PROCESSING (\n included in potential phrase)
            background_docs = [x for j, x in enumerate(combined_docs) if j != i]
            keywords.append(do_get_expand_sfpd_phrases([combined_doc], background_docs))

    # IF YAKE, MAKE SURE TO STRIP
    if method == 'yake':
        stripped_keywords = strip_scores_from_grouped_keywords(keywords)
        return stripped_keywords
    else:
        return keywords


def aggregate_compare_with_average_of_cluster(grouped_documents, average_doc_embeddings, num_clusters, method='yake',
                                              bert_comparison_model_name='all-MiniLM-L6-v2',
                                              aggregate_similairty_measure='cosine', precomputed_embeddings=None):
    available_methods = ['yake', 'bert', 'sfpd']
    if method not in available_methods:
        print("Please choose an availble method from: {}".format(available_methods))
        return None
    if method == 'yake':
        yake_model = create_yake(num_of_keywords=100)
    model = create_transformer_model(bert_comparison_model_name)


    grouped_candidate_keywords = []
    for i in range(num_clusters):
        temp_candidate_keywords = []
        for l, doc in enumerate(grouped_documents[i]):
            # temp_keywords = get_yake_keywords(yak, doc)
            # temp_candidate_keywords = temp_candidate_keywords + temp_keywords

            if method == 'yake':
                temp_candidate_keywords = temp_candidate_keywords + get_yake_keywords(yake_model, doc)
            elif method == 'bert':
                if precomputed_embeddings == None:
                    precomputed_embedding = None
                else:
                    precomputed_embedding = precomputed_embeddings[i][l]
                temp_candidate_keywords = temp_candidate_keywords + do_bert_keyword_extraction([doc], model, precomputed_embeddings=precomputed_embedding)[0]
            elif method == 'sfpd':
                background_docs = []
                for j in range(num_clusters):
                    if j != i:
                        background_docs = background_docs + grouped_documents[i]
                temp_candidate_keywords = temp_candidate_keywords + do_get_expand_sfpd_phrases([doc], background_docs)
        grouped_candidate_keywords.append(list(set(temp_candidate_keywords)))

    if method == 'yake':
        # NEEDED FOR YAKE
        grouped_stripped_candidates = strip_scores_from_grouped_keywords(grouped_candidate_keywords)
    else:
        # BERT OR SFPD USE THIS!!!!!
        grouped_stripped_candidates = grouped_candidate_keywords

    no_dupe_grouped_stripped = []
    for grs in grouped_stripped_candidates:
        no_dupe_grouped_stripped.append(list(set(grs)))

    # model = create_transformer_model(bert_comparison_model)
    grouped_candidate_embeddings = [[model.encode(cand) for cand in candidates] for candidates in
                                    no_dupe_grouped_stripped]

    aggregate_average_keywords = []
    for i in range(num_clusters):
        if aggregate_similairty_measure == 'cosine':
            aggregate_average_keywords.append(
                get_most_cosine_similar(average_doc_embeddings[i].reshape(1, -1), grouped_candidate_embeddings[i],
                                        no_dupe_grouped_stripped[i], top_n=25))
        elif aggregate_similairty_measure == 'max_sum':
            # Keep nr_candidates < 20% of total words of unique words in doc
            aggregate_average_keywords.append(
                max_sum_sim(average_doc_embeddings[i].reshape(1, -1), grouped_candidate_embeddings[i],
                            no_dupe_grouped_stripped[i], top_n=25, nr_candidates=50))
        elif aggregate_similairty_measure == 'mmr':
            aggregate_average_keywords.append(
                # Diversity ranges from 0 to 1, higher will equal more diverse phrases,
                mmr(average_doc_embeddings[i].reshape(1, -1), grouped_candidate_embeddings[i],
                    no_dupe_grouped_stripped[i], top_n=25, diversity=0.2))
    return aggregate_average_keywords


def group_and_average_doc_embeddings(cluster_labels, doc_embeddings, num_clusters):
    clustered_doc_embeddings = {}
    for i, label in enumerate(cluster_labels):
        doc_embeddings_in_cluster = clustered_doc_embeddings.get(label, [])
        doc_embeddings_in_cluster.append(doc_embeddings[i])
        clustered_doc_embeddings[label] = doc_embeddings_in_cluster

    average_clusters_embedding = []
    for i in range(num_clusters):
        temp_stack_embedding = torch.stack(clustered_doc_embeddings[i])
        average_clusters_embedding.append(torch.squeeze(torch.mean(temp_stack_embedding, 0, keepdim=True)))
    return average_clusters_embedding


def group_keywords_from_file(cluster_labels, keywords, num_clusters):
    clustered_keywords = {}
    for i, label in enumerate(cluster_labels):
        keywords_in_cluster = clustered_keywords.get(label, [])
        keywords_in_cluster.append(keywords[i])
        clustered_keywords[label] = keywords_in_cluster

    combined_keywords = []
    for i in range(num_clusters):
        temp_combined_keywords = []
        for kws in clustered_keywords[i]:
            temp_combined_keywords = temp_combined_keywords + kws
        combined_keywords.append(list(set(temp_combined_keywords)))
    return combined_keywords


def strip_scores_from_grouped_keywords(grouped_keywords):
    stripped_keywords = []
    for group in grouped_keywords:
        temp_stripped = []
        for non_stripped_keywords in group:
            temp_stripped.append(non_stripped_keywords[0])
        stripped_keywords.append(temp_stripped)
    return stripped_keywords


def strip_yake_scores(keywords):
    stripped_keywords = []
    for doc_keywords in keywords:
        doc_stripped = []
        for keyword in doc_keywords:
            doc_stripped.append(keyword[0])
        stripped_keywords.append(doc_stripped)
    return stripped_keywords


def do_aggregate_approach(cluster_assignments, documents, document_embeddings, num_clusters, method='yake'):
    grouped_documents = group_clustered_documents(cluster_assignments, documents)
    grouped_document_embeddings = group_clustered_documents(cluster_assignments, document_embeddings)
    average_grouped_documents = group_and_average_doc_embeddings(cluster_assignments, document_embeddings, num_clusters)
    aggregate_average_kw = aggregate_compare_with_average_of_cluster(grouped_documents, average_grouped_documents, num_clusters, method=method, precomputed_embeddings=grouped_document_embeddings)
    return aggregate_average_kw


def do_large_doc_approach(cluster_assignments, documents, num_clusters, method='yake'):
    grouped_documents = group_clustered_documents(cluster_assignments, documents)
    large_doc_keywords = large_doc_extract(grouped_documents, num_clusters, method=method)
    return large_doc_keywords


def do_baseline_approach(documents, method='yake', background_documents=[], precomputed_embeddings=None, bert_model_name="all-MiniLM-L6-v2"):
    if method == 'yake':
        yake_model = create_yake(max_ngram_size=6)
        non_stripped_keywords = [get_yake_keywords(yake_model, doc) for doc in documents]
        keywords = strip_yake_scores(non_stripped_keywords)
    elif method == 'bert':
        model = create_transformer_model(bert_model_name)
        keywords = do_bert_keyword_extraction(documents, model, precomputed_embeddings=precomputed_embeddings)
    elif method == 'sfpd':
        keywords = [do_get_expand_sfpd_phrases([doc], background_documents) for doc in documents]
    return keywords


def tensor_to_list(tensors):
    tensor_list = []
    for tensor in tensors:
        tensor_list.append(tensor)
    return tensor_list

# with open("Kravpivin2009_all-MiniLM-L6-v2.pickle", 'rb') as handle:
#     Kravpivin_miniLM = pickle.load(handle)
#
#
# with open('{}.pickle'.format("marujo"), 'rb') as handle:
#     background_docss = pickle.load(handle)[1]
#
# with open('{}.pickle'.format("Kravpivin2009"), 'rb') as handle:
#     corpus_data = pickle.load(handle)
# corpus_documents = corpus_data[1]
# corpus_true_keywords = corpus_data[2]
#
# pre_comp_embed = tensor_to_list(Kravpivin_miniLM)
#
# all_kws, all_scores, all_approach = [], [], []
# keyword_extraction_methods = ['bert']
# for kw_method in keyword_extraction_methods:
#     temp_kw = do_baseline_approach(corpus_documents, method=kw_method, background_documents=background_docss, precomputed_embeddings=pre_comp_embed)
#     all_kws.append(temp_kw)
#     all_scores.append(
#         calculate_mean_average_precision(temp_kw, corpus_true_keywords)
#     )
#     all_approach.append("baseline-{}".format(kw_method))
#
# df_baseline = pd.DataFrame(list(zip(all_approach, all_scores, all_kws)), columns=['approach', 'scores', 'keywords'])
# df_baseline.to_csv("kw_extraction_baseline_test4.csv")



# df = pd.read_csv("Kravpivin2009_with_keywords.csv")
# docs = df['1'].tolist()
# docs = docs[:50]
#
# df = pd.read_csv("Marujo_with_keywords.csv")
# background_docs = df['1'].tolist()
# background_docs = background_docs[:50]
#
# with open("Kravpivin2009_all-MiniLM-L6-v2.pickle", 'rb') as handle:
#     Kravpivin_miniLM = pickle.load(handle)
# Kravpivin_miniLM = Kravpivin_miniLM[:50]
#
# with open('Kravpivin2009.pickle', 'rb') as handle:
#     Krav_data = pickle.load(handle)
#
# # clusters = cluster_doc_representation(Kravpivin_miniLM, method='aglo', num_clusters=15)
# true_kw = Krav_data[2]
# true_kw = true_kw[:50]
# # grouped_true_kw = group_keywords_from_file(clusters, true_kw, 15)
# # large_kw = do_large_doc_approach(clusters, docs, 15)
# # aggregate_kw = do_aggregate_approach(clusters, docs, Kravpivin_miniLM, 15)
#
# # yakee = create_yake(max_ngram_size=6)
# # yakee_keywords = [get_yake_keywords(yakee, doc) for doc in docs]
# # s_keywords = strip_yake_scores(yakee_keywords)
# # print(s_keywords)
#
# print("Starting BERT")
# # s_keywords = do_bert_keyword_extraction(docs)
# s_keywords = [do_get_expand_sfpd_phrases([doc], background_docs) for doc in docs]
# print(s_keywords)
# print('MAP for BASE YAKE: {}'.format(calculate_mean_average_precision(s_keywords, true_kw)))

# print("MAP For YAKE Large DOC: {}".format(calculate_mean_average_precision(large_kw, grouped_true_kw)))
# print("MAP For YAKE aggregate average DOC: {}".format(calculate_mean_average_precision(aggregate_kw, grouped_true_kw)))


# grouped_aglo = group_clustered_documents(clusters, docs)




# print(len(Kravpivin_miniLM))
# print(type(Kravpivin_miniLM))
# print(Kravpivin_miniLM[0:2].size())




# yakee = create_yake(max_ngram_size=6)
# yakee_keywords = [get_yake_keywords(yakee, doc) for doc in docs]
#
# s_keywords = strip_yake_scores(yakee_keywords)
# print(s_keywords)
# print(calculate_mean_average_precision(s_keywords, true_kw))
# #
# grouped_true_kw = group_keywords_from_file(clusters, true_kw, 15)
#
# avg_docs = group_and_average_doc_embeddings(clusters, Kravpivin_miniLM, 15)
#
# yake_aggregate_average_kw = aggregate_compare_with_average_of_cluster(grouped_aglo, avg_docs, 15)
# yake_large_doc_kw = large_doc_extract(grouped_aglo, 15)
#
# sfpd_aggregate_average_kw = aggregate_compare_with_average_of_cluster(grouped_aglo, avg_docs, 15, method='sfpd')
# sfpd_large_doc_kw = large_doc_extract(grouped_aglo, 15, method='sfpd')
#
# print(yake_large_doc_kw)
# print(yake_aggregate_average_kw)
#
# # for i in range(15):
# #     print(calculate_mean_average_precision(stripped_kw[i], grouped_true_kw[i]))
#
# print("MAP For YAKE Large DOC: {}".format(calculate_mean_average_precision(yake_large_doc_kw, grouped_true_kw)))
# print("MAP For YAKE aggregate average DOC: {}".format(calculate_mean_average_precision(yake_aggregate_average_kw, grouped_true_kw)))
#
# print("MAP For SFPD Large DOC: {}".format(calculate_mean_average_precision(sfpd_large_doc_kw, grouped_true_kw)))
# print("MAP For SFPD aggregate average DOC: {}".format(calculate_mean_average_precision(sfpd_aggregate_average_kw, grouped_true_kw)))
