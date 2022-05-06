# from preprocessing import *
#
#
# a = ["This is a 1 TEST sentence that contains", "ALL OF THE STOPWORDS !@!@!@@", "Good TESTS go For EdGe CaSes"]
#
# print(clean_corpus(a, [], False, True))
#
#
# from sentence_transformers import SentenceTransformer
#
# data = ["This is a 1 TEST sentence that contains", "ALL OF THE STOPWORDS !@!@!@@", "Good TESTS go For EdGe CaSes"]
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# embeddings = model.encode(data, show_progress_bar=True)

# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer('allenai/scibert_scivocab_uncased')
#
# print(model)
import pandas as pd
import torch

from keyword_extraction_yake import create_yake, get_yake_keywords, do_get_expand_sfpd_phrases
from bert import generate_candidate_keywords, create_transformer_model, get_most_cosine_similar, \
    do_bert_keyword_extraction
import pickle
from runner import cluster_doc_representation, group_clustered_documents
from ast import literal_eval


def calculate_mean_average_precision(predicted_keywords, true_keywords):
    all_precisions = []
    for i, keywords in enumerate(predicted_keywords):
        temp_precisions = []
        temp_correct = 0
        for j, kw in enumerate(keywords):
            if kw in true_keywords[i]:
                temp_correct += 1
                temp_precisions.append(temp_correct / (j + 1))
        if temp_correct == 0:
            all_precisions.append(0)
        else:
            all_precisions.append(sum(temp_precisions) / temp_correct)
    mean_average_precision = sum(all_precisions) / len(predicted_keywords)
    return mean_average_precision


# import numpy as np
# from sklearn.metrics import average_precision_score
#
# y=["yes", "no", "maybe"]
# x=["yes", "nah", "chance"]
#
# print(average_precision_score(y, x))

# tkeywords = [["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "10"]]
# true_kw = [["2", "4"], ["6", "9", "10"]]
#
# print(calculate_mean_average_precision(tkeywords, true_kw))

df = pd.read_csv("Kravpivin2009_with_keywords.csv")
docs = df['1'].tolist()
docs = docs[:50]
with open("Kravpivin2009_all-MiniLM-L6-v2.pickle", 'rb') as handle:
    Kravpivin_miniLM = pickle.load(handle)
Kravpivin_miniLM = Kravpivin_miniLM[:50]
#
clusters = cluster_doc_representation(Kravpivin_miniLM, method='aglo', num_clusters=15)
grouped_aglo = group_clustered_documents(clusters, docs)

print(len(Kravpivin_miniLM))
print(type(Kravpivin_miniLM))
print(Kravpivin_miniLM[0:2].size())


# print(torch.mean(Kravpivin_miniLM[:2],0  ,keepdim=True))
# print(Kravpivin_miniLM[0])
# print(Kravpivin_miniLM[1])
# print(Kravpivin_miniLM[:3].size())
#
# mean_tensor = torch.mean(Kravpivin_miniLM[:3], 0, keepdim=True)
#
# print(mean_tensor.size())
# print(Kravpivin_miniLM[0].size())
#
# reshaped_mean = torch.squeeze(mean_tensor)
# print(reshaped_mean.size())


def large_doc_extract(grouped_documents, num_clusters):
    yak = create_yake(num_of_keywords=50)
    # grouped_large_kw = ["" for i in range(num_clusters)]
    combined_docs = []
    for i in range(num_clusters):
        print('STARTING ON CLUSTER {}'.format(str(i)))
        temp_combined_doc = ''
        for doc in grouped_documents[i]:
            temp_combined_doc = temp_combined_doc + ' ' + doc
        combined_docs.append(temp_combined_doc)
    keywords = []
    for i, combined_doc in enumerate(combined_docs):
        print('STARTING KEYWORD ON CLUSTER {}'.format(str(i)))

        # CODE FOR YAKE KEYWORDS, MUST!!! STRIP THE KEYWORDS AFTER
        # keywords.append(get_yake_keywords(yak, combined_doc))

        # BERT APPROACH SHOULD WORK HERE, LOOK AT HYPERPARAMETERS FOR NGRAM RANGE
        # keywords.append(do_bert_keyword_extraction([combined_doc])[0])

        # SFPD SHOULD WORK HERE, LOOK INTO PRE PROCESSING (\n included in potential phrase)
        background_docs = [x for j,x in enumerate(combined_docs) if j!=i]
        keywords.append(do_get_expand_sfpd_phrases([combined_doc], background_docs))

    # IF YAKE, MAKE SURE TO STRIP
    # stripped_keywords = strip_scores_from_keywords(keywords)
    # keywords = [get_yake_keywords(yak, combined_doc) for combined_doc in combined_docs]
    return keywords


def aggregate_compare_with_average_of_cluster(grouped_documents, average_doc_embeddings, num_clusters):
    yak = create_yake(num_of_keywords=100)
    grouped_candidate_keywords = []
    for i in range(num_clusters):
        temp_candidate_keywords = []
        for doc in grouped_documents[i]:
            # temp_keywords = get_yake_keywords(yak, doc)
            # temp_candidate_keywords = temp_candidate_keywords + temp_keywords

            # keywords.append(do_bert_keyword_extraction([combined_doc])[0])
            # temp_candidate_keywords = temp_candidate_keywords + do_bert_keyword_extraction([doc])[0]
            # temp_candidate_keywords = temp_candidate_keywords + get_yake_keywords(yak, doc)
            background_docs = []
            for j in range(num_clusters):
                if j != i:
                    background_docs = background_docs + grouped_documents[i]
            temp_candidate_keywords = temp_candidate_keywords + do_get_expand_sfpd_phrases([doc], background_docs)

        grouped_candidate_keywords.append(list(set(temp_candidate_keywords)))

    # for ces in grouped_candidate_keywords:
    #     dup = {}
    #     for ce in ces:
    #         kles = dup.get(ce, 'a')
    #         if ce != 'a':
    #             print("DUPE {}".format(ce))
    #         dup[ce] = kles

    #NEEDED FOR YAKE
    # grouped_stripped_candidates = strip_scores_from_keywords(grouped_candidate_keywords)

    #BERT OR SFPD USE THIS!!!!!
    grouped_stripped_candidates = grouped_candidate_keywords

    no_dupe_grouped_stripped = []
    for grs in grouped_stripped_candidates:
        no_dupe_grouped_stripped.append(list(set(grs)))

    model = create_transformer_model('all-MiniLM-L6-v2')
    grouped_candidate_embeddings = [[model.encode(cand) for cand in candidates] for candidates in
                                    no_dupe_grouped_stripped]

    aggregate_average_keywords = []
    for i in range(num_clusters):
        aggregate_average_keywords.append(
            get_most_cosine_similar(average_doc_embeddings[i].reshape(1, -1), grouped_candidate_embeddings[i],
                                    no_dupe_grouped_stripped[i], top_n=25))
    return aggregate_average_keywords


def group_and_average_doc_embeddings(cluster_labels, doc_embeddings, num_clusters):
    clustered_doc_embeddings = {}
    for i, label in enumerate(cluster_labels):
        doc_embeddings_in_cluster = clustered_doc_embeddings.get(label, [])
        doc_embeddings_in_cluster.append(doc_embeddings[i])

        clustered_doc_embeddings[label] = doc_embeddings_in_cluster

    average_clusters_embedding = []
    # print(clustered_doc_embeddings[1])
    # average_clusters_embedding = [torch.squeeze(torch.mean(clustered_doc_embeddings[i], 0, keepdim=True)) for i in range(num_clusters)]
    for i in range(num_clusters):
        temp_stack_embedding = torch.stack(clustered_doc_embeddings[i])
        # print(temp_cat_embedding.size())
        # print(clustered_doc_embeddings[i])
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
        combined_keywords.append(temp_combined_keywords)
    return combined_keywords


def strip_scores_from_keywords(grouped_keywords):
    stripped_keywords = []
    for group in grouped_keywords:
        temp_stripped = []
        for non_stripped_keywords in group:
            temp_stripped.append(non_stripped_keywords[0])
        stripped_keywords.append(temp_stripped)
    return stripped_keywords


# with open('Kravpivin2009.pickle', 'rb') as handle:
#     Krav_data = pickle.load(handle)
#
# true_kw = Krav_data[2]
# true_kw = true_kw[:50]
# grouped_true_kw = group_keywords_from_file(clusters, true_kw, 15)
#
# avg_docs = group_and_average_doc_embeddings(clusters, Kravpivin_miniLM, 15)
# aggregate_average_kw = aggregate_compare_with_average_of_cluster(grouped_aglo, avg_docs, 15)
#
# # large_doc_kw = large_doc_extract(grouped_aglo, 15)
# # print(large_doc_kw)
# print(aggregate_average_kw)
#
# # for i in range(15):
# #     print(calculate_mean_average_precision(stripped_kw[i], grouped_true_kw[i]))
#
# # print("MAP For Large DOC: {}".format(calculate_mean_average_precision(large_doc_kw, grouped_true_kw)))
# print("MAP For aggregate average DOC: {}".format(calculate_mean_average_precision(aggregate_average_kw, grouped_true_kw)))
# print(type(Kravpivin_miniLM))
# print(len(avg_docs))
# for do in avg_docs:
#     print(do.size())
# grouped_kw = large_doc_extract(grouped_aglo, 15)
# print(grouped_kw)
# with open('Kravpivin_15_cluster_50_keyword_test.pickle', 'wb') as handle:
#     pickle.dump(grouped_kw, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('Kravpivin_15_cluster_50_keyword_test.pickle', 'rb') as handle:
#     group_kw = pickle.load(handle)
#

# with open('Kravpivin2009.pickle', 'rb') as handle:
#     Krav_data = pickle.load(handle)
#
# true_kw = Krav_data[2]
# for tr_kw in true_kw:
#     temp_kws = [len(kw.split()) for kw in tr_kw]
#     print(tr_kw[temp_kws.index(max(temp_kws))])
#     print(max(temp_kws))


# print(true_kw)

# grouped_true_kw = group_keywords_from_file(clusters, true_kw, 15)
# print(grouped_true_kw[1])
# print(group_kw[1])
# print(len(Krav_data[2]))
# stripped_kw = strip_scores_from_keywords(group_kw)
# print(stripped_kw[1])
#
# for i in range(15):
#     print('')
#     # print(calculate_mean_average_precision(stripped_kw[i], grouped_true_kw[i]))


# print(len(grouped_aglo[0]))
#
# for gr in grouped_aglo:
#     print(gr)
#
# print(max([len(gr) for gr in grouped_aglo]))

# print(grouped_aglo[1])
# for g in grouped_aglo[1]:
#     print('+++++++++')
#     print(g)
# for k, v in grouped_aglo:
#     print('{} -------- {}'.format(k, v[:300]))
# df = pd.read_csv("Kravpivin2009_with_keywords.csv")
# docs = df['1'].tolist()
# docs = docs[:10]
#
# appended_text = ''
# for doc in docs:
#     appended_text = appended_text + doc
#
# # print(appended_text)
# app_text = [appended_text]
#
# yak = create_yake()
# kw = get_yake_keywords(yak, appended_text)
# print(kw)
#
# candidates = generate_candidate_keywords(app_text)
# print(candidates)
