from preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import itertools

def create_transformer_model(model_name='distilbert-base-nli-mean-tokens'):
    model = SentenceTransformer(model_name)
    return model


def encode_corpus_with_model(data, model):
    embeddings = model.encode(data, show_progress_bar=True, convert_to_tensor=True)
    return embeddings


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
sh

def fast_cluster(corpus_embeddings, min_community_size=25, threshold=0.75):
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold,
                                        init_max_size=len(corpus_embeddings))
    assignments = [None for x in range(len(corpus_embeddings))]
    for i, cluster in enumerate(clusters):
        for doc in cluster:
            assignments[doc] = i
    # print(assignments)
    return assignments


def print_clusters(cluster_assignment, corpus):
    clustered_sentences = []
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i + 1)
        print(cluster)
        print("")


def normalise_embeddings(embeddings):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def generate_candidate_keywords(doc, min_ngram=1, max_ngram=1, remove_stopwords=True):
    n_gram_range = (min_ngram, max_ngram)
    stop_words = "english"

    # Extract candidate words/phrases
    if remove_stopwords:
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(doc)
    else:
        count = CountVectorizer(ngram_range=n_gram_range).fit(doc)
    candidates = count.get_feature_names()
    return candidates


def get_most_cosine_similar(doc_embedding, candidate_embeddings, candidates, top_n=5):
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords


def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings,
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def mmr(doc_embedding, word_embeddings, candidates, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [candidates[idx] for idx in keywords_idx]


def do_bert_keyword_extraction(data, sim_method="cosine", top_n=5, nr_candidates=10, diversity=0.2):
    candidates = [generate_candidate_keywords([doc]) for doc in data]
    model = create_transformer_model()
    doc_embeddings = [model.encode([doc]) for doc in data]
    candidate_embeddings = [model.encode(cand) for cand in candidates]
    keywords = []
    for i in range(len(data)):
        if sim_method == "cosine":
            keywords.append(get_most_cosine_similar(doc_embeddings[i], candidate_embeddings[i], candidates[i], top_n=top_n))
        elif sim_method == "max_sum":
            keywords.append(max_sum_sim(doc_embeddings[i], candidate_embeddings[i], candidates[i], top_n, nr_candidates))
        elif sim_method == "max_marginal":
            keywords.append(mmr(doc_embeddings[i], candidate_embeddings[i], candidates[i], top_n, diversity))
    return keywords
    # if sim_method == "cosine":
    #     get_most_cosine_similar()
    # print(candidates)
    # print(len(doc_embeddings))
    # print(len(candidate_embeddings))
    # print(len(candidates))




# data = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes",
#         "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
#     , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"]
# #
# # # ['cases', 'contains', 'edge', 'good', 'sentence', 'stopwords', 'test', 'tests']
# #
# # # print(generate_candidate_keywords(data[0]))
# # print(do_bert_keyword_extraction(data))
# model = create_transformer_model('distilbert-base-nli-mean-tokens')
# embeddings = encode_corpus_with_model(data, model)
# # embeddings = normalise_embeddings(embeddings)
# assignments = kmeans_cluster(embeddings, 3)
# # for i, cluster in enumerate(clusters):
# #     print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
# #     for sentence_id in cluster[0:3]:
# #         print("\t", corpus_sentences[sentence_id])
# #     print("\t", "...")
# #     for sentence_id in cluster[-3:]:
# #         print("\t", corpus_sentences[sentence_id])
# print(assignments)
# print_clusters(assignments, data)
#
# # print(embeddings)
