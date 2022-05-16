import pandas as pd
import matplotlib.pyplot as plt

def lda_compare_best():
    df_1 = pd.read_csv("experiment/kdd-science/kdd-science-aglo-lda-topic-nums-10-clean.csv")
    df_3 = pd.read_csv("experiment/kdd-science/kdd-science-aglo-lda-topic-nums-10-no-pre.csv")
    df_2 = pd.read_csv("experiment/kdd-science/kdd-science-aglo-lda-topic-nums-35-clean-no-lemma.csv")

    cluster_nums_1 = df_1['n_cluster'].tolist()
    cluster_nums_2 = df_2['n_cluster'].tolist()
    cluster_nums_3 = df_3['n_cluster'].tolist()

    silhouette_1 = df_1['silhouette score'].tolist()
    silhouette_2 = df_2['silhouette score'].tolist()
    silhouette_3 = df_3['silhouette score'].tolist()

    plt.plot(cluster_nums_1[:12], silhouette_1[:12], label="10: clean")
    plt.plot(cluster_nums_2[:12], silhouette_2[:12], label="10: no pre")
    plt.plot(cluster_nums_3[:12], silhouette_3[:12], label="35: clean no lemma")
    # plt.plot(epoch_num[:500], end_data[1], label = "Best genotype at Epoch: 475")/
    plt.xlabel('Cluster Numbers')
    plt.ylabel('Silhouette Score')
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.legend(title='Number of topics, Preprocess Method')
    plt.title("kdd-science Agglomerative LDA Best preprocessing & Number of Topics")
    plt.show()


def full_bert_aglo_kmeans_plot(corpus_name, plot_amount='all', upper_cluster_limit=14):
    df_1 = pd.read_csv(
        "experiment/{}/{}-aglo-allenai-scibert_scivocab_uncased-no-pre.csv".format(corpus_name, corpus_name))
    df_2 = pd.read_csv(
        "experiment/{}/{}-kmeans-allenai-scibert_scivocab_uncased-no-pre.csv".format(corpus_name, corpus_name))
    df_3 = pd.read_csv(
        "experiment/{}/{}-aglo-distilbert-base-nli-mean-tokens-no-pre.csv".format(corpus_name, corpus_name))
    df_4 = pd.read_csv(
        "experiment/{}/{}-kmeans-distilbert-base-nli-mean-tokens-no-pre.csv".format(corpus_name, corpus_name))
    df_5 = pd.read_csv("experiment/{}/{}-aglo-all-MiniLM-L6-v2-no-pre.csv".format(corpus_name, corpus_name))
    df_6 = pd.read_csv("experiment/{}/{}-kmeans-all-MiniLM-L6-v2-no-pre.csv".format(corpus_name, corpus_name))

    cluster_nums_1 = df_1['n_cluster'].tolist()
    cluster_nums_2 = df_2['n_cluster'].tolist()
    cluster_nums_3 = df_3['n_cluster'].tolist()
    cluster_nums_4 = df_4['n_cluster'].tolist()
    cluster_nums_5 = df_5['n_cluster'].tolist()
    cluster_nums_6 = df_6['n_cluster'].tolist()
    silhouette_1 = df_1['silhouette score'].tolist()
    silhouette_2 = df_2['silhouette score'].tolist()
    silhouette_3 = df_3['silhouette score'].tolist()
    silhouette_4 = df_4['silhouette score'].tolist()
    silhouette_5 = df_5['silhouette score'].tolist()
    silhouette_6 = df_6['silhouette score'].tolist()
    if plot_amount == 'all':
        plt.plot(cluster_nums_1[:upper_cluster_limit], silhouette_1[:upper_cluster_limit], label="Agglomerate scibert")
        plt.plot(cluster_nums_2[:upper_cluster_limit], silhouette_2[:upper_cluster_limit], label="KMeans scibert")
        plt.plot(cluster_nums_3[:upper_cluster_limit], silhouette_3[:upper_cluster_limit], label="Agglomerate distil")
        plt.plot(cluster_nums_4[:upper_cluster_limit], silhouette_4[:upper_cluster_limit], label="KMeans distil")
        plt.plot(cluster_nums_5[:upper_cluster_limit], silhouette_5[:upper_cluster_limit], label="Agglomerate MiniLM")
        plt.plot(cluster_nums_6[:upper_cluster_limit], silhouette_6[:upper_cluster_limit], label="KMeans MiniLM")
    elif plot_amount == 'kmeans':
        plt.plot(cluster_nums_2[:upper_cluster_limit], silhouette_2[:upper_cluster_limit], label="KMeans scibert")
        plt.plot(cluster_nums_4[:upper_cluster_limit], silhouette_4[:upper_cluster_limit], label="KMeans distil")
        plt.plot(cluster_nums_6[:upper_cluster_limit], silhouette_6[:upper_cluster_limit], label="KMeans MiniLM")
    elif plot_amount == 'aglo':
        plt.plot(cluster_nums_1[:upper_cluster_limit], silhouette_1[:upper_cluster_limit], label="Agglomerate scibert")
        plt.plot(cluster_nums_3[:upper_cluster_limit], silhouette_3[:upper_cluster_limit], label="Agglomerate distil")
        plt.plot(cluster_nums_5[:upper_cluster_limit], silhouette_5[:upper_cluster_limit], label="Agglomerate MiniLM")
    # plt.plot(epoch_num[:500], end_data[1], label = "Best genotype at Epoch: 475")/
    plt.xlabel('Cluster Numbers')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, 0.95),
               ncol=2, fancybox=True, shadow=True)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.title('Kravpivin2009 BERT Clustering: Silhouette Score vs Cluster Numbers')
    plt.show()


def lda_aglo_kmeans_plot(corpus_name, topic_num_limit=9, upper_cluster_limit=12, preprocessing='no-pre'):
    topic_nums = [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 250, 300]
    df_array, cluster_nums_aglo, scores_aglo, cluster_nums_kmeans, scores_kmeans = [], [], [], [], []
    # plot_process_title = ['No Preprocessing', 'Full Preprocessing', 'Full Preprocessing No Lemmatisation']
    if preprocessing == 'clean':
        plot_process_title = 'Full Preprocessing'
    elif preprocessing == 'no-pre':
        plot_process_title = 'No Preprocessing'
    elif preprocessing == 'clean-no-lemma':
        plot_process_title = 'Full Preprocessing no Lemmatisation'
    for topic_num in topic_nums[:topic_num_limit]:
        # df_array.append(pd.read_csv("experiment/{}/{}-aglo-lda-topic-nums-{}-clean.csv".format(corpus_name, corpus_name, topic_num)))
        temp_df = pd.read_csv("experiment/{}/{}-aglo-lda-topic-nums-{}-{}.csv".format(corpus_name, corpus_name, topic_num, preprocessing))
        cluster_nums_aglo.append(temp_df['n_cluster'].tolist())
        scores_aglo.append(temp_df['silhouette score'].tolist())
        temp_df = pd.read_csv(
            "experiment/{}/{}-kmeans-lda-topic-nums-{}-{}.csv".format(corpus_name, corpus_name, topic_num, preprocessing))
        cluster_nums_kmeans.append(temp_df['n_cluster'].tolist())
        scores_kmeans.append(temp_df['silhouette score'].tolist())
    for i, cluster_num in enumerate(cluster_nums_aglo):
        plt.subplot(2, 1, 1)
        plt.plot(cluster_num[:upper_cluster_limit], scores_aglo[i][:upper_cluster_limit], label=" {}".format(str(topic_nums[i])))
        plt.subplot(2, 1, 2)
        plt.plot(cluster_num[:upper_cluster_limit], scores_kmeans[i][:upper_cluster_limit],
                 label=" {}".format(str(topic_nums[i])))
    plt.subplot(2, 1, 1)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.xlabel('Cluster Numbers')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.5),
               ncol=3, fancybox=True, shadow=True, title='Topic Numbers')
    plt.title("500n-KPCrowd Agglomerative LDA - {}".format(plot_process_title))
    plt.subplot(2, 1, 2)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.xlabel('Cluster Numbers')
    plt.ylabel('Silhouette Ccore')
    plt.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.5),
              ncol=3, fancybox=True, shadow=True, title='Topic Numbers')
    plt.title("500n-KPCrowd Kmeans LDA - {}".format(plot_process_title))
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# def lda_compare_best(topic_num):
#     topic_nums = [5, 10, 15, 25, 35, 50]
#     df_array, cluster_nums_aglo, scores_aglo, cluster_nums_kmeans, scores_kmeans = [], [], [], [], []
#     pre_process = ['clean', 'clean-no-lemma', 'no-pre']
#     for topic_num in topic_nums:
#         # df_array.append(pd.read_csv("experiment/{}/{}-aglo-lda-topic-nums-{}-clean.csv".format(corpus_name, corpus_name, topic_num)))
#         temp_df = pd.read_csv(
#             "experiment/{}/{}-aglo-lda-topic-nums-{}-{}.csv".format(corpus_name, corpus_name, topic_num, preprocessing))
#         cluster_nums_aglo.append(temp_df['n_cluster'].tolist())
#         scores_aglo.append(temp_df['silhouette score'].tolist())
#         temp_df = pd.read_csv(
#             "experiment/{}/{}-kmeans-lda-topic-nums-{}-{}.csv".format(corpus_name, corpus_name, topic_num,
#                                                                       preprocessing))
#         cluster_nums_kmeans.append(temp_df['n_cluster'].tolist())
#         scores_kmeans.append(temp_df['silhouette score'].tolist())
#     for i, cluster_num in enumerate(cluster_nums_aglo):
#         plt.subplot(2, 1, 1)
#         plt.plot(cluster_num[:upper_cluster_limit], scores_aglo[i][:upper_cluster_limit],
#                  label=" {}".format(str(topic_nums[i])))
#         plt.subplot(2, 1, 2)
#         plt.plot(cluster_num[:upper_cluster_limit], scores_kmeans[i][:upper_cluster_limit],
#                  label=" {}".format(str(topic_nums[i])))
#     plt.subplot(2, 1, 1)
#     plt.grid(True, which='both')
#     plt.minorticks_on()
#     plt.xlabel('Cluster Nums')
#     plt.ylabel('silhouette score')
#     plt.legend()
#     plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.5),
#                ncol=3, fancybox=True, shadow=True, title='Topic Nums')
#     plt.title("kdd-science Agglomerative LDA - full preprocessing no lemmatisation")
#     plt.subplot(2, 1, 2)
#     plt.grid(True, which='both')
#     plt.minorticks_on()
#     plt.xlabel('Cluster Nums')
#     plt.ylabel('silhouette score')
#     plt.legend()
#     plt.legend(loc='upper center', bbox_to_anchor=(0.75, 0.5),
#                ncol=3, fancybox=True, shadow=True, title='Topic Nums')
#     plt.title("kdd-science Kmeans LDA - full preprocessing no lemmatisation")
#     plt.grid(True, which='both')
#     plt.minorticks_on()
#     plt.tight_layout()
#     plt.show()


# full_bert_aglo_kmeans_plot("marujo", 'all', 12)
# lda_aglo_kmeans_plot("marujo")

# full_bert_aglo_kmeans_plot("500n-KPCrowd", 'all', 14)
# lda_aglo_kmeans_plot("500n-KPCrowd")
# lda_compare_best()
# full_bert_aglo_kmeans_plot("Kravpivin2009", 'all', 18)
# lda_aglo_kmeans_plot("500n-KPCrowd", 6, preprocessing='no-pre')
# lda_aglo_kmeans_plot("500n-KPCrowd", 6, preprocessing='clean')
# lda_aglo_kmeans_plot("500n-KPCrowd", 6, preprocessing='clean-no-lemma')
# full_bert_aglo_kmeans_plot("Kravpivin2009", 'all', 12)
# lda_aglo_kmeans_plot("Kravpivin2009")

# df = pd.read_csv("kw_extraction_baseline_test.csv")
# print(df)
#
# app = df['approach'].tolist()
# scores = df['scores'].tolist()
#
# for i, ap in enumerate(app):
#     print("{} ----- {}".format(ap, scores[i]))

# df = pd.read_csv("kdd-science__kw_extraction_all.csv")
# #
# app = df['approach'].tolist()
# scores = df['scores'].tolist()
# # #
# for i, ap in enumerate(app):
#     print("{} ----- {}".format(ap, scores[i]))
#
# print("++++++++")
#
# df = pd.read_csv("kdd-science_lda35_kw_extraction_all.csv")
# #
# app = df['approach'].tolist()
# scores = df['scores'].tolist()
# # #
# for i, ap in enumerate(app):
#     print("{} ----- {}".format(ap, scores[i]))

df = pd.read_csv("marujo__kw_extraction_all.csv")
# df_yake = pd.read_csv("marujo__kw_extraction_all_yake.csv")

approach = df['approach'].tolist()
score = df['scores'].tolist()

# approach_1 = df_yake['approach'].tolist()
# score_1 = df_yake['scores'].tolist()
#
# full_approach = approach + approach_1
# full_scores = score + score_1

full_approach = approach
full_scores = score
# print(approach)
# print(approach_1)

for i, ctx in enumerate(full_approach):
    print('{} & {}  \\\\'.format(full_approach[i].replace('-', ' '), str(full_scores[i]*100)[:6]))
    print('\hline')

# #
# print("=============")
# df = pd.read_csv("Backup KW/LDA/KDD - 10  clean no lemma/kdd-science__kw_extraction_all_yake.csv")
# #
# app = df['approach'].tolist()
# scores = df['scores'].tolist()
# # #
# for i, ap in enumerate(app):
#     print("{} ----- {}".format(ap, scores[i]))
# df = pd.read_csv("marujo__kw_extraction_all.csv")
# #
# app = df['approach'].tolist()
# scores = df['scores'].tolist()
# # #
# for i, ap in enumerate(app):
#     print("{} ----- {}".format(ap, scores[i]))
# import pickle
# with open('{}__kw_testing_data.pickle'.format('500n-KPCrowd'), 'rb') as handle:
#     infoo = pickle.load(handle)
# #
# print(len(infoo[2][2]))
# print(infoo[0][2])
# with open('{}.pickle'.format('kdd-science'), 'rb') as handle:
#     corpus_data = pickle.load(handle)
# # corpus_documents = corpus_data[1]
# corpus_true_keywords = corpus_data[2]
#
# app = infoo[0]
# scores = infoo[1]
# keyw = infoo[2]
# sums = []
# new_kw = []
#
# print(app)
# # print(keyw[1])
#
# agg_large_app = []
# agg_large_kw = []
#
# base_app = []
# base_kw = []
#
# base_app.append(app[2])
# base_app.append(app[5])
#
# base_kw.append(keyw[2])
# base_kw.append(keyw[5])
#
# agg_large_app.append(app[0])
# agg_large_app.append(app[1])
# agg_large_app.append(app[3])
# agg_large_app.append(app[4])
#
# agg_large_kw.append(keyw[0])
# agg_large_kw.append(keyw[1])
# agg_large_kw.append(keyw[3])
# agg_large_kw.append(keyw[4])
#
# for i, kw in enumerate(agg_large_kw):
#     print('ON APPROACH {}'.format(agg_large_app[i]))
#     # print('Number of cl'len(kw))
#     sum = 0
#     for k in kw:
#         sum = sum + len(k)
#         # print(len(k))
#     print(sum)
#
# for i, kw in enumerate(base_kw):
#     print('ON APPROACH {}'.format(base_app[i]))
#     # print('Number of cl'len(kw))
#     sum = 0
#     for k in kw:
#         sum = sum + len(k)
#         # print(len(k))
#     print(sum)
#
# new_agg_large_kw = []
# new_base_kw = []
# print(len(agg_large_kw))
# print('wadwadwadwadwadawdaw')
# for i, kwz in enumerate(agg_large_kw):
#     temp_new_kw = []
#     for kw in kwz:
#         # print(len(kw[:290]))
#         temp_new_kw.append(kw[:300])
#         # for k in kw:
#         #     # print(k)
#         #     # temp_new_kw.append([k[:290]])
#     new_agg_large_kw.append(temp_new_kw)
#
# for i, kwz in enumerate(base_kw):
#     temp_new_kw = []
#     for kw in kwz:
#         # print(len(kw[:290]))
#         temp_new_kw.append(kw[:5])
#         # for k in kw:
#         #     # print(k)
#         #     # temp_new_kw.append([k[:290]])
#     new_base_kw.append(temp_new_kw)
#
# print('wadwadwadwadwadawdaw')
# for i, kwz in enumerate(new_agg_large_kw):
#     temp_new_kw = []
#     for kw in kwz:
#         print(len(kw))
#         # temp_new_kw.append(kw[:290])
#         # for k in kw:
#         #     # print(k)
#         #     # temp_new_kw.append([k[:290]])
#     # new_agg_large_kw.append(temp_new_kw)
# print('BASEAWESE')
# # for i, kwz in enumerate(new_base_kw):
# #     temp_new_kw = []
# #     for kw in kwz:
# #         print(len(kw))
#
# for i, kwv in enumerate(new_agg_large_kw):
#     print('ON!!!!! APPROACH {}'.format(agg_large_app[i]))
#     # print('Number of cl'len(kw))
#     sum = 0
#     for kz in kwv:
#         sum = sum + len(kz)
#         # print(len(k))
#     print(sum)
#
# for i, kwv in enumerate(new_base_kw):
#     print('ON!!!!! APPROACH {}'.format(base_app[i]))
#     # print('Number of cl'len(kw))
#     sum = 0
#     for kz in kwv:
#         sum = sum + len(kz)
#         # print(len(k))
#     print(sum)

# for i, ap in enumerate(app):
#     print("{} ----- {} ----- KW: {} ----- NO SUM: {}".format(ap, scores[i], sums[i], len(keyw[i])))





# for key in keyw:
#     temp_sum = 0
#     temp_kw = []
#     for ke in key:
#         # temp_kw.append(ke[:45])
#         temp_kw = temp_kw + ke[:45]
#         temp_sum = temp_sum + len(ke)
#         sums.append(temp_sum)
#     new_kw.append(temp_kw)
#
# for i, ap in enumerate(app):
#     print("{} ----- {} ----- KW: {} ----- NO SUM: {}".format(ap, scores[i], sums[i], len(keyw[i])))
#
# new_sums = []
# for key in new_kw:
#     temp_sum = 0
#     temp_kw = []
#     for ke in key:
#         # temp_kw.append(ke[:45])
#         temp_sum = temp_sum + len(ke)
#         new_sums.append(temp_sum)
#     # new_kw.append(temp_kw)
#
# print('=========================')
#
# for i, ap in enumerate(app):
#     print("{} ----- {} ----- KW: {} ----- NO SUM: {}".format(ap, scores[i], new_sums[i], len(new_kw[i])))
#
