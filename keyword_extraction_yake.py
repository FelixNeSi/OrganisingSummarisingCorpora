import yake
from sfpd.words import top_words_llr, top_words_sfpd, top_words_chi2, count_words
from sfpd.phrases import get_top_phrases


def create_yake(language="en", max_ngram_size=6, deduplication_thresold=0.9, deduplication_algo='jaro',
                window_size=3, num_of_keywords=20):
    custom_kw_extractor = yake.KeywordExtractor(lan=language,
                                                n=max_ngram_size,
                                                dedupLim=deduplication_thresold,
                                                dedupFunc=deduplication_algo,
                                                windowsSize=window_size,
                                                top=num_of_keywords,
                                                features=None)

    return custom_kw_extractor


def get_yake_keywords(yake_model, text):
    keywords = yake_model.extract_keywords(text)
    return keywords


def get_sfpd_target_background_counts(target_docs, background_docs, min_count=4):
    target_counts = count_words(target_docs, min_count=min_count, language="en")
    background_counts = count_words(background_docs, min_count=min_count, language="en")
    return target_counts, background_counts


def get_sfpd_root_words(target_counts, background_counts, method="sfpd", num_keywords=450):
    if method == "sfpd":
        words = top_words_sfpd(target_counts, background_counts, num_keywords)
    elif method == "loglikeli":
        words = top_words_llr(target_counts, background_counts)
    elif method == "chisquare":
        words = top_words_chi2(target_counts, background_counts)
    else:
        print("Please select a known method: sfpd, loglikeli, chisquare")
        return "error!"
    return words


def expand_sfpd_phrases(words, data):
    top_phrases = get_top_phrases(words["word"].values, data)
    return top_phrases


def do_get_expand_sfpd_phrases(data, background, min_count=4, root_word_method="sfpd", num_keywords=450):
    t, b = get_sfpd_target_background_counts(data, background, 2)
    word = get_sfpd_root_words(t, b, num_keywords=num_keywords)
    phrases = expand_sfpd_phrases(word, data).raw_phrases()
    print(phrases)
    return phrases

# data = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes",
#         "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
#     , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"]
# #
# # background = ["Dont use this dopython -m spacy download encument!", "No testing here"]
# #
# # do_get_expand_sfpd_phrases(data, background)
#
# yak = create_yake()
# kw = [get_yake_keywords(yak, dat) for dat in data]
# print(kw)
