import yake


def create_yake(language="en", max_ngram_size=1, deduplication_thresold=0.9, deduplication_algo='jaro',
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
