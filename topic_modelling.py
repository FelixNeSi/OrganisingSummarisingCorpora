import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel


def prep_docs_for_topic_modelling(documents):
    documents = [doc.split() for doc in documents]
    lda_dictionary = Dictionary(documents)
    lda_corpus = [lda_dictionary.doc2bow(doc) for doc in documents]
    temp = lda_dictionary[0]
    id2word = lda_dictionary.id2token

    return lda_corpus, id2word


def lda(lda_corpus, id2word, num_topics=100, chunksize=2000, passes=10, iterations=400, eval_every=None, alpha='auto',
        eta='auto', minimum_probability=0.0):
    model = LdaModel(
        corpus=lda_corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every,
        minimum_probability=minimum_probability
    )

    return model


def get_topic_distributions(model, lda_corpus):
    topic_distributions = [[d[1] for d in dist] for dist in [model[text] for text in lda_corpus]]
    # only_topic_dist = [[d[1] for d in dist] for dist in topic_distributions]
    return topic_distributions


def do_get_topic_model(data):
    lda_corp, id2Word = prep_docs_for_topic_modelling(data)
    model = lda(lda_corp, id2Word, num_topics=2)
    topic_dist = get_topic_distributions(model, lda_corp)
    return topic_dist


# data = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes", "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
#         , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"]
#
# print(do_get_topic_model(data))


