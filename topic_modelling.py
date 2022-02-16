import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def LDAvis(documents, file_name, num_of_topics=100):
    lda_dictionary = Dictionary(documents)

    lda_corpus = [lda_dictionary.doc2bow(doc) for doc in documents]

    print('Number of unique tokens: %d' % len(lda_dictionary))
    print('Number of documents: %d' % len(lda_corpus))

    num_topics = num_of_topics
    chunksize = 2000
    passes = 10
    iterations = 400
    eval_every = None

    temp = lda_dictionary[0]
    id2word = lda_dictionary.id2token

    model = LdaModel(
        corpus=lda_corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    top_topics = model.top_topics(lda_corpus)

    from pprint import pprint
    pprint(top_topics)

    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    vis_data = gensimvis.prepare(model, lda_corpus, lda_dictionary)
    pyLDAvis.save_html(vis_data, file_name)
