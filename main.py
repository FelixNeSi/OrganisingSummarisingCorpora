from preprocessing import *


a = ["This is a 1 TEST sentence that contains", "ALL OF THE STOPWORDS !@!@!@@", "Good TESTS go For EdGe CaSes"]

print(clean_corpus(a, [], False, True))


from sentence_transformers import SentenceTransformer

data = ["This is a 1 TEST sentence that contains", "ALL OF THE STOPWORDS !@!@!@@", "Good TESTS go For EdGe CaSes"]
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)