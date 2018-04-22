from lib import features_word2vec, model_siamese_lstm
import pandas as pd

# The next steps:
# 1. Check

# Will ingest/clean data and save the following:
# 1. cleaned text translated to array of word indices: imdb_indices.pickle
# 2. word2vec model, where the indices/word vecs are stored:  300features_40minwords_10context
# 3. word embeddings: this is the index to wordvec mapping derived from 2.

# ingestion clean data
# create word embedding
# create word indices that can be mapped to word embedding
labeled_data_path = "./data/train.csv"
model_path = "./model/300features_40minwords_10context_quora"
embedding_path = "./model/embedding_weights_quora.pkl"
text2indices_path = "./model/quora_indices.pickle"
maxSeqLength = 50

def data_prep_quora():
    # Read data
    # Use the kaggle Bag of words vs Bag of popcorn data:
    # https://www.kaggle.com/c/word2vec-nlp-tutorial/data

    data = pd.read_csv(labeled_data_path, delimiter=",", engine = "python", encoding = "utf8")
    data_concat = pd.concat([data[["question1"]], data[["question2"]].rename(columns={"question2":"question1"})], axis = 0)
    model = features_word2vec.get_word2vec_model(data_concat, "question1", num_features=300, downsampling=1e-3, model_path=model_path)

    embedding_weights = features_word2vec.create_embedding_weights(model, writeEmbeddingFileName = "./model/embedding_weights_quora_tmp.pkl" )

    features1 = features_word2vec.get_indices_word2vec(data, "question1", model, maxLength=maxSeqLength,
                                                             writeIndexFileName="./model/quora_indices1.pickle",
                                                             padLeft=True)
    features2 = features_word2vec.get_indices_word2vec(data, "question2", model, maxLength=maxSeqLength,
                                                         writeIndexFileName="./model/quora_indices2.pickle",
                                                         padLeft=True)
    label = data["is_duplicate"]

    return model, embedding_weights, features1, features2, label

if __name__ == '__main__':

    word2vecmodel, embedding_weights, features1, features2, label = data_prep_quora()
    model = model_siamese_lstm.SiameseLSTMModel(features1, features2, label, embedding_weights, maxSeqLength)
    model.train_epochs(20)






