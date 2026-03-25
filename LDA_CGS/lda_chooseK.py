import pickle
import tomotopy as tp
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


def load_data(address):
    with open(address, 'rb') as f:
        tokenized_articles = pickle.load(f)
    return tokenized_articles


if __name__ == "__main__":

    training_tokenized_articles = load_data("Dataset/train_data_for_LDA.pkl")
    validation_tokenized_articles = load_data("Dataset/val_data_for_LDA.pkl")

    # dictionary for coherence
    dictionary = Dictionary(training_tokenized_articles)

    K = 20
    eta = 0.01

    model = tp.LDAModel(
        k=K,
        eta=eta,
        seed=1
    )

    #Self-optimizing alpha (hyperparameter for topic distributions in articles)
    model.optim_interval = 10

    for doc in training_tokenized_articles:
        model.add_doc(doc)

    model.burn_in = 200

    for i in range(0, 1000, 50): #Train 20 times
        model.train(50)
        print(f"Iteration {i+50}, alpha={model.alpha}")

    model.save("LDA_CGS/lda_cgs.bin") #Save the LDA model

    #Get topic words
    topics = []
    for k in range(model.k):
        topic_words = [w for w, _ in model.get_topic_words(k, top_n=10)]
        topics.append(topic_words)

    coherence_model = CoherenceModel(
        topics=topics,
        texts=validation_tokenized_articles,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence = coherence_model.get_coherence()

    print("Final alpha:", model.alpha)
    print("Coherence:", coherence)

    # Coherence:  0.5388787444118792
    # Final alpha: [0.06291872 0.09931275 0.03810117 0.10546406 0.08161369 0.11097357
        #0.07579993 0.08741148 0.07599209 0.21594176 0.08060992 0.1488358
        #0.18991071 0.07399414 0.06240193 0.08549261 0.08227756 0.05900126
        #0.05309661 0.0664641 ]