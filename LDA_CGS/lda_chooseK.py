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

    # dictionary cho coherence
    dictionary = Dictionary(training_tokenized_articles)

    K = 170
    eta = 0.01

    model = tp.LDAModel(
        k=K,
        eta=eta,
        seed=1
    )

    # bật tự tối ưu alpha
    model.optim_interval = 10

    for doc in training_tokenized_articles:
        model.add_doc(doc)

    model.burn_in = 200

    for i in range(0, 2000, 50):
        model.train(50)
        print(f"Iteration {i+50}, alpha={model.alpha}")

    model.save("LDA_CGS/lda_cgs.bin")

    # lấy topic words
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

    # Coherence:  0.6075839815033406
    # Final alpha: [0.04333441 0.02324084 0.01448482 0.01464726 0.0174387  0.01596867
    #    0.00804897 0.00873917 0.00756997 0.00838167 0.00819667 0.01629101
    #    0.01240583 0.02385472 0.01438595 0.01961191 0.01024708 0.01648283
    #    0.02924608 0.025671   0.07206335 0.03258068 0.03310993 0.03199318
    #    0.05366294 0.01921228 0.00939754 0.0152085  0.01215358 0.0890089
    #    0.02592721 0.11669052 0.02982567 0.01044151 0.01025772 0.02043045
    #    0.0207249  0.14428109 0.0134557  0.02113348 0.01544503 0.0084021
    #    0.01235437 0.04611001 0.03526329 0.02438884 0.00738518 0.01998173
    #    0.02643898 0.02423259 0.0194717  0.02439296 0.02819843 0.03629735
    #    0.06454354 0.02397699 0.12248772 0.01041363 0.0164582  0.02739595
    #    0.009129   0.02211297 0.13108087 0.03002579 0.01925194 0.03101462
    #    0.03966102 0.07621409 0.00934055 0.0268712  0.01242323 0.01178014
    #    0.04836091 0.01521945 0.03666236 0.01518571 0.05248911 0.06518489
    #    0.02677529 0.01553565 0.01944318 0.01164857 0.01045451 0.00814589
    #    0.0156094  0.01751492 0.03406273 0.01428766 0.06028875 0.02407254
    #    0.01652781 0.02421345 0.01548615 0.01541582 0.02179429 0.01465298
    #    0.03306933 0.04364436 0.00719807 0.01655564 0.01551737 0.02461631
    #    0.09012523 0.01951896 0.00753955 0.02351633 0.0303056  0.01226569
    #    0.01834178 0.02125836 0.02793021 0.01270757 0.01548327 0.04499729
    #    0.01843279 0.02510175 0.10085906 0.01122282 0.01255424 0.02824844
    #    0.04514321 0.0218269  0.02534801 0.06894824 0.01452798 0.03067168
    #    0.02141812 0.02503852 0.01788523 0.02102406 0.02806411 0.01921217
    #    0.01476118 0.01040152 0.04067678 0.01895035 0.01735326 0.01308235
    #    0.02394711 0.02690187 0.02534753 0.03089438 0.03194364 0.0848883
    #    0.01342107 0.08389002 0.01260276 0.00889939 0.00938337 0.07869133
    #    0.06137612 0.01908484 0.04760981 0.10381908 0.01243927 0.01138944
    #    0.00846034 0.01365979 0.01164834 0.0301141  0.00876758 0.02126697
    #    0.00836925 0.02301139 0.06920676 0.00835859 0.04602877 0.0414193
    #    0.01639781 0.08571059]