from sklearn.cluster import KMeans, FeatureAgglomeration, DBSCAN, SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from MyTokenizer import MeanEmbeddingVectorizer, get_word2vec, MySentences
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
import matplotlib.cm as cm
import umap
import gensim
import sklearn
import copy
import numpy as np
import pandas


# Pasi:
# 1. Aplicam algoritmi de prelucrare a textului (BagOfWords, TFIDF, Word2Vec)
# 2. Curatam setul de date
# 3. Aplicam algoritmi de reducere a dimensionalitatii (UMAP, SVD, PCA)
# 4. Aplicam algorimtii din cealalta parte (SVC, Regr, KNN, NB, RF, DT)
# 5. Facem o clusterizare cu dupa curatarea setului de date


def compute_accuracy(prediction, cuisines):
    correct_predictions = 0
    for index in range(0, len(prediction)):
        if prediction[index] == cuisines[index]:
            correct_predictions += 1

    return (correct_predictions / len(prediction)) * 100


def print_plot(data, reducer, target, extractor):
    reduced_data = []
    if reducer == "UMAP":
        umap = UMAP()
        reduced_data = umap.fit_transform(data)
    elif reducer == "TSNE":
        tsne = TSNE(
                perplexity=100,
                n_components=2,
                init='pca',
                n_iter=5000,
                random_state=32
            )
        if extractor == "Word-2-Vec":
            reduced_data = tsne.fit_transform(data)
        else:
            reduced_data = tsne.fit_transform(data.toarray())

    targets = [i for i in range(0, len(target))]

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=targets, s=5, cmap='rainbow',  alpha=0.5)
    plt.suptitle(reducer + " " + extractor)
    plt.show()


def prediction(train, test, classes, test_cuisines):
    accuracies = dict()

    classifier_svc = LinearSVC(C=0.6)
    classifier_svc.fit(train, classes)
    prediction_svc = classifier_svc.predict(test)

    accuracies['Linear_SVC'] = compute_accuracy(prediction_svc, test_cuisines)

    classifier_regr = LogisticRegression(C=0.6)
    classifier_regr.fit(train, classes)
    prediction_regr = classifier_regr.predict(test)

    accuracies['Logistic_Regression'] = compute_accuracy(prediction_regr, test_cuisines)

    classfier_nb = RandomForestClassifier()
    classfier_nb.fit(train, classes)
    prediction_nb = classfier_nb.predict(test)

    accuracies['Random Forrest'] = compute_accuracy(prediction_nb, test_cuisines)

    accuracy_file = open("accuracies.txt", "w+")
    for classfier, accuracy in accuracies.items():
        print("{} : {}%".format(classfier, round(accuracy, 2)))
        accuracy_file.write("{} : {}%\n".format(classfier, round(accuracy, 2)))


def tf_idf(train, test, cuisines):
    corpus_train = train['directions']
    # vectorize_train = TfidfVectorizer(stop_words='english', ngram_range=(2, 2), min_df=2)
    vectorize_train = TfidfVectorizer(stop_words='english')

    tfidf_vect = vectorize_train.fit(corpus_train)
    tfidf_train = tfidf_vect.transform(corpus_train)

    corpus_test = test['directions']
    tfidf_test = tfidf_vect.transform(corpus_test)

    prediction(tfidf_train, tfidf_test, train['cuisine'], cuisines)

    # print_plot(tfidf_train, "UMAP", train['cuisine'], "TF-IDF")
    # print_plot(tfidf_train, "TSNE", train['cuisine'], "TF-IDF")


def bag_of_words(train, test, cuisines):
    vectorize_train = CountVectorizer(stop_words='english', ngram_range=(1, 5))

    corpus_train = train['directions']
    bow_vect = vectorize_train.fit(corpus_train)
    bow_train = bow_vect.transform(corpus_train)

    corpus_test = test['directions']
    bow_test = bow_vect.transform(corpus_test)

    prediction(bow_train, bow_test, train['cuisine'], cuisines)

    # print_plot(bow_train, "UMAP", train['cuisine'], "Bag-Of-Words")
    # print_plot(bow_train, "TSNE", train['cuisine'], "Bag-Of-Words")


def word_2_vec(train, test, cuisines):
    w2vec = get_word2vec(
        MySentences(
            train['directions'],
        ),
        'w2vmodel'
    )
    mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)

    corpus_train = train['directions']
    w2v_train = mean_embedding_vectorizer.fit_transform(corpus_train)

    corpus_test = test['directions']
    w2v_test = mean_embedding_vectorizer.transform(corpus_test)

    prediction(w2v_train, w2v_test, train['cuisine'], cuisines)

    # print_plot(w2v_train, "UMAP", train['cuisine'], "Word-2-Vec")
    # print_plot(w2v_train, "TSNE", train['cuisine'], "Word-2-Vec")


if __name__ == "__main__":
    data_set = pandas.read_json("testing.json")
    data_set_2 = pandas.read_json("text_creole.json")

    data_set.update(data_set_2)
    traindf = shuffle(data_set)

    # traindf['ingredients_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

    new_test = traindf[-538:]
    # new_test = traindf[-512:]
    new_test_without_cuisine = copy.copy(new_test)
    new_test_without_cuisine.pop('cuisine')

    test_cuisines = list()
    for cuisine in new_test['cuisine']:
        test_cuisines.append(cuisine)

    # new_train = traindf[:5000]
    new_train = traindf[:4000]

    input_func = input("Select function: \n"
                       "tf_idf(1) --- bag_of_words(2) --- word_2_vec(3)"
                       "\n#################################################\n")

    if input_func == "tf_idf" or input_func == "1":
        print("`````TF-IDF`````")
        tf_idf(new_train, new_test_without_cuisine, test_cuisines)
    elif input_func == "bag_of_words" or input_func == "2":
        print("`````BAG-OF-WORDS`````")
        bag_of_words(new_train, new_test_without_cuisine, test_cuisines)
    elif input_func == "word_2_vec" or input_func == "3":
        print("`````WORD-2-VEC`````")
        word_2_vec(new_train, new_test_without_cuisine, test_cuisines)
