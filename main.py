import copy
import os
import gensim
import matplotlib.pyplot as plt
import nltk
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from umap import UMAP

from MyTokenizer import MeanEmbeddingVectorizer

PARAM_TRIALS = "param_trials.txt"


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
        umap = UMAP(
            n_neighbors=5,
            min_dist=0.3,
            metric='correlation')
        reduced_data = umap.fit_transform(data)
    elif reducer == "TSNE":
        tsne = TSNE(
            perplexity=50,
            n_components=2,
            init='pca',
            n_iter=5000,
            random_state=0,
            verbose=1
        )

        # if extractor == "Word-2-Vec":
        reduced_data = tsne.fit_transform(data)
        # else:
        #     reduced_data = tsne.fit_transform(data.toarray())

    targets = [i for i in range(0, len(target))]

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=targets, s=5, cmap='rainbow')
    plt.suptitle(reducer + " " + extractor)
    plt.show()


def prediction(train, test, classes, test_cuisines, extractor):
    accuracies = dict()

    if extractor == "TF-IDF":
        # LinearSVC
        svc_c = 0.5

        # LOGISTIC REGRESSION
        regression_c = 10

        # RANDOM FORREST
        min_samples_split = 5
        n_estimators = 1400
        max_depth = 80
        max_features = 'sqrt'
        bootstrap = False
        min_samples_leaf = 1
    elif extractor == "BOW":
        svc_c = 0.1

        # LOGISTIC REGRESSION
        regression_c = 1

        # RANDOM FORREST
        min_samples_split = 2
        n_estimators = 1400
        max_depth = 40
        max_features = 'auto'
        bootstrap = False
        min_samples_leaf = 1
    else:
        svc_c = 25

        # LOGISTIC REGRESSION
        regression_c = 1000

        # RANDOM FORREST
        min_samples_split = 2
        n_estimators = 1000
        max_depth = 50
        max_features = 'auto'
        bootstrap = False
        min_samples_leaf = 1

    classifier_svc = LinearSVC(C=svc_c)
    classifier_svc.fit(train, classes)
    prediction_svc = classifier_svc.predict(test)

    accuracies['Linear_SVC'] = compute_accuracy(prediction_svc, test_cuisines)

    classifier_regr = LogisticRegression(C=regression_c)
    classifier_regr.fit(train, classes)
    prediction_regr = classifier_regr.predict(test)

    accuracies['Logistic_Regression'] = compute_accuracy(prediction_regr, test_cuisines)

    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    # rf = RandomForestClassifier()
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
    #                                random_state=42, n_jobs=-1)
    # rf_random.fit(train, classes)
    # print(rf_random.best_params_)

    classfier_rf = RandomForestClassifier(min_samples_split=min_samples_split,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          max_features=max_features,
                                          bootstrap=bootstrap,
                                          min_samples_leaf=min_samples_leaf)

    classfier_rf.fit(train, classes)
    prediction_rf = classfier_rf.predict(test)

    # tree_in_forest = classfier_rf.estimators_[5]
    #
    # export_graphviz(tree_in_forest,
    #                 out_file='tree.dot',
    #                 feature_names=features,
    #                 class_names=classes,
    #                 filled=True,
    #                 rounded=True)
    #
    # os.system('dot -Tpng tree.dot -o tree.png')

    accuracies['Random Forrest'] = compute_accuracy(prediction_rf, test_cuisines)

    accuracy_file = open("accuracies.txt", "w+")
    for classfier, accuracy in accuracies.items():
        print("{} : {}%".format(classfier, round(accuracy, 2)))
        accuracy_file.write("{} : {}%\n".format(classfier, round(accuracy, 2)))
    accuracy_file.write("\n")


def tf_idf(train, test, cuisines):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), min_df=4)

    corpus_train = train['directions']
    tfidf_vect = vectorizer.fit(corpus_train)
    tfidf_train = tfidf_vect.transform(corpus_train)

    corpus_test = test['directions']
    tfidf_test = tfidf_vect.transform(corpus_test)

    prediction(tfidf_train.toarray(), tfidf_test.toarray(), train['cuisine'], cuisines, "TF-IDF")

    print_plot(tfidf_train.toarray(), "UMAP", train['cuisine'], "TF-IDF")
    print_plot(tfidf_train.toarray(), "TSNE", train['cuisine'], "TF-IDF")


def bag_of_words(train, test, cuisines):

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5)

    corpus_train = train['directions']
    bow_vect = vectorizer.fit(corpus_train)
    bow_train = bow_vect.transform(corpus_train)

    corpus_test = test['directions']
    bow_test = bow_vect.transform(corpus_test)

    prediction(bow_train, bow_test, train['cuisine'], cuisines, "BOW")

    print_plot(bow_train, "UMAP", train['cuisine'], "Bag-Of-Words")
    print_plot(bow_train, "TSNE", train['cuisine'], "Bag-Of-Words")


def generate_sentences(recipe):
    for array in recipe:
        for document in array:
            for sent in nltk.sent_tokenize(document):
                yield nltk.word_tokenize(sent)


def word_2_vec(train, test, cuisines):
    location = 'w2vmodel'

    if os.path.exists(location):
        w2vec = gensim.models.Word2Vec.load(location)
    else:
        w2vec = gensim.models.Word2Vec(generate_sentences(train['directions']), size=250, window=5, min_count=5, workers=8)
        w2vec.save(location)

    # w2vec = get_word2vec(
    #     MySentences(
    #         train['directions'],
    #     ),
    #     'w2vmodel'
    # )

    mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)

    corpus_train = train['directions']
    w2v_train = mean_embedding_vectorizer.fit_transform(corpus_train)

    corpus_test = test['directions']
    w2v_test = mean_embedding_vectorizer.transform(corpus_test)

    prediction(w2v_train, w2v_test, train['cuisine'], cuisines, "W2V")

    print_plot(w2v_train, "UMAP", train['cuisine'], "Word-2-Vec")
    print_plot(w2v_train, "TSNE", train['cuisine'], "Word-2-Vec")


if __name__ == "__main__":
    data_set = pandas.read_json("dataset.json")

    traindf = shuffle(data_set)

    # traindf['ingredients_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

    new_test = traindf[-474:]

    new_test_without_cuisine = copy.copy(new_test)
    new_test_without_cuisine.pop('cuisine')

    test_cuisines = list()
    for cuisine in new_test['cuisine']:
        test_cuisines.append(cuisine)

    new_train = traindf[:4500]

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
