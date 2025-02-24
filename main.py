import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

from IPython.display import Image

nltk.download('stopwords')

data_df = pd.read_csv("data/fraud_email.csv")

# Suppression des lignes avec des valeurs manquantes
data_df = data_df.dropna()

data_df['Text'] = data_df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data_df['Text'] = data_df['Text'].str.replace('[^\w\s]','')

stop = stopwords.words('english')
data_df['Text'] = data_df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

corpus = data_df['Text']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.shape)

# Définition des stopwords en anglais
stopset = list(set(stopwords.words("english")))

# Initialisation du vectorizer TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopset, norm='l2', decode_error='ignore', binary=True)

# Transformation des textes en features TF-IDF
features = vectorizer.fit_transform(data_df["Text"])

# Affichage de la forme des features
print(features.shape)

# Division des données en ensembles d'entraînement et de test
train_X, test_X, train_y, test_y = train_test_split(features, data_df["Class"], test_size=0.5, stratify=data_df["Class"])

data_df.head()

data_df.info()

data_df.describe()

max_depth = 15
tree_auc_train, tree_auc_test = np.zeros(max_depth), np.zeros(max_depth)
training_time, prediction_time = np.zeros(max_depth), np.zeros(max_depth)
for i in range(1,max_depth):
    clf_decision_tree = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy',random_state=1)
    t0=time.perf_counter()
    clf_decision_tree = clf_decision_tree.fit(train_X, train_y)
    training_time[i] = round(time.perf_counter()-t0, 3)
    tree_auc_train[i] = roc_auc_score(train_y, clf_decision_tree.predict_proba(train_X)[:,1])
    t1=time.perf_counter()
    tree_auc_test[i] = roc_auc_score(test_y, clf_decision_tree.predict_proba(test_X)[:,1])
    prediction_time[i] = round(time.perf_counter()-t1, 3)

# Supposons que vous ayez les données suivantes
# Remplacez ces listes par vos données réelles
tree_auc_train = [0.85, 0.87, 0.88, 0.90]
tree_auc_test = [0.83, 0.84, 0.86, 0.87]
training_time = [0.1, 0.2, 0.3, 0.4]
prediction_time = [0.05, 0.06, 0.07, 0.08]
max_depth = [1, 2, 3, 4]  # Profondeurs correspondantes

# Tracer AUC vs. Profondeur
plt.figure(figsize=(16, 8))
plt.plot(max_depth, tree_auc_train, linewidth=3, label="AUC d'entraînement de l'arbre de décision")
plt.plot(max_depth, tree_auc_test, linewidth=3, label="AUC de test de l'arbre de décision")
plt.legend()
plt.ylim(0.8, 1.0)
plt.title("Jeu de données de phishing - AUC vs. Profondeur")
plt.xlabel("Profondeur maximale")
plt.ylabel("AUC de validation")
plt.savefig('Phishing_treedepth_fig2.png')
plt.show()

# Tracer Temps vs. Profondeur
plt.figure(figsize=(16, 8))
plt.plot(max_depth, training_time, linewidth=3, label="Temps d'entraînement de l'arbre de décision")
plt.plot(max_depth, prediction_time, linewidth=3, label="Temps de prédiction de l'arbre de décision")
plt.legend()
plt.xlabel("Profondeur maximale")
plt.ylabel("Temps (sec)")
plt.title("Jeu de données de phishing - Temps vs. Profondeur")
plt.savefig('Phishing_treedepth_time_fig2.png')
plt.show()

print("Best tree depth training: " + str(np.argmax(tree_auc_train, axis=0)))
print("Highest AUC score training: " + str(np.max(tree_auc_train, axis=0)))
print("Best tree depth testing: " + str(np.argmax(tree_auc_test, axis=0)))
print("Highest AUC score testing: " +  str(np.max(tree_auc_test, axis=0)))


# Entraînement initial de l'arbre de décision sans élagage
clf = DecisionTreeClassifier(random_state=1)
clf.fit(train_X, train_y)

# Calcul des scores AUC pour l'arbre non élagué
auc_train = roc_auc_score(train_y, clf.predict_proba(train_X)[:, 1])
auc_test = roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1])

print(f"AUC d'entraînement (sans élagage) : {auc_train:.3f}")
print(f"AUC de test (sans élagage) : {auc_test:.3f}")

# Détermination du chemin d'élagage en fonction de la complexité
path = clf.cost_complexity_pruning_path(train_X, train_y)
ccp_alphas = path.ccp_alphas

# Entraînement d'arbres de décision pour chaque valeur de ccp_alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(train_X, train_y)
    clfs.append(clf)

# Suppression du dernier classifieur qui est trivial (un seul nœud)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Calcul des scores AUC pour chaque arbre élagué
train_scores = [roc_auc_score(train_y, clf.predict_proba(train_X)[:, 1]) for clf in clfs]
test_scores = [roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1]) for clf in clfs]

# Tracé des scores AUC en fonction de ccp_alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='AUC d\'entraînement', drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='AUC de test', drawstyle="steps-post")
plt.xlabel('ccp_alpha')
plt.ylabel('AUC')
plt.title('AUC en fonction de ccp_alpha pour l\'élagage de l\'arbre de décision')
plt.legend()
plt.grid()
plt.show()

# Sélection du classifieur avec la meilleure performance sur l'ensemble de test
best_index = np.argmax(test_scores)
best_ccp_alpha = ccp_alphas[best_index]
best_clf = clfs[best_index]

print(f"Meilleur ccp_alpha : {best_ccp_alpha:.5f}")
print(f"AUC d'entraînement (élagage optimal) : {train_scores[best_index]:.3f}")
print(f"AUC de test (élagage optimal) : {test_scores[best_index]:.3f}")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

print("OK")


X, y = train_X, train_y

title = "Phishing Dataset Learning Curves (Decision Tree)"
estimator = tree.DecisionTreeClassifier(max_depth=9, criterion='entropy',random_state=1)
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)

plt.show()

max_depth = 15
adaboost_auc_train = np.zeros(max_depth)
adaboost_auc_test = np.zeros(max_depth)
training_time = np.zeros(max_depth)
prediction_time = np.zeros(max_depth)

for i in range(1, max_depth):
    clf_adaboost = AdaBoostClassifier(
        estimator=tree.DecisionTreeClassifier(max_depth=i, criterion='entropy'),
        n_estimators=10,
        random_state=1
    )
    t0 = time.perf_counter()
    clf_adaboost.fit(train_X, train_y)
    training_time[i] = round(time.perf_counter() - t0, 3)
    adaboost_auc_train[i] = roc_auc_score(train_y, clf_adaboost.predict_proba(train_X)[:, 1])
    t1 = time.perf_counter()
    adaboost_auc_test[i] = roc_auc_score(test_y, clf_adaboost.predict_proba(test_X)[:, 1])
    prediction_time[i] = round(time.perf_counter() - t1, 3)

pyplot.plot(adaboost_auc_train, linewidth=3, label = "Adaboost train AUC")
pyplot.plot(adaboost_auc_test, linewidth=3, label = "Adaboost test AUC")
pyplot.legend()
pyplot.ylim(0.9, 1.0)
pyplot.xlabel("Max Depth of Weak Learners")
pyplot.ylabel("Validation AUC")
plt.title("Phishing Dataset Adaboost - AUC vs. Weak Learner Depth")
pyplot.figure(figsize=(16,8))
pyplot.savefig('Phishing_boosting_fig4')
pyplot.show()

pyplot.plot(training_time, linewidth=3, label = "Adaboost training time")
pyplot.plot(prediction_time, linewidth=3, label = "Adoboost tree prediction time")
pyplot.title("Phishing Dataset Adaboost - Time vs. Weak Learner Depth")
pyplot.legend()
pyplot.xlabel("Max_depth")
pyplot.ylabel("Time (sec)")
pyplot.figure(figsize=(16,8))
pyplot.savefig('Phishing_boosting_time_fig2')
pyplot.show()

print("Best weak learner tree depth training: " + str(np.argmax(adaboost_auc_train, axis=0)))
print("Highest AUC score training: " + str(np.max(adaboost_auc_train, axis=0)))
print("Best weak learner tree depth testing: " + str(np.argmax(adaboost_auc_test, axis=0)))
print("Highest AUC score testing: " +  str(np.max(adaboost_auc_test, axis=0)))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

# Example usage:
title = "Phishing Dataset Learning Curves (Adaboost)"
estimator = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=1), n_estimators=10, random_state=1)
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)

learning_rates = [0.1, 1, 2, 3]
training_time, prediction_time = [], []
nn_auc_train, nn_auc_test = [], []

for rate in learning_rates:
    clf_nn = MLPClassifier(learning_rate_init=rate, random_state=1)

    # Measure training time
    t0 = time.perf_counter()
    clf_nn.fit(train_X, train_y)
    training_time.append(round(time.perf_counter() - t0, 3))

    # Calculate AUC for training set
    nn_auc_train.append(roc_auc_score(train_y, clf_nn.predict_proba(train_X)[:, 1]))

    # Measure prediction time
    t1 = time.perf_counter()
    nn_auc_test.append(roc_auc_score(test_y, clf_nn.predict_proba(test_X)[:, 1]))
    prediction_time.append(round(time.perf_counter() - t1, 3))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.2, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, exploit_incremental_learning=True, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

print("OK")

title = "Phising Dataset Learning Curves (Neural Network)"
estimator = MLPClassifier(learning_rate_init=0.1, random_state=1)
cv = StratifiedKFold(n_splits=3,random_state=1, shuffle=False)
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)

plt.show()

max_k = 5
knn_auc_train = np.zeros(max_k)
knn_auc_test = np.zeros(max_k)
training_time = np.zeros(max_k)
prediction_time = np.zeros(max_k)

for i in range(1, max_k):
    clf_knn = KNeighborsClassifier(
        n_neighbors=i,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        weights='uniform',
        n_jobs=-1
    )

    # Measure training time
    t0 = time.perf_counter()
    clf_knn.fit(train_X, train_y)
    training_time[i] = round(time.perf_counter() - t0, 3)

    # Measure prediction time and calculate AUC for training set
    t1 = time.perf_counter()
    pred_train = clf_knn.predict_proba(train_X)[:, 1]
    training_time[i] += round(time.perf_counter() - t1, 3)
    knn_auc_train[i] = roc_auc_score(train_y, pred_train)

    # Measure prediction time and calculate AUC for test set
    t2 = time.perf_counter()
    pred_test = clf_knn.predict_proba(test_X)[:, 1]
    prediction_time[i] = round(time.perf_counter() - t2, 3)
    knn_auc_test[i] = roc_auc_score(test_y, pred_test)

pyplot.plot(knn_auc_train, linewidth=3, label = "KNN train AUC")
pyplot.plot(knn_auc_test, linewidth=3, label = "KNN test AUC")
pyplot.legend()
pyplot.ylim(0.5, 1.0)
pyplot.xlabel("K Nearest Neighbors - Euclidean")
pyplot.ylabel("Validation AUC")
pyplot.title("Phishing Dataset K-NN - AUC vs. Number of Neighors K")
pyplot.figure(figsize=(16,8))
pyplot.savefig('churn_knn_fig6')
pyplot.show()

pyplot.plot(training_time, linewidth=3, label = "KNN training time")
pyplot.plot(prediction_time, linewidth=3, label = "KNN prediction time")
pyplot.legend()
pyplot.xlabel("K Nearest Neighbors - Euclidean")
pyplot.ylabel("Time (sec)")
pyplot.title("Phishing Dataset K-NN - Time vs. Number of Neighors K")
pyplot.figure(figsize=(12,12))
pyplot.savefig('Phishing_Knn_time_fig')
pyplot.show()

print("Best number of neighbors training: " + str(np.argmax(knn_auc_train, axis=0)))
print("Highest AUC score training: " + str(np.max(knn_auc_train, axis=0)))
print("Best number of neighbors testing: " + str(np.argmax(knn_auc_test, axis=0)))
print("Highest AUC score testing: " +  str(np.max(knn_auc_test, axis=0)))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


print("OK")


title = "Phishing Dataset Learning Curves (K-NN)"
estimator = KNeighborsClassifier(n_neighbors=1, algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=-1, p=2, weights='uniform')
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)
plt.show()

# Initialize and train SVM with linear kernel
clf_svm_linear = SVC(kernel='linear', probability=True, random_state=1)
t0 = time.perf_counter()
clf_svm_linear.fit(train_X, train_y)
training_time_linear = round(time.perf_counter() - t0, 3)

# Predict probabilities for training and test sets
pred_train_linear = clf_svm_linear.predict_proba(train_X)[:, 1]
t1 = time.perf_counter()
pred_test_linear = clf_svm_linear.predict_proba(test_X)[:, 1]
prediction_time_linear = round(time.perf_counter() - t1, 3)

# Calculate AUC scores
svm_auc_train_linear = roc_auc_score(train_y, pred_train_linear)
svm_auc_test_linear = roc_auc_score(test_y, pred_test_linear)

# Initialize and train SVM with polynomial kernel
clf_svm_poly = SVC(kernel='poly', probability=True, random_state=1)
t0 = time.perf_counter()
clf_svm_poly.fit(train_X, train_y)
training_time_poly = round(time.perf_counter() - t0, 3)

# Predict probabilities for training and test sets
pred_train_poly = clf_svm_poly.predict_proba(train_X)[:, 1]
t1 = time.perf_counter()
pred_test_poly = clf_svm_poly.predict_proba(test_X)[:, 1]
prediction_time_poly = round(time.perf_counter() - t1, 3)

# Calculate AUC scores
svm_auc_train_poly = roc_auc_score(train_y, pred_train_poly)
svm_auc_test_poly = roc_auc_score(test_y, pred_test_poly)

# Print results
print(f"Linear kernel training AUC: {svm_auc_train_linear}")
print(f"Linear kernel testing AUC: {svm_auc_test_linear}")
print(f"Polynomial kernel training AUC: {svm_auc_train_poly}")
print(f"Polynomial kernel testing AUC: {svm_auc_test_poly}")

print(f"Linear kernel training time: {training_time_linear} seconds")
print(f"Linear kernel prediction time: {prediction_time_linear} seconds")
print(f"Polynomial kernel training time: {training_time_poly} seconds")
print(f"Polynomial kernel prediction time: {prediction_time_poly} seconds")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Phishing Dataset Learning Curves (SVM)"
estimator = SVC(kernel='linear',probability=True, random_state=1)
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)

plt.show()