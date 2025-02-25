import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from preprocessing import load_and_preprocess_data
from utils import plot_learning_curve

def train_decision_tree(train_X, test_X, train_y, test_y, max_depth=15):
    """
    Entraîne un Decision Tree sur les données fournies et trace les courbes AUC vs Profondeur.

    Paramètres :
    ------------
    train_X, test_X : sparse matrix
        Matrices TF-IDF des données d'entraînement et de test.
    train_y, test_y : Series
        Labels des données d'entraînement et de test.
    max_depth : int
        Profondeur maximale à tester pour le Decision Tree.

    Retourne :
    ----------
    best_clf : DecisionTreeClassifier
        Meilleur modèle après élagage.
    """

    print("Entraînement du Decision Tree Classifier...")

    # Initialisation des métriques
    tree_auc_train, tree_auc_test = np.zeros(max_depth), np.zeros(max_depth)
    training_time, prediction_time = np.zeros(max_depth), np.zeros(max_depth)

    for i in range(1, max_depth):
        clf_decision_tree = DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=1)

        # Temps d'entraînement
        t0 = time.perf_counter()
        clf_decision_tree.fit(train_X, train_y)
        training_time[i] = round(time.perf_counter() - t0, 3)

        # Calcul de l'AUC
        tree_auc_train[i] = roc_auc_score(train_y, clf_decision_tree.predict_proba(train_X)[:, 1])
        t1 = time.perf_counter()
        tree_auc_test[i] = roc_auc_score(test_y, clf_decision_tree.predict_proba(test_X)[:, 1])
        prediction_time[i] = round(time.perf_counter() - t1, 3)

    # Tracé AUC vs Profondeur
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_depth), tree_auc_train[1:], linewidth=2, label="AUC d'entraînement")
    plt.plot(range(1, max_depth), tree_auc_test[1:], linewidth=2, label="AUC de test")
    plt.legend()
    plt.ylim(0.8, 1.0)
    plt.xlabel("Profondeur maximale")
    plt.ylabel("AUC")
    plt.title("AUC vs. Profondeur pour Decision Tree")
    plt.savefig('Phishing_treedepth_auc.png')
    plt.show()

    print(f"Meilleure profondeur d'entraînement : {np.argmax(tree_auc_train)}")
    print(f"Meilleur score AUC d'entraînement : {np.max(tree_auc_train):.3f}")
    print(f"Meilleure profondeur de test : {np.argmax(tree_auc_test)}")
    print(f"Meilleur score AUC de test : {np.max(tree_auc_test):.3f}")

    # Entraînement initial du Decision Tree
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(train_X, train_y)

    # AUC sans élagage
    auc_train = roc_auc_score(train_y, clf.predict_proba(train_X)[:, 1])
    auc_test = roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1])

    print(f"AUC d'entraînement (sans élagage) : {auc_train:.3f}")
    print(f"AUC de test (sans élagage) : {auc_test:.3f}")

    # Élagage de l'arbre
    path = clf.cost_complexity_pruning_path(train_X, train_y)
    ccp_alphas = path.ccp_alphas[:-1]  # Suppression du dernier alpha qui est trivial
    clfs = [DecisionTreeClassifier(random_state=1, ccp_alpha=alpha).fit(train_X, train_y) for alpha in ccp_alphas]

    # Calcul des AUC pour chaque arbre élagué
    train_scores = [roc_auc_score(train_y, clf.predict_proba(train_X)[:, 1]) for clf in clfs]
    test_scores = [roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1]) for clf in clfs]

    # Tracé des AUC vs ccp_alpha
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, train_scores, marker='o', label="AUC d'entraînement", drawstyle="steps-post")
    plt.plot(ccp_alphas, test_scores, marker='o', label="AUC de test", drawstyle="steps-post")
    plt.xlabel("ccp_alpha")
    plt.ylabel("AUC")
    plt.title("AUC vs. ccp_alpha pour l'élagage du Decision Tree")
    plt.legend()
    plt.grid()
    plt.savefig('Phishing_tree_pruning_auc.png')
    plt.show()

    # Sélection du meilleur modèle
    best_index = np.argmax(test_scores)
    best_clf = clfs[best_index]

    print(f"Meilleur ccp_alpha : {ccp_alphas[best_index]:.5f}")
    print(f"AUC d'entraînement (élagage optimal) : {train_scores[best_index]:.3f}")
    print(f"AUC de test (élagage optimal) : {test_scores[best_index]:.3f}")

    return best_clf


if __name__ == "__main__":
    train_X, test_X, train_y, test_y = load_and_preprocess_data("data/fraud_email.csv")
    best_clf = train_decision_tree(train_X, test_X, train_y, test_y, max_depth=15)

    # Tracé de la courbe d'apprentissage
    title = "Phishing Dataset Learning Curves (Decision Tree)"
    plot_learning_curve(best_clf, title, train_X, train_y, cv=None, n_jobs=-1)
    plt.show()
