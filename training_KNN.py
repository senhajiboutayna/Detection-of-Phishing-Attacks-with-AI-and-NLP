from preprocessing import load_and_preprocess_data
from utils import plot_learning_curve
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

def train_knn(train_X, test_X, train_y, test_y, max_k=5):
    """
    Entraîne un modèle K-Nearest Neighbors (KNN) avec différentes valeurs de K.
    
    :param train_X: Features d'entraînement
    :param test_X: Features de test
    :param train_y: Labels d'entraînement
    :param test_y: Labels de test
    :param max_k: Nombre maximal de voisins à tester
    """
    print("Training KNN...")

    # Initialisation des dictionnaires de stockage des résultats
    results = {
        "knn_auc_train": np.zeros(max_k),
        "knn_auc_test": np.zeros(max_k),
        "training_time": np.zeros(max_k),
        "prediction_time": np.zeros(max_k)
    }

    for k in range(1, max_k + 1):
        clf_knn = KNeighborsClassifier(
            n_neighbors=k,
            algorithm='auto',
            leaf_size=30,
            metric='minkowski',
            p=2,
            weights='uniform',
            n_jobs=-1
        )

        # Mesure du temps d'entraînement
        t0 = time.perf_counter()
        clf_knn.fit(train_X, train_y)
        results["training_time"][k-1] = round(time.perf_counter() - t0, 3)

        # Prédictions et calcul de l'AUC pour l'ensemble d'entraînement
        t1 = time.perf_counter()
        pred_train = clf_knn.predict_proba(train_X)[:, 1]
        results["knn_auc_train"][k-1] = roc_auc_score(train_y, pred_train)

        # Prédictions et calcul de l'AUC pour l'ensemble de test
        t2 = time.perf_counter()
        pred_test = clf_knn.predict_proba(test_X)[:, 1]
        results["prediction_time"][k-1] = round(time.perf_counter() - t2, 3)
        results["knn_auc_test"][k-1] = roc_auc_score(test_y, pred_test)

    # Affichage des courbes AUC
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_k + 1), results["knn_auc_train"], linewidth=3, label="KNN train AUC")
    plt.plot(range(1, max_k + 1), results["knn_auc_test"], linewidth=3, label="KNN test AUC")
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.xlabel("K Nearest Neighbors - Euclidean")
    plt.ylabel("Validation AUC")
    plt.title("Phishing Dataset K-NN - AUC vs. Number of Neighbors K")
    plt.savefig('churn_knn_fig6')
    plt.show()

    # Affichage des courbes de temps d'exécution
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_k + 1), results["training_time"], linewidth=3, label="KNN training time")
    plt.plot(range(1, max_k + 1), results["prediction_time"], linewidth=3, label="KNN prediction time")
    plt.legend()
    plt.xlabel("K Nearest Neighbors - Euclidean")
    plt.ylabel("Time (sec)")
    plt.title("Phishing Dataset K-NN - Time vs. Number of Neighbors K")
    plt.savefig('Phishing_Knn_time_fig')
    plt.show()

    # Affichage des meilleurs résultats
    print(f"Best number of neighbors training: {np.argmax(results['knn_auc_train']) + 1}")
    print(f"Highest AUC score training: {np.max(results['knn_auc_train'])}")
    print(f"Best number of neighbors testing: {np.argmax(results['knn_auc_test']) + 1}")
    print(f"Highest AUC score testing: {np.max(results['knn_auc_test'])}")

    print("OK")

    # Courbes d'apprentissage
    X, y = train_X, train_y
    title = "Phishing Dataset Learning Curves (K-NN)"
    estimator = KNeighborsClassifier(n_neighbors=1, algorithm='auto', leaf_size=30, metric='minkowski',
                                     n_jobs=-1, p=2, weights='uniform')
    plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)
    plt.show()

# Chargement des données et entraînement du modèle
train_X, test_X, train_y, test_y = load_and_preprocess_data("data/fraud_email.csv")
train_knn(train_X, test_X, train_y, test_y)
