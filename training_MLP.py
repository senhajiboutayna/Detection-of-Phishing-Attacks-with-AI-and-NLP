from preprocessing import load_and_preprocess_data
from utils import plot_learning_curve
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def train_neural_network(train_X, test_X, train_y, test_y, learning_rates=[0.1, 1, 2, 3]):
    """
    Entraîne un réseau de neurones MLPClassifier avec différentes valeurs de taux d'apprentissage.
    
    :param train_X: Features d'entraînement
    :param test_X: Features de test
    :param train_y: Labels d'entraînement
    :param test_y: Labels de test
    :param learning_rates: Liste des taux d'apprentissage à tester
    """
    print("Training Neural Network...")

    training_time, prediction_time = [], []
    nn_auc_train, nn_auc_test = [], []

    for rate in learning_rates:
        clf_nn = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=rate, random_state=1, max_iter=100, verbose=1, warm_start=True)

        # Mesure du temps d'entraînement
        t0 = time.perf_counter()
        clf_nn.fit(train_X, train_y)
        training_time.append(round(time.perf_counter() - t0, 3))

        # Calcul de l'AUC pour l'ensemble d'entraînement
        nn_auc_train.append(roc_auc_score(train_y, clf_nn.predict_proba(train_X)[:, 1]))

        # Mesure du temps de prédiction
        t1 = time.perf_counter()
        nn_auc_test.append(roc_auc_score(test_y, clf_nn.predict_proba(test_X)[:, 1]))
        prediction_time.append(round(time.perf_counter() - t1, 3))

    print("Training Completed.")

    # Affichage des courbes d'apprentissage
    X, y = train_X, train_y
    title = "Phishing Dataset Learning Curves (Neural Network)"
    estimator = MLPClassifier(learning_rate_init=0.1, random_state=1)
    cv = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
    plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=-1)

    plt.show()

# Chargement des données et entraînement du modèle
train_X, test_X, train_y, test_y = load_and_preprocess_data("data/fraud_email.csv")
train_neural_network(train_X, test_X, train_y, test_y)
