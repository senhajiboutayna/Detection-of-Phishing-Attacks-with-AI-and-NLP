import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from preprocessing import load_and_preprocess_data
from utils import plot_learning_curve

def train_adaboost(train_X, test_X, train_y, test_y, max_depth=15, n_estimators=10):
    """
    Entraîne un modèle AdaBoost avec des arbres de décision faibles et trace les courbes AUC et temps vs profondeur.

    Paramètres :
    -------------
    train_X, test_X : sparse matrix
        Matrices TF-IDF des données d'entraînement et de test.
    train_y, test_y : Series
        Labels des données d'entraînement et de test.
    max_depth : int
        Profondeur maximale à tester pour les weak learners.
    n_estimators : int
        Nombre d'estimateurs dans AdaBoost.
    """
    print("Entraînement d'AdaBoost...")

    adaboost_auc_train, adaboost_auc_test = np.zeros(max_depth), np.zeros(max_depth)
    training_time, prediction_time = np.zeros(max_depth), np.zeros(max_depth)

    for i in range(1, max_depth):
        clf_adaboost = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=1),
            n_estimators=n_estimators,
            random_state=1
        )
        
        # Temps d'entraînement
        t0 = time.perf_counter()
        clf_adaboost.fit(train_X, train_y)
        training_time[i] = round(time.perf_counter() - t0, 3)
        
        # AUC d'entraînement et de test
        adaboost_auc_train[i] = roc_auc_score(train_y, clf_adaboost.predict_proba(train_X)[:, 1])
        t1 = time.perf_counter()
        adaboost_auc_test[i] = roc_auc_score(test_y, clf_adaboost.predict_proba(test_X)[:, 1])
        prediction_time[i] = round(time.perf_counter() - t1, 3)

    # Tracé AUC vs Profondeur
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_depth), adaboost_auc_train[1:], linewidth=3, label="AUC d'entraînement")
    plt.plot(range(1, max_depth), adaboost_auc_test[1:], linewidth=3, label="AUC de test")
    plt.legend()
    plt.ylim(0.9, 1.0)
    plt.xlabel("Profondeur des weak learners")
    plt.ylabel("Validation AUC")
    plt.title("Phishing Dataset Adaboost - AUC vs. Weak Learner Depth")
    plt.savefig('Phishing_boosting_auc.png')
    plt.show()

    # Tracé Temps vs Profondeur
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_depth), training_time[1:], linewidth=3, label="Temps d'entraînement")
    plt.plot(range(1, max_depth), prediction_time[1:], linewidth=3, label="Temps de prédiction")
    plt.legend()
    plt.xlabel("Profondeur des weak learners")
    plt.ylabel("Temps (sec)")
    plt.title("Phishing Dataset Adaboost - Temps vs. Weak Learner Depth")
    plt.savefig('Phishing_boosting_time.png')
    plt.show()

    print(f"Meilleure profondeur pour l'entraînement : {np.argmax(adaboost_auc_train)}")
    print(f"Meilleur score AUC d'entraînement : {np.max(adaboost_auc_train):.3f}")
    print(f"Meilleure profondeur pour le test : {np.argmax(adaboost_auc_test)}")
    print(f"Meilleur score AUC de test : {np.max(adaboost_auc_test):.3f}")

    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=np.argmax(adaboost_auc_test), criterion='entropy', random_state=1),
        n_estimators=n_estimators,
        random_state=1
    )

if __name__ == "__main__":
    train_X, test_X, train_y, test_y = load_and_preprocess_data("data/fraud_email.csv")
    best_adaboost = train_adaboost(train_X, test_X, train_y, test_y, max_depth=15)

    # Tracé des courbes d'apprentissage
    title = "Phishing Dataset Learning Curves (Adaboost)"
    plot_learning_curve(best_adaboost, title, train_X, train_y, cv=None, n_jobs=-1)
    plt.show()
