from preprocessing import load_and_preprocess_data
from utils import plot_learning_curve
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

def train_svm(train_X, test_X, train_y, test_y, kernels=["linear", "poly", "rbf"]):
    """
    Entraîne un modèle SVM avec plusieurs noyaux et évalue les performances.
    
    :param train_X: Features d'entraînement
    :param test_X: Features de test
    :param train_y: Labels d'entraînement
    :param test_y: Labels de test
    :param kernels: Liste des noyaux à tester (par défaut ["linear", "poly", "rbf"])
    """
    print("Training SVM...")

    results = {}

    for kernel in kernels:
        print(f"\n--- Training SVM with {kernel} kernel ---")
        clf_svm = SVC(kernel=kernel, probability=True, random_state=1)

        # Mesure du temps d'entraînement
        t0 = time.perf_counter()
        clf_svm.fit(train_X, train_y)
        training_time = round(time.perf_counter() - t0, 3)

        # Prédictions et calcul des scores AUC
        pred_train = clf_svm.predict_proba(train_X)[:, 1]
        t1 = time.perf_counter()
        pred_test = clf_svm.predict_proba(test_X)[:, 1]
        prediction_time = round(time.perf_counter() - t1, 3)

        auc_train = roc_auc_score(train_y, pred_train)
        auc_test = roc_auc_score(test_y, pred_test)

        results[kernel] = {
            "auc_train": auc_train,
            "auc_test": auc_test,
            "training_time": training_time,
            "prediction_time": prediction_time
        }

        # Affichage des résultats
        print(f"{kernel.capitalize()} kernel training AUC: {auc_train:.4f}")
        print(f"{kernel.capitalize()} kernel testing AUC: {auc_test:.4f}")
        print(f"{kernel.capitalize()} kernel training time: {training_time} sec")
        print(f"{kernel.capitalize()} kernel prediction time: {prediction_time} sec")

    # Comparaison des résultats sous forme de graphique
    plt.figure(figsize=(10, 5))
    for kernel in kernels:
        plt.bar(kernel, results[kernel]["auc_test"], label=f"{kernel} kernel")

    plt.ylim(0.5, 1.0)
    plt.ylabel("AUC Score")
    plt.title("SVM AUC Score Comparison")
    plt.legend()
    plt.savefig("SVM_AUC_Comparison.png")
    plt.show()

    print("\nTraining complete.")

    # Courbes d'apprentissage pour SVM linéaire (exemple)
    X, y = train_X, train_y
    title = "Phishing Dataset Learning Curves (SVM - Linear)"
    estimator = SVC(kernel="linear", probability=True, random_state=1)
    plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1)

    plt.show()

# Chargement des données et exécution
train_X, test_X, train_y, test_y = load_and_preprocess_data("data/fraud_email.csv")
train_svm(train_X, test_X, train_y, test_y)
