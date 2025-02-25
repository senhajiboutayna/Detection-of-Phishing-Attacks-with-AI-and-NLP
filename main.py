from preprocessing import load_and_preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from training_decision_tree import train_decision_tree
from training_adaboost import train_adaboost
from training_MLP import train_neural_network
from training_KNN import train_knn
from training_SVM import train_svm

filepath = "data/fraud_email.csv"  # Modifie le chemin si nécessaire
X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

# Fonction pour entraîner et évaluer les modèles
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    best_model = None
    best_score = 0
    results = {}

    for name, train_func in models.items():
        print(f"\n--- Training {name} ---")
        
        try:
            model = train_func(X_train, y_train)  # Entraînement du modèle
            y_pred = model.predict(X_test)  # Prédiction sur le jeu de test

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="binary")
            recall = recall_score(y_test, y_pred, average="binary")
            f1 = f1_score(y_test, y_pred, average="binary")

            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print("-" * 40)

            # Sélection du modèle avec le meilleur F1-score
            if f1 > best_score:
                best_score = f1
                best_model = name

        except Exception as e:
            print(f"Erreur lors de l'entraînement de {name} : {e}")

    return best_model, best_score, results

# Liste des modèles
models = {
    "Decision Tree": train_decision_tree,
    "AdaBoost": train_adaboost,
    "MLP": train_neural_network,
    "KNN": train_knn,
    "SVM": train_svm
}

# Exécution
best_model, best_score, results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

print(f"\nLe meilleur modèle est : {best_model} avec un F1-score de {best_score:.4f}")
