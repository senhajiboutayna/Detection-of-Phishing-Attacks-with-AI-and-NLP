# Detection of Phishing Attacks with AI and NLP

## Description
Ce projet implémente plusieurs modèles d'apprentissage automatique pour détecter les attaques de phishing dans les e-mails. Il inclut la prétraitement des données, l'entraînement de divers modèles et l'évaluation des performances.

## Structure du Projet
```
Detection-of-Phishing-Attacks-with-AI-and-NLP/
│── data/                      # Dossier contenant les jeux de données
│── results/                   # Dossier contenant les figures et les modèles entraînés
│── preprocessing.py           # Prétraitement des données
│── utils.py                   # Fonctions utilitaires
│── training_decision_tree.py  # Entraînement du modèle Decision Tree
│── training_adaboost.py       # Entraînement du modèle AdaBoost
│── training_MLP.py            # Entraînement du modèle MLP (Neural Network)
│── training_KNN.py            # Entraînement du modèle KNN
│── training_SVM.py            # Entraînement du modèle SVM
│── main.py                    # Comparaison des modèles et sélection du meilleur
│__ README.md                  # Documentation du projet
```

## Installation
1. Cloner ce dépôt :
   ```bash
   git clone https://github.com/senhajiboutayna/Detection-of-Phishing-Attacks-with-AI-and-NLP.git
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
### 1. Prétraitement des données
Assurez-vous que le fichier de données est bien placé dans `data/fraud_email.csv`.

### 2. Entraînement des modèles
Chaque script `training_*.py` entraîne un modèle différent. Par exemple, pour entraîner le modèle Decision Tree :
```bash
python training_decision_tree.py
```

### 3. Comparaison des modèles
Exécutez `main.py` pour comparer les performances et sélectionner le meilleur modèle :
```bash
python main.py
```

## Modèles entraînés
- **Decision Tree**
- **AdaBoost**
- **MLP (Neural Network)**
- **KNN (K-Nearest Neighbors)**
- **SVM (Support Vector Machine)**

## Évaluation des performances
Le script `main.py` compare les modèles en utilisant plusieurs métriques :
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Le modèle avec le meilleur **F1-score** est sélectionné comme le meilleur modèle.

## Résultats
Les performances des modèles sont affichées sous forme de tableau dans la sortie console, avec un résumé indiquant le modèle le plus performant.
