# Detection of Phishing Attacks with AI and NLP

## Description
Ce projet vise à développer un modèle d'IA utilisant le traitement du langage naturel (NLP) et des techniques de machine learning pour détecter et classer automatiquement les emails de phishing. Le système analyse divers aspects des emails (contenu, métadonnées, URLs, etc.) pour identifier les messages malveillants.

## Structure du Projet
```
Detection-of-Phishing-Attacks-with-AI-and-NLP/
│── data/                      # Dossier contenant les jeux de données
│   ├── CEAS_08.csv                # Dataset original
│   └── cleaned_phishing_data.csv  # Dataset nettoyé et prétraité
│── models/                    # Dossier contenant les différents modèles entraînés
│── Presentation.pdf
│── README.md                  # Documentation du projet
│── Rapport.pdf
├── api.py                     # Script pour interagir avec l'API Gmail
├── app.py                     # Script pour l'interface de l'application
├── models.py                  # Implémentation des modèles ML
│__ preprocessing.py           # Prétraitement des données
```


## Utilisation
### 1. Prétraitement des données
Assurez-vous que le fichier de données est bien placé dans `data/CEAS_08.csv `.
```bash
python preprocessing.py
```

### 2. Entraînement des modèles
```bash
python models.py
```

### 3. Analyse des emails (via API Gmail)
```bash
python api.py
```

## Modèles entraînés
- **XGBoost**
- **SVM (Support Vector Machine)**
- **Random Forest**


## Résultats
Les performances des modèles sont affichées sous forme de tableau dans la sortie console, avec un résumé indiquant le modèle le plus performant.

Les visualisations incluent :
- **Matrices de confusion pour chaque modèle**
- **Courbes ROC comparatives**
