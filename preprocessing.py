import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_and_preprocess_data(filepath):
    """
    Charge et prétraite les données textuelles depuis un fichier CSV.

    Paramètres :
    ------------
    filepath : str
        Chemin du fichier CSV contenant les données.

    Retourne :
    ----------
    train_X : sparse matrix
        Matrice TF-IDF des données d'entraînement.
    test_X : sparse matrix
        Matrice TF-IDF des données de test.
    train_y : Series
        Labels des données d'entraînement.
    test_y : Series
        Labels des données de test.
    """

    # Chargement des données depuis un fichier CSV
    data_df = pd.read_csv("data/fraud_email.csv")

    # Suppression des lignes avec des valeurs manquantes
    data_df = data_df.dropna()

    # Nettoyage du texte : mise en minuscule et suppression des caractères spéciaux
    data_df['Text'] = data_df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data_df['Text'] = data_df['Text'].str.replace('[^\w\s]','')

    # Suppression des stopwords
    stop = stopwords.words('english')
    data_df['Text'] = data_df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stop, norm='l2', decode_error='ignore', binary=True)
    features = vectorizer.fit_transform(data_df["Text"])

    # Transformation des textes en features TF-IDF
    features = vectorizer.fit_transform(data_df["Text"])

    # Affichage de la forme des features
    print(features.shape)

    # Division des données en ensembles d'entraînement et de test
    train_X, test_X, train_y, test_y = train_test_split(features, data_df["Class"], test_size=0.2, stratify=data_df["Class"])

    data_df.head()

    data_df.info()

    data_df.describe()

    return train_X, test_X, train_y, test_y

