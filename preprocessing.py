import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

### Nettoyage des données

# Load the dataset
df = pd.read_csv('data/CEAS_08.csv')
print(df.head())

# Remove duplicates
df = df.drop_duplicates()

# Drop rows where critical columns are empty
df = df.dropna(subset=['sender', 'receiver', 'subject', 'body', 'label'])

# Convert 'date' column to datetime, handling mixed time zones and invalid dates
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

# Remove rows where 'date' is NaT (invalid dates)
df = df.dropna(subset=['date'])

# Extract year, month, day of week, and hour from the 'date' column
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour

# Extract domains from email addresses
df['sender_domain'] = df['sender'].str.extract(r'@([\w\.-]+)')
df['receiver_domain'] = df['receiver'].str.extract(r'@([\w\.-]+)')

# Check the first few rows
print(df.head())

# Check dataset info
print(df.info())

# Save the cleaned dataset
#df.to_csv('data/cleaned_CEAS_08.csv', index=False)

# Comparer le nombre de 0 et 1 dans le label
label_counts = df['label'].value_counts()
print(label_counts)

# Calculer la proportion de chaque classe
label_proportions = df['label'].value_counts(normalize=True) * 100
print("\nProportion des classes (%) :")
print(label_proportions)

df['suspicious_sender'] = df['sender'].str.split('@').str[0].str.contains(r'\d', regex=True).astype(int)



### Extraire les URLs de body et créer num_urls et suspicious_urls


# Liste de domaines suspects (a developper)
suspicious_domains = ['bit.ly', 'tinyurl.com', 'goo.gl']

# Extraire les URLs
def extract_urls(text):
    return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

df['urls_list'] = df['body'].apply(extract_urls)
df['num_urls'] = df['urls_list'].apply(len)

# Vérifier les domaines suspects
def check_suspicious_urls(urls):
    for url in urls:
        for domain in suspicious_domains:
            if domain in url:
                return 1
    return 0

df['suspicious_urls'] = df['urls_list'].apply(check_suspicious_urls)
df = df.drop(columns=['urls_list'])  # Supprimer la colonne temporaire


#Calculer la longueur du nom d'utilisateur dans sender.
df['sender_name_length'] = df['sender'].str.split('@').str[0].str.len()

#Définir une liste de mots suspects et compter leur présence dans subject et body.
phishing_keywords = [
    ##### 1. Urgence & Menaces #####
    'urgent', 'immédiat', 'délai', 'dernière chance', 'expir', 'suspendu', 
    'bloqué', 'hacké', 'piraté', 'séquestr', 'confisqu', 'résili', 'litige',
    'restrict', 'violation', 'avertissement', 'pénalité', 'désactiv',
    
    ##### 2. Sécurité & Comptes #####
    'vérif', 'authentif', 'sécur', 'connexion', 'identif', 'mot de passe', 
    'credentials', '2FA', 'MFA', 'OTP', 'code', 'validation', 'protect',
    'lock', 'unlock', 'reactiv', 'recovery', 'reset',
    
    ##### 3. Argent & Paiements #####
    'facture',  '€' , '$', 'paiement', 'rembours', 'virement', 'solde', 'impayé', 
    'prélèvement', 'carte bancaire', 'IBAN', 'SWIFT', 'crypto', 'bitcoin',
    'transaction', 'fraude', 'remise', 'réduct', '100', '99', '1000' , 'million', 'prix',
    
    ##### 4. Récompenses & Offres #####
    'gratuit', 'cadeau', 'lotterie', 'prime', 'bonus', 'exclusif', 
    'limitée', 'spécial', 'promo', 'rabais', 'coupon', 'gagnant', 
    'million', 'euro', 'dollar', 'cash',
    
    ##### 5. Faux Problèmes Techniques #####
    'erreur', 'bug', 'corrompu', 'mise à jour', 'maintenance', 'crash',
    'panne', 'sauvegard', 'restaur', 'version', 'patch', 'correctif',
    
    ##### 6. Social Engineering #####
    'cliquez', 'télécharg', 'ouvrir', 'confirmer', 'mettre à jour',
    'connectez', 'vérifiez', 'répondre', 'contactez', 'support', 'aide',
    'assistance', 'service client',
    
    ##### 7. Typosquatting & Caractères Spéciaux #####
    r'[àéèçù]',  # Caractères accentués suspects
    r'\d+€',     # Montants monétaires
    r'[\!\?\*]{2,}',  # Ponctuations multiples
    r'\b(?:paypal|amazon|google|microsoft)\W+\w+',  # Marques + caractères spéciaux
    
    ##### 8. Patterns Multilingues #####
    # Anglais
    'verify', 'account', 'password', 'login', 'security', 'limited',
    'offer', 'click', 'banking', 'alert',
    
    # Espagnol
    'urgente', 'cuenta', 'contraseña', 'banco', 'seguridad',
    
    # Allemand
    'dringend', 'konto', 'passwort', 'überprüfen'
]

def count_phishing_words(text, keywords):
    text = text.lower()
    return sum(text.count(word) for word in keywords)


df['phishing_words'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['phishing_words'] = df['phishing_words'].apply(lambda x: count_phishing_words(x, phishing_keywords))

#Calculer la proportion de caractères spéciaux dans body.
special_chars = r'[!@#$%^&*(),.?":{}|<>]'

def special_char_density(text):
    if not text or len(text) == 0:
        return 0
    special_count = len(re.findall(special_chars, text))
    return special_count / len(text)

df['special_char_density'] = df['body'].apply(special_char_density)

# Check the first few rows
print(df.head())

# Check dataset info
print(df.info())

### Démarrer le prétraitement NLP

# Télécharger les ressources nécessaires
nltk.download('punkt')        # Déjà téléchargé, mais conservé pour la complétude
nltk.download('punkt_tab')    # Ajout de la ressource manquante
nltk.download('stopwords')    # Déjà téléchargé
nltk.download('wordnet')      # Déjà téléchargé

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Mise en minuscules
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprimer caractères spéciaux
    text = re.sub(r'\s+', ' ', text).strip()  # Enlève les espaces multiples
    tokens = word_tokenize(text)   # Tokenisation 
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

df['subject_clean'] = df['subject'].apply(preprocess_text)
df['body_clean'] = df['body'].apply(preprocess_text)


### Encodage et Vectorisation

le_sender = LabelEncoder()
le_receiver = LabelEncoder()

df['sender_domain_encoded'] = le_sender.fit_transform(df['sender_domain'].astype(str))
df['receiver_domain_encoded'] = le_receiver.fit_transform(df['receiver_domain'].astype(str))

# Vectoriser subject_clean et body_clean
tfidf_subject = TfidfVectorizer(max_features=1000)
tfidf_body = TfidfVectorizer(max_features=1000)

subject_vecs = tfidf_subject.fit_transform(df['subject_clean'])
body_vecs = tfidf_body.fit_transform(df['body_clean'])

joblib.dump(tfidf_subject, 'models/tfidf_subject.pkl')
joblib.dump(tfidf_body, 'models/tfidf_body.pkl')


### Assemblage des Features

# Liste des colonnes numériques utiles
numeric_features = [
    'num_urls', 
    'suspicious_urls', 
    'sender_name_length', 
    'phishing_words', 
    'special_char_density',
    'sender_domain_encoded', 
    'receiver_domain_encoded'
]

# Convertir les features numériques en matrice (2D)
numeric_matrix = df[numeric_features].to_numpy()




numeric_matrix = df[numeric_features].to_numpy()


# Combiner toutes les features (text subject + text body + numériques)
X = hstack([subject_vecs, body_vecs, numeric_matrix])

# Extraire les labels
y = df['label'].to_numpy()

#Sauvegarde du Dataset
final_df = df[numeric_features + ['day_of_week','tfidf_subject', 'tfidf_body', 'label']]
final_df.to_csv('data/cleaned_phishing_data.csv', index=False)
print(final_df.head())

joblib.dump(le_sender, 'models/le_sender.pkl')
joblib.dump(le_receiver, 'models/le_receiver.pkl')

joblib.dump(tfidf_subject, 'models/tfidf_subject.pkl')
joblib.dump(tfidf_body, 'models/tfidf_body.pkl')
joblib.dump(X, 'models/features_X.pkl')
joblib.dump(y, 'models/labels_y.pkl')
