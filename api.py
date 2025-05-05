# Importation des bibliothèques nécessaires
import base64  # Pour décoder les contenus encodés en base64 (ex: corps d'email)
import os      # Pour vérifier l’existence de fichiers comme 'token.json'
import re      # Pour la recherche d'expressions régulières (URLs, caractères spéciaux, etc.)
import pandas as pd  # Pour manipuler les données tabulaires (features → DataFrame)
from bs4 import BeautifulSoup  # Pour extraire le texte à partir de contenu HTML
from google.auth.transport.requests import Request  # Pour rafraîchir les jetons expirés
from google.oauth2.credentials import Credentials  # Pour gérer les credentials OAuth
from google_auth_oauthlib.flow import InstalledAppFlow  # Pour gérer le processus d'authentification OAuth2
from googleapiclient.discovery import build  # Pour interagir avec l’API Gmail
import joblib  # Pour charger des modèles ou objets (LabelEncoders, modèles ML...)

# Portée d’accès : ici lecture seule sur la boîte mail
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Authentification à Gmail avec OAuth2
def authenticate_gmail():
    creds = None  # Initialisation des credentials
    # Vérifie si un jeton existe déjà (connexion précédente)
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # Si pas de jeton ou jeton invalide
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Rafraîchit le jeton s’il est expiré
            creds.refresh(Request())
        else:
            # Sinon, lance le processus d’authentification utilisateur
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Sauvegarde du nouveau jeton
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    # Retourne le service Gmail prêt à l’usage
    return build('gmail', 'v1', credentials=creds)

# Extraction du corps d'un email (texte brut ou HTML converti)
def get_email_body(message_payload):
    parts = message_payload.get('parts', [])
    body = ""
    for part in parts:
        mime_type = part.get('mimeType')
        data = part.get('body', {}).get('data')
        # Si texte brut
        if mime_type == 'text/plain' and data:
            body = base64.urlsafe_b64decode(data).decode('utf-8')
        # Si HTML (utilisé seulement si le texte brut est vide)
        elif mime_type == 'text/html' and data and not body:
            html = base64.urlsafe_b64decode(data).decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            body = soup.get_text()
    return body.strip()  # Nettoyage du texte

# Extraction des URLs depuis le texte de l'email
def extract_urls(text):
    return re.findall(r'https?://\S+', text)  # Match toutes les URLs commençant par http ou https

# Extraction des métadonnées des pièces jointes
def get_attachments(service, msg):
    attachments = []
    for part in msg['payload'].get('parts', []):
        filename = part.get('filename')
        body = part.get('body', {})
        # Vérifie s’il y a bien un fichier attaché
        if filename and 'attachmentId' in body:
            attachments.append({
                'filename': filename,
                'mimeType': part.get('mimeType'),
                'size': body.get('size')
            })
    return attachments

# Chargement des LabelEncoders pré-entraînés (pour encoder les domaines)
le_sender = joblib.load('models/le_sender.pkl')
le_receiver = joblib.load('models/le_receiver.pkl')

# Fonction de création des features à partir d’un email
def process_email(subject, body, sender, receiver_domain="example.com"):
    # Récupération du domaine de l'expéditeur
    sender_domain = sender.split('@')[-1].strip('> ') if '@' in sender else "unknown"
    
    # Encodage du domaine de l’expéditeur
    try:
        sender_domain_encoded = le_sender.transform([sender_domain])[0]
    except ValueError:
        sender_domain_encoded = -1  # Si domaine inconnu → -1

    # Encodage du domaine du destinataire
    try:
        receiver_domain_encoded = le_receiver.transform([receiver_domain])[0]
    except ValueError:
        receiver_domain_encoded = -1

    # Création du dictionnaire de caractéristiques (features)
    features = {
        'sender_name_length': len(sender.split('@')[0]),  # Longueur du nom d’expéditeur (avant @)
        'special_char_density': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', body)) / max(len(body), 1),  # Densité des caractères spéciaux
        'sender_domain_encoded': sender_domain_encoded,
        'num_urls': len(extract_urls(body)),  # Nombre d’URLs dans le corps de l’email
        'suspicious_urls': 1 if any(domain in url for url in extract_urls(body) for domain in ['bit.ly', 'tinyurl.com', 'goo.gl']) else 0,  # Présence de raccourcisseurs d’URL
        'phishing_words': sum(1 for word in ['urgent', 'gratuit', 'prix', 'compte', 'confidentiel'] if word in (subject + ' ' + body).lower()),  # Nombre de mots suspects
        'receiver_domain_encoded': receiver_domain_encoded,
    }

    return features  # Retourne les features pour l’email

# Fonction de prédiction de phishing
def predict_phishing(features):
    # Chargement du modèle de classification entraîné
    model = joblib.load('models/meta_model.pkl')
    # Conversion des features en DataFrame (attendu par le modèle)
    df = pd.DataFrame([features], columns=[
        'sender_domain_encoded',
        'phishing_words',
        'receiver_domain_encoded'
    ])
    # Prédiction à partir du modèle
    prediction = model.predict(df)

    return prediction[0]  # Retourne 0 (safe) ou 1 (phishing)

# Fonction principale pour récupérer, analyser et prédire les emails
def fetch_emails():
    service = authenticate_gmail()  # Authentifie et construit le service Gmail
    results = service.users().messages().list(userId='me', maxResults=10).execute()  # Liste les 10 derniers messages
    messages = results.get('messages', [])

    print(f"\n📬 {len(messages)} emails found:\n")

    # Parcours de chaque email
    for i, message in enumerate(messages, 1):
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        headers = msg['payload']['headers']

        subject = sender = "N/A"
        # Récupère le sujet et l'expéditeur depuis les en-têtes
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']

        body = get_email_body(msg['payload'])  # Récupération du corps de l'email
        urls = extract_urls(body)  # Extraction des URLs
        attachments = get_attachments(service, msg)  # Extraction des pièces jointes

        # Création des features + prédiction
        features = process_email(subject, body, sender)
        is_phishing = predict_phishing(features)

        # Affichage des informations de l’email + prédiction
        print(f"📩 Email {i}")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Body:\n{body[:300]}...")  # Affiche un extrait du corps
        print(f"URLs found: {urls}")
        print(f"Prediction: {'Phishing' if is_phishing else 'Safe'}\n")
        print("-" * 80)

# Point d’entrée du script
if __name__ == '__main__':
    fetch_emails()
