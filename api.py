import base64
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import joblib  # pour charger le modèle sauvegardé

# Scope lecture seule des emails
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_email_body(message_payload):
    parts = message_payload.get('parts', [])
    body = ""
    for part in parts:
        mime_type = part.get('mimeType')
        data = part.get('body', {}).get('data')
        if mime_type == 'text/plain' and data:
            body = base64.urlsafe_b64decode(data).decode('utf-8')
        elif mime_type == 'text/html' and data and not body:
            html = base64.urlsafe_b64decode(data).decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            body = soup.get_text()
    return body.strip()

def extract_urls(text):
    return re.findall(r'https?://\S+', text)

def get_attachments(service, msg):
    attachments = []
    for part in msg['payload'].get('parts', []):
        filename = part.get('filename')
        body = part.get('body', {})
        if filename and 'attachmentId' in body:
            attachments.append({
                'filename': filename,
                'mimeType': part.get('mimeType'),
                'size': body.get('size')
            })
    return attachments

le_sender = joblib.load('models/le_sender.pkl')
le_receiver = joblib.load('models/le_receiver.pkl')
def process_email(subject, body, sender, receiver_domain="example.com"):
    # Extraire le domaine de l'expéditeur
    sender_domain = sender.split('@')[-1].strip('> ') if '@' in sender else "unknown"
    
    # Encoder les domaines (avec gestion des nouvelles valeurs)
    try:
        sender_domain_encoded = le_sender.transform([sender_domain])[0]
    except ValueError:
        sender_domain_encoded = -1  # Valeur pour "inconnu"
    
    try:
        receiver_domain_encoded = le_receiver.transform([receiver_domain])[0]
    except ValueError:
        receiver_domain_encoded = -1

    features = {
        'sender_name_length': len(sender.split('@')[0]),
        'special_char_density': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', body)) / max(len(body), 1),
        'sender_domain_encoded': sender_domain_encoded,  # Ajouté
        'num_urls': len(extract_urls(body)),
        'suspicious_urls': 1 if any(domain in url for url in extract_urls(body) for domain in ['bit.ly', 'tinyurl.com', 'goo.gl']) else 0,
        'phishing_words': sum(1 for word in ['urgent', 'gratuit', 'prix', 'compte', 'confidentiel'] if word in (subject + ' ' + body).lower()),
        'receiver_domain_encoded': receiver_domain_encoded,  # Ajouté
    }

    return features

def predict_phishing(features):
    # Charger le modèle entraîné (XGBoost par exemple)
    model = joblib.load('models/xgboost_model.pkl')  # Assurez-vous d'avoir sauvegardé le modèle
    # Convertir les features en DataFrame
    df = pd.DataFrame([features], columns=[
        'sender_name_length',
        'special_char_density',
        'sender_domain_encoded',
        'num_urls',
        'suspicious_urls',
        'phishing_words',
        'receiver_domain_encoded'
    ])
    # Prédiction
    prediction = model.predict(df)

    return prediction[0]


def fetch_emails():
    service = authenticate_gmail()
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])

    print(f"\n📬 {len(messages)} emails found:\n")

    for i, message in enumerate(messages, 1):
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        headers = msg['payload']['headers']

        subject = sender = "N/A"
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']

        body = get_email_body(msg['payload'])
        urls = extract_urls(body)
        attachments = get_attachments(service, msg)

        # Prétraiter et prédire
        features = process_email(subject, body, sender)
        is_phishing = predict_phishing(features)

        print(f"📩 Email {i}")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Body:\n{body[:300]}...")  # Affiche un extrait
        print(f"URLs found: {urls}")
        print(f"Attachments: {[att['filename'] for att in attachments]}\n")
        print(f"Prediction: {'Phishing' if is_phishing else 'Safe'}\n")
        print("-" * 80)

if __name__ == '__main__':
    fetch_emails()
