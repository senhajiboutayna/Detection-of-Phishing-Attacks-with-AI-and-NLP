import streamlit as st
import base64
import os
import re
import pandas as pd
import joblib
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Chargement des modèles
le_sender = joblib.load('models/le_sender.pkl')
le_receiver = joblib.load('models/le_receiver.pkl')
model = joblib.load('models/xgb_model.pkl')

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

def process_email(subject, body, sender, receiver_domain="example.com"):
    sender_domain = sender.split('@')[-1].strip('> ') if '@' in sender else "unknown"
    try:
        sender_domain_encoded = le_sender.transform([sender_domain])[0]
    except ValueError:
        sender_domain_encoded = -1
    try:
        receiver_domain_encoded = le_receiver.transform([receiver_domain])[0]
    except ValueError:
        receiver_domain_encoded = -1

    features = {
        'sender_name_length': len(sender.split('@')[0]),
        'special_char_density': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', body)) / max(len(body), 1),
        'sender_domain_encoded': sender_domain_encoded,
        'num_urls': len(extract_urls(body)),
        'suspicious_urls': 1 if any(domain in url for url in extract_urls(body) for domain in ['bit.ly', 'tinyurl.com', 'goo.gl']) else 0,
        'phishing_words': sum(1 for word in ['urgent', 'gratuit', 'prix', 'compte', 'confidentiel'] if word in (subject + ' ' + body).lower()),
        'receiver_domain_encoded': receiver_domain_encoded,
    }

    return pd.DataFrame([features])

# Streamlit UI
st.title("📧 Détecteur de Phishing Gmail avec IA")

if st.button("📬 Scanner mes 10 derniers emails"):
    with st.spinner("Authentification Gmail en cours..."):
        service = authenticate_gmail()
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])

    st.success(f"{len(messages)} emails récupérés.")

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
        features = process_email(subject, body, sender)
        prediction = model.predict(features)[0]

        st.markdown(f"### Email {i}")
        st.write(f"**De :** {sender}")
        st.write(f"**Sujet :** {subject}")
        st.write(f"**Prévision :** {' *Phishing*' if prediction else ' *Safe*'}")
        with st.expander("🔍 Contenu de l'email"):
            st.write(body[:500] + '...')

