import streamlit as st
import base64
import os
import re
import pandas as pd
import joblib
import html  # <-- Ajouté pour échapper les caractères spéciaux
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
            html_content = base64.urlsafe_b64decode(data).decode('utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
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

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{b64_img}"

logo_base64 = get_base64_image("img/orange.jpg")
ml_base64 = get_base64_image("img/ml.png")




# Fonction pour appliquer l'image en fond d'écran
def set_background_image(image_path):
    bin_str = get_base64_image(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Applique l'image OB.png comme fond
set_background_image('img/OB.jpg')


# Style CSS
st.markdown(f"""
    <style>
    

    .header-banner {{
        background-color: #1f1f1f;
        padding: 80px 30px;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: start;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        height: 80px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }}

    .header-banner img {{
        height: 70px;
        margin-right: 30px;
        margin-top: 50px;
    }}

    .header-text {{
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-top: 70px;
    }}

    .main-container {{
        max-width: 1000px;
        margin: auto;
        padding: 120px 30px 30px 30px;
    }}

    .phishing-label {{
        font-size: 20px;
        font-weight: bold;
        padding: 8px 15px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 10px;
        margin-left: 90%; /* Espacement entre le cercle et le label */

    }}

    .safe {{color: #155724; }}
    .phishing {{color: #721c24; }}

    .email-card {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap; /* Pour permettre à l'élément de se réorganiser en fonction de l'espace */

    }}

    .email-info {{
        flex: 1;
        color: black
    }}

    .circle-graph {{
        width: 90px;
        height: 90px;
        position: relative;
        margin-left: 90%; /* Espacement entre le contenu de l'email et le cercle */

    }}

    .circle-graph svg {{
        transform: rotate(360deg);
    }}
            
    .email-card > .circle-graph, .email-card > .phishing-label {{
        order: 2; /* Positionne le cercle et le label à droite */
    }}

    .email-card > .email-info {{
        order: 1; /* Positionne le texte à gauche */
    }}

    .circle-graph text {{
        font-size: 16px;
        fill: black;
        text-anchor: middle;
        dominant-baseline: middle;
    }}
    </style>

    <div class="header-banner">
        <img src="{logo_base64}" alt="Orange Logo">
        <div class="header-text">Détection de Phishing - Projet Orange Business
        </div>
    </div>
""", unsafe_allow_html=True)



st.markdown("""
    <style> h1 { margin-top: -80px; } </style>
""", unsafe_allow_html=True)

st.markdown("""<div class="main-container">""", unsafe_allow_html=True)


st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 20px;'>
        <img src="{ml_base64}" alt="ML Logo" style="height: 60px;">
        <h1 style='color: black; margin: 0;'>Analyse de votre boîte Gmail</h1>
    </div>
""", unsafe_allow_html=True)



if st.button("Scanner mes 10 derniers emails"):
    with st.spinner("Authentification Gmail..."):
        service = authenticate_gmail()
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])

    st.success(f"{len(messages)} emails récupérés !")

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
        proba = model.predict_proba(features)[0][1]
        percent = int(proba * 100)

        # Protection HTML
        safe_sender = html.escape(sender)
        safe_subject = html.escape(subject)

        # SVG cercle de probabilité
        circle_svg = f"""
        <div class="circle-graph">
            <svg width="90" height="90">
                <g transform="rotate(-90, 45, 45)">
                    <circle cx="45" cy="45" r="40" stroke="#eee" stroke-width="10" fill="none"/>
                    <circle cx="45" cy="45" r="40" stroke="#dc3545" stroke-width="10"
                        stroke-dasharray="{percent * 2.51} 251" fill="none" />
                    <text x="45" y="50" transform="rotate(90, 45, 50)">{percent}%</text>
                </g>
            </svg>
        </div>
        
        """


        st.markdown(f"<hr style='border: 3px solid #F37117;'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: black;'>Email {i}</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="email-card">
                <div class="email-info">
                    <div><strong>De :</strong> {safe_sender}</div>
                    <div><strong>Sujet :</strong> {safe_subject}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        label_html = (
            "<div class='phishing-label phishing'>PHISHING</div>" if prediction
            else "<div class='phishing-label safe'>SAIN</div>"
        )

        st.markdown(circle_svg, unsafe_allow_html=True)
        st.markdown(label_html, unsafe_allow_html=True)

        with st.expander("Aperçu du contenu de l'email"):
            st.write(body[:700] + '...')

st.markdown("</div>", unsafe_allow_html=True)
