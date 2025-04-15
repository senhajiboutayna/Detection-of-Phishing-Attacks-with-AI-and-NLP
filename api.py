import base64
import os
import re
import mimetypes
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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

        print(f"📩 Email {i}")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Body:\n{body[:300]}...")  # Affiche un extrait
        print(f"URLs found: {urls}")
        print(f"Attachments: {[att['filename'] for att in attachments]}\n")
        print("-" * 80)

if __name__ == '__main__':
    fetch_emails()
