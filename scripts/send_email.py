# scripts/send_email.py

import smtplib
import os
from email.message import EmailMessage
import json

def send_email(subject, body):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

if __name__ == "__main__":
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
        subject = "Model Training & Deployment Successful"
        body = f"Model trained and evaluated successfully.\nResults:\nAccuracy: {results['accuracy']:.4f}\nF1 Score: {results['f1_score']:.4f}"
    except Exception as e:
        subject = "Model Training or Deployment Failed"
        body = f"An error occurred:\n{e}"

    send_email(subject, body)
