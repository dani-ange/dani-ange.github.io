import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase

def send_email(subject, body, zip_path=None):
    """
    Sends an email with optional ZIP attachment using SMTP.

    Environment Variables:
        EMAIL_USER (str): Sender's email address.
        EMAIL_PASS (str): Sender's email password or app password.
        EMAIL_RECEIVER (str): Recipient's email address.

    Args:
        subject (str): Email subject line.
        body (str): Plain-text body of the email.
        zip_path (str, optional): Path to a ZIP file to attach.

    Raises:
        FileNotFoundError: If zip_path is provided but the file doesn't exist.
        smtplib.SMTPException: If there's an error sending the email.
        KeyError: If required environment variables are missing.
    """
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"]

    # Create a multipart message
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject

    # Attach the email body as plain text
    msg.attach(MIMEText(body, "plain"))

    # Attach ZIP file if provided
    if zip_path:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Attachment not found: {zip_path}")

        with open(zip_path, "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(zip_path)}"'
            )
            msg.attach(part)

    # Send email via secure SMTP (SSL)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

if __name__ == "__main__":
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
        subject = "Model Training & Deployment Successful"
        body = (
            f"Model trained and evaluated successfully.\n"
            f"Results:\nAccuracy: {results['accuracy']:.4f}\n"
            f"F1 Score: {results['f1_score']:.4f}"
            f"📄 View full training log at:\n"
            f"https://dani-ange.github.io/"
        )
        zip_path = "bundle.zip"  # <-- Set your zip file name here
    except Exception as e:
        subject = "Model Training or Deployment Failed"
        body = f"An error occurred:\n{e}"
        zip_path = None

    send_email(subject, body, zip_path=zip_path)
