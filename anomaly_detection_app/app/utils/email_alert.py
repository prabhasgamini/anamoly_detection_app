import smtplib
from email.mime.text import MIMEText
from flask import current_app

def send_alert(user_email, machine_id, timestamp, message):
    sender = current_app.config['MAIL_USERNAME']
    password = current_app.config['MAIL_PASSWORD']
    smtp_server = current_app.config['MAIL_SERVER']
    smtp_port = current_app.config['MAIL_PORT']

    subject = f"🚨 Anomaly Alert for Machine {machine_id}"
    body = f"Anomaly detected at {timestamp}.\nDetails: {message}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = user_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
    except Exception as e:
        print(f"Email failed: {e}")  # log to console