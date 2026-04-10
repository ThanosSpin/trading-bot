# notify_retrain.py
import smtplib
from email.message import EmailMessage
from datetime import datetime
from config.config import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER

def send_retrain_email(success=True, errors=None):
    msg = EmailMessage()
    status = "✅ Success" if success else "❌ Failed"
    msg["Subject"] = f"Model Retrain {status} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    if success:
        body = f"✅ All models retrained successfully at {datetime.now()}"
    else:
        body = f"❌ Retrain completed with errors:\n" + "\n".join(errors or [])

    msg.set_content(body)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[INFO] Retrain notification sent.")
    except Exception as e:
        print(f"[Email Error] {e}")