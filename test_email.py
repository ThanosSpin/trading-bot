import smtplib
from config import EMAIL_PASSWORD
from email.mime.text import MIMEText

msg = MIMEText("This is a test email.")
msg["Subject"] = "Test"
msg["From"] = "roispinola@gmail.com"
msg["To"] = "roispinola@gmail.com"

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login("roispinola@gmail.com", EMAIL_PASSWORD)
server.send_message(msg)
server.quit()