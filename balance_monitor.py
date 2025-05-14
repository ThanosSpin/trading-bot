# balance_monitor_email.py
import alpaca_trade_api as tradeapi
import smtplib
import json
import os
from email.message import EmailMessage
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER


BALANCE_FILE = "data/last_balance.json"

def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[Email Error] {e}")

def load_last_balance():
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "r") as f:
            return json.load(f)
    return None

def save_last_balance(balance_dict):
    with open(BALANCE_FILE, "w") as f:
        json.dump(balance_dict, f, indent=2)

def check_balance():
    try:
        print("[INFO] Checking balance...")
        api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)
        account = api.get_account()
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)

        print(f"[INFO] Current cash: ${cash:.2f}, Portfolio value: ${portfolio_value:.2f}")

        last = load_last_balance()

        current = {
            "cash": round(cash, 2),
            "portfolio_value": round(portfolio_value, 2)
        }

        if last is None:
            send_email(
                subject="🔔 Alpaca Balance Initialized",
                body=f"Initial balance:\nCash: ${cash:.2f}\nPortfolio: ${portfolio_value:.2f}"
            )
            save_last_balance(current)

        elif current != last:
            send_email(
                subject="🔔 Alpaca Balance Update",
                body=f"Balance changed:\n"
                     f"Cash: ${last['cash']:.2f} → ${cash:.2f}\n"
                     f"Portfolio: ${last['portfolio_value']:.2f} → ${portfolio_value:.2f}"
            )
            save_last_balance(current)
        else:
            print("[INFO] No change in balance.")

    except Exception as e:
        print(f"[Balance Check Error] {e}")

if __name__ == "__main__":
    check_balance()