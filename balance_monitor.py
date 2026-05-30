# balance_monitor_email.py
import alpaca_trade_api as tradeapi
import smtplib
import json
import os
from email.message import EmailMessage
from config import (
    API_KEY,
    API_SECRET,
    BASE_URL,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
)


BOT_ENV = (os.getenv("BOT_ENV") or os.getenv("BOTENV") or "live").strip().lower()
if BOT_ENV not in {"live", "paper"}:
    BOT_ENV = "live"

BALANCE_FILE = os.path.join("data", f"last_balance_{BOT_ENV}.json")

print(f"[DEBUG] Using balance file for env '{BOT_ENV}': {BALANCE_FILE}")

def _mask_secret(value, keep=4):
    if not value:
        return "None"
    value = str(value)
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep)


def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        print(f"[DEBUG] Preparing email: subject={subject}")
        print(f"[DEBUG] Email sender={EMAIL_SENDER}")
        print(f"[DEBUG] Email receiver={EMAIL_RECEIVER}")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[Email Error] {e}")


def load_last_balance():
    print(f"[DEBUG] Loading last balance from: {BALANCE_FILE}")
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "r") as f:
            data = json.load(f)
            print(f"[DEBUG] Last balance loaded: {data}")
            return data
    print("[DEBUG] No existing balance file found.")
    return None


def save_last_balance(balance_dict):
    os.makedirs(os.path.dirname(BALANCE_FILE), exist_ok=True)
    print(f"[DEBUG] Saving balance to: {BALANCE_FILE}")
    print(f"[DEBUG] Saved balance payload: {balance_dict}")
    with open(BALANCE_FILE, "w") as f:
        json.dump(balance_dict, f, indent=2)


def check_balance():
    try:
        print("[INFO] Checking balance...")

        # -------------------------------------------------
        # Runtime environment debug
        # -------------------------------------------------
        print(f"[DEBUG] BOT_ENV={os.getenv('BOT_ENV')}")
        print(f"[DEBUG] ENV_NAME={os.getenv('ENV_NAME')}")
        print(f"[DEBUG] BASE_URL={BASE_URL}")
        print(f"[DEBUG] API_KEY={_mask_secret(API_KEY)}")
        print(f"[DEBUG] API_SECRET={_mask_secret(API_SECRET)}")
        print(f"[DEBUG] cwd={os.getcwd()}")
        print(f"[DEBUG] balance_file_abs={os.path.abspath(BALANCE_FILE)}")

        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
        print("[DEBUG] Alpaca REST client created successfully.")

        account = api.get_account()
        print("[DEBUG] Alpaca account object fetched successfully.")

        # -------------------------------------------------
        # Account debug
        # -------------------------------------------------
        print(f"[DEBUG] account.id={getattr(account, 'id', None)}")
        print(f"[DEBUG] account.account_number={getattr(account, 'account_number', None)}")
        print(f"[DEBUG] account.status={getattr(account, 'status', None)}")
        print(f"[DEBUG] account.currency={getattr(account, 'currency', None)}")
        print(f"[DEBUG] account.buying_power={getattr(account, 'buying_power', None)}")
        print(f"[DEBUG] account.cash={getattr(account, 'cash', None)}")
        print(f"[DEBUG] account.portfolio_value={getattr(account, 'portfolio_value', None)}")
        print(f"[DEBUG] account.equity={getattr(account, 'equity', None)}")
        print(f"[DEBUG] account.pattern_day_trader={getattr(account, 'pattern_day_trader', None)}")
        print(f"[DEBUG] account.trading_blocked={getattr(account, 'trading_blocked', None)}")
        print(f"[DEBUG] account.transfers_blocked={getattr(account, 'transfers_blocked', None)}")
        print(f"[DEBUG] account.account_blocked={getattr(account, 'account_blocked', None)}")

        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)

        print(f"[INFO] Current cash: ${cash:.2f}, Portfolio value: ${portfolio_value:.2f}")

        last = load_last_balance()

        current = {
            "cash": round(cash, 2),
            "portfolio_value": round(portfolio_value, 2)
        }

        print(f"[DEBUG] Current balance snapshot: {current}")

        if last is None:
            print("[DEBUG] No previous balance found. Sending initialization email.")
            send_email(
                subject="🔔 Alpaca Balance Initialized",
                body=(
                    f"Initial balance:\n"
                    f"Cash: ${cash:.2f}\n"
                    f"Portfolio: ${portfolio_value:.2f}\n\n"
                    f"BOT_ENV: {os.getenv('BOT_ENV')}\n"
                    f"ENV_NAME: {os.getenv('ENV_NAME')}\n"
                    f"Base URL: {BASE_URL}\n"
                    f"Account ID: {getattr(account, 'id', None)}\n"
                    f"Account Number: {getattr(account, 'account_number', None)}"
                )
            )
            save_last_balance(current)

        elif current != last:
            print(f"[DEBUG] Balance change detected. Previous={last} New={current}")
            send_email(
                subject="🔔 Alpaca Balance Update",
                body=(
                    f"Balance changed:\n"
                    f"Cash: ${last['cash']:.2f} → ${cash:.2f}\n"
                    f"Portfolio: ${last['portfolio_value']:.2f} → ${portfolio_value:.2f}\n\n"
                    f"BOTENV: {os.getenv('BOTENV')}\n"
                    f"ENVNAME: {os.getenv('ENVNAME')}\n"
                    f"Base URL: {BASE_URL}\n"
                    f"Account ID: {getattr(account, 'id', None)}\n"
                    f"Account Number: {getattr(account, 'account_number', None)}"
                )
            )
            save_last_balance(current)
        else:
            print("[INFO] No change in balance.")

    except Exception as e:
        print(f"[Balance Check Error] {e}")


if __name__ == "__main__":
    check_balance()