from data_loader import get_stock_data
from model import train_model, load_model, predict_next
from strategy import should_trade
from trader import place_order
import os

def main():
    df = get_stock_data('NVDA', days=60)

    if not os.path.exists('models/model.pkl'):
        model = train_model(df)
    else:
        model = load_model()

    prob_up = predict_next(df, model)
    action = should_trade(prob_up)
    
    print(f"Prediction: {prob_up:.2f}, Action: {action}")
    
    if action in ['buy', 'sell']:
        place_order(action)

if __name__ == "__main__":
    main()
