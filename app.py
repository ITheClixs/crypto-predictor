import os
import warnings
warnings.filterwarnings('ignore')

# Import Flask (needed for the app); keep other heavy imports lazy so module
# can be imported in environments that don't have all ML/data packages.
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Crypto Predictor Class
class CryptoPredictor:
    def __init__(self):
        # Defer heavy ML object creation to training time so importing the
        # module doesn't require xgboost/sklearn to be installed.
        self.model = None
        self.scaler = None
        self.daily_return = 0.0
        self.is_trained = False

    def get_data(self, ticker='BTC-USD', days=60):
        """Get clean standardized data (lazy-import yfinance/pandas).

        Raises a clear ValueError when dependencies are missing so callers
        can return a user-friendly message instead of an ImportError.
        """
        try:
            import yfinance as yf
            import pandas as pd
        except Exception as e:
            raise ValueError("Missing package: please install 'yfinance' and 'pandas' (pip install -r requirements.txt)")

        try:
            data = yf.download(ticker, period=f"{days}d", progress=False)
            # Standardize column names
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            return data.dropna()
        except Exception as e:
            raise ValueError(f"Data download failed: {str(e)}")

    def add_features(self, data):
        """Create features without TA-Lib"""
        # Ensure pandas-like operations are available. If `data` is a
        # DataFrame coming from get_data this will work. We don't import
        # pandas here to avoid extra imports at module import time.
        # Simple moving averages
        data['SMA_7'] = data['Close'].rolling(7).mean()
        data['SMA_14'] = data['Close'].rolling(14).mean()

        # Price momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(4)

        # Volatility
        data['Volatility'] = data['Close'].rolling(7).std()

        return data.dropna()

    def train_model(self, ticker='BTC-USD'):
        """Train model with latest data"""
        try:
            data = self.get_data(ticker, 365)  # 1 year of data
            data = self.add_features(data)

            # Calculate daily return trend
            self.daily_return = data['Close'].pct_change().mean()

            # Prepare features
            X = data.drop('Close', axis=1)
            y = data['Close']

            # Lazy-import ML dependencies and create objects
            try:
                from sklearn.preprocessing import MinMaxScaler
                from xgboost import XGBRegressor
            except Exception:
                raise ValueError("Missing ML packages: please install 'scikit-learn' and 'xgboost' (pip install -r requirements.txt)")

            self.scaler = MinMaxScaler()
            self.model = XGBRegressor(objective='reg:squarederror', n_estimators=150)

            # Scale and train
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            return True
        except Exception as e:
            raise ValueError(f"Training failed: {str(e)}")

    def predict_price(self, crypto, days=7):
        """Predict future price after X days"""
        if not self.is_trained:
            try:
                self.train_model()  # Ensure model is trained
            except Exception as e:
                raise ValueError(f"Model training failed: {str(e)}")
        
        try:
            # Get recent data
            ticker = crypto if '-' in crypto else f"{crypto}-USD"
            data = self.get_data(ticker, 30)
            data = self.add_features(data)
            
            if len(data) < 7:
                raise ValueError("Not enough historical data")
            
            # Prepare latest features
            latest = data.iloc[-1:].drop('Close', axis=1)
            if self.scaler is None or self.model is None:
                raise ValueError('Model or scaler not available; train the model first')
            features = self.scaler.transform(latest)
            
            # Make base prediction
            base_price = self.model.predict(features)[0]
            
            # Apply trend projection
            projected_price = base_price * (1 + self.daily_return) ** days
            
            return round(projected_price, 2)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

# Initialize predictor (defer training until needed)
predictor = CryptoPredictor()

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        crypto = request.form.get("crypto", "BTC").strip().upper()
        days = min(int(request.form.get("days", 7)), 90)  # Max 90 days
        
        try:
            # Get current price
            ticker = crypto if '-' in crypto else f"{crypto}-USD"
            current_data = predictor.get_data(ticker, 1)
            current_price = current_data['Close'].iloc[-1]
            
            # Get predicted price
            predicted_price = predictor.predict_price(crypto, days)
            
            prediction = {
                'success': True,
                'crypto': crypto,
                'days': days,
                'current_price': f"${current_price:,.2f}",
                'predicted_price': f"${predicted_price:,.2f}",
                'change': f"{((predicted_price - current_price)/current_price*100):.1f}%",
                'is_up': predicted_price > current_price
            }
        except Exception as e:
            prediction = {
                'success': False,
                'error': str(e)
            }
    
    return render_template("index.html", prediction=prediction)

# Create templates folder and index.html
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

with open('templates/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Crypto Price Prediction</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
        form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        input, button { padding: 10px; margin: 10px 0; width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #3498db; color: white; border: none; cursor: pointer; }
        .result { padding: 20px; border-radius: 8px; }
        .success { background: #e8f5e9; border: 1px solid #c8e6c9; }
        .error { background: #ffebee; border: 1px solid #ffcdd2; }
        .price { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .up { color: #27ae60; }
        .down { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>Cryptocurrency Price Prediction</h1>
    <form method="POST">
        <input type="text" name="crypto" placeholder="Enter crypto symbol (e.g. BTC)" value="BTC" required>
        <input type="number" name="days" placeholder="Days to predict (1-90)" value="7" min="1" max="90" required>
        <button type="submit">Predict Price</button>
    </form>
    {% if prediction %}
        <div class="result {% if prediction.success %}success{% else %}error{% endif %}">
            {% if prediction.success %}
                <h2>{{ prediction.crypto }} Price Prediction</h2>
                <p>Current Price: <strong>{{ prediction.current_price }}</strong></p>
                <p>Predicted in {{ prediction.days }} days: 
                    <strong class="price {% if prediction.is_up %}up{% else %}down{% endif %}">
                        {{ prediction.predicted_price }}
                    </strong>
                </p>
                <p>Projected Change: <span class="{% if prediction.is_up %}up{% else %}down{% endif %}">
                    {{ prediction.change }}
                </span></p>
            {% else %}
                <h2>Error</h2>
                <p>{{ prediction.error }}</p>
                <p>Try symbols like: BTC, ETH, XRP, SOL</p>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>''')

if __name__ == "__main__":
    app.run(debug=True)
