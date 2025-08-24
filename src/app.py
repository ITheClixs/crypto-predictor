import os
import warnings
warnings.filterwarnings('ignore')
import pickle

# Import Flask (needed for the app); keep other heavy imports lazy so module
# can be imported in environments that don't have all ML/data packages.
from flask import Flask, render_template, request

# Initialize Flask app and ensure it uses the repo-level `templates` folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=TEMPLATES_DIR)


def _to_scalar(val):
    """Coerce pandas/numpy/iterable values to a Python float scalar."""
    try:
        return float(val)
    except Exception:
        pass
    # pandas Series / numpy types
    try:
        # pandas Series: try .iloc or .values
        if hasattr(val, 'iloc'):
            return float(val.iloc[-1])
        if hasattr(val, 'item'):
            return float(val.item())
        if hasattr(val, 'values'):
            arr = val.values
            return float(arr.flatten()[-1])
    except Exception:
        pass
    # list/tuple
    try:
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return float(val[-1])
    except Exception:
        pass
    raise ValueError(f"Unable to convert value to scalar: {type(val)}")

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
            if data is None or data.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'. Please check the symbol and try again.")
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
    # Do NOT auto-train here. Training requires heavy dependencies and
    # often times out on user machines. Prefer using a pre-trained model
    # (loaded at startup) or the lightweight fallback implemented below.
        
        try:
            # Get recent data
            ticker = crypto if '-' in crypto else f"{crypto}-USD"
            try:
                data = self.get_data(ticker, 30)
                data = self.add_features(data)
            except Exception as e_data:
                # If yfinance/pandas aren't available or data download fails,
                # fall back to CoinGecko public API (no heavy deps required).
                try:
                    # Prefer requests if available; otherwise use urllib (stdlib)
                    try:
                        import requests
                        def http_get(url, timeout=10):
                            r = requests.get(url, timeout=timeout)
                            r.raise_for_status()
                            return r.text
                    except Exception:
                        import urllib.request, json
                        def http_get(url, timeout=10):
                            with urllib.request.urlopen(url, timeout=timeout) as resp:
                                return resp.read().decode('utf-8')

                    # map common symbols to CoinGecko ids
                    symbol = crypto.split('-')[0]
                    cg_map = {
                        'BTC': 'bitcoin',
                        'ETH': 'ethereum',
                        'XRP': 'ripple',
                        'SOL': 'solana',
                        'LTC': 'litecoin',
                        'DOGE': 'dogecoin'
                    }
                    cg_id = cg_map.get(symbol)
                    if not cg_id:
                        raise ValueError("Unknown symbol for fallback. Install yfinance/pandas or use BTC/ETH/XRP/SOL/LTC/DOGE")

                    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=30"
                    txt = http_get(url)
                    import json
                    obj = json.loads(txt)
                    prices = obj.get('prices') or []
                    if not prices:
                        raise ValueError('CoinGecko returned no price data')

                    # prices is list of [timestamp, price]; build a simple closes list
                    closes = [p[1] for p in prices]
                    # create a minimal data-like object with Close series compatible access
                    class SimpleData:
                        def __init__(self, closes):
                            self._closes = closes
                        @property
                        def Close(self):
                            return self._closes
                        def __len__(self):
                            return len(self._closes)
                        def iloc(self, idx):
                            return self._closes[idx]
                    # For our fallback we only need 'Close' values as a pandas-like series
                    # We'll represent it as a dict-like object for downstream code.
                    data = {'Close': closes}
                except Exception as e2:
                    raise ValueError(f"Data download failed and CoinGecko fallback failed: {e2}")
            
            # Determine number of historical points available. If data is
            # a dict (CoinGecko fallback) use the 'Close' list length; if
            # it's a pandas DataFrame/Series use len(data).
            try:
                n_points = len(data['Close']) if isinstance(data, dict) else len(data)
            except Exception:
                # conservative fallback
                n_points = 0

            if n_points < 7:
                raise ValueError("Not enough historical data")
            
            # If model or scaler are missing (pre-trained model couldn't be
            # loaded because sklearn/xgboost aren't installed), fall back to a
            # lightweight heuristic: project recent average daily return.
            if self.scaler is None or self.model is None:
                # If 'data' came from CoinGecko fallback it is a dict with 'Close'
                # as a plain list; otherwise it's a pandas DataFrame/Series.
                try:
                    if isinstance(data, dict):
                        closes = data['Close']
                        # compute returns on consecutive entries
                        recent_returns = []
                        for i in range(1, len(closes)):
                            prev = closes[i-1]
                            cur = closes[i]
                            if prev:
                                recent_returns.append((cur - prev) / prev)
                    else:
                        # pandas Series branch; compute pct_change safely
                        recent_returns = data['Close'].pct_change().dropna()

                    # safe way to compute mean of last up to 7 entries for both
                    # list and pandas Series types
                    mean_daily = 0.0
                    if hasattr(recent_returns, '__len__') and len(recent_returns) > 0:
                        try:
                            # pandas Series supports tail/mean
                            if hasattr(recent_returns, 'tail'):
                                mean_daily = float(recent_returns.tail(7).mean())
                            else:
                                last7 = recent_returns[-7:]
                                mean_daily = float(sum(last7) / len(last7))
                        except Exception:
                            mean_daily = 0.0
                except Exception:
                    mean_daily = 0.0

                # remember the fallback daily return
                self.daily_return = float(mean_daily) if mean_daily is not None else 0.0

                # get current price
                if isinstance(data, dict):
                    current_price = data['Close'][-1]
                else:
                    current_price = float(data['Close'].iloc[-1])

                projected_price = current_price * (1 + self.daily_return) ** days
                return round(projected_price, 2)

            # Normal path: scale features and predict with the model
            # compute latest row/features (pandas DataFrame expected)
            latest = data.iloc[-1:].drop('Close', axis=1)
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

# Try to load a bundled pre-trained model to avoid expensive local training.
# The file is expected at repo-root `data/crypto_predictor.pkl` and can be
# either the model object itself or a dict {'model': ..., 'scaler': ..., 'daily_return': ...}.
MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'crypto_predictor.pkl')
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as fh:
            blob = pickle.load(fh)
        if isinstance(blob, dict):
            # common keys fallback
            predictor.model = blob.get('model') or blob.get('clf') or blob.get('estimator')
            predictor.scaler = blob.get('scaler', predictor.scaler)
            predictor.daily_return = blob.get('daily_return', predictor.daily_return)
        else:
            predictor.model = blob

        if predictor.model is not None:
            predictor.is_trained = True
            # best-effort message
            print(f"Loaded pre-trained model from {MODEL_PATH}")
    except Exception as e:
        # Don't crash app if loading fails; fall back to training path when requested
        print(f"Warning: failed to load pre-trained model: {e}")

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

            # Coerce to numeric scalars to avoid pandas Series formatting errors
            try:
                current_price = _to_scalar(current_price)
            except Exception:
                current_price = float(current_price)
            try:
                predicted_price = _to_scalar(predicted_price)
            except Exception:
                predicted_price = float(predicted_price)
            
            prediction = {
                'success': True,
                'crypto': crypto,
                'days': days,
                'current_price': f"${current_price:,.2f}",
                'predicted_price': f"${predicted_price:,.2f}",
                'change': f"{((predicted_price - current_price)/current_price*100 if current_price else 0):.1f}%",
                'is_up': predicted_price > current_price
            }
        except Exception as e:
            prediction = {
                'success': False,
                'error': str(e)
            }
    
    return render_template("index.html", prediction=prediction)

# Create templates folder and index.html at repo root if missing
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

index_path = os.path.join(TEMPLATES_DIR, 'index.html')
# Only write the template if it doesn't exist to avoid overwriting manual edits
if not os.path.exists(index_path):
    with open(index_path, 'w') as f:
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
        <div class="result {{ 'success' if prediction.success else 'error' }}">
            {% if prediction.success %}
                <h2>{{ prediction.crypto }} Price Prediction</h2>
                <p>Current Price: <strong>{{ prediction.current_price }}</strong></p>
                <p>Predicted in {{ prediction.days }} days: 
                    <strong class="price {{ 'up' if prediction.is_up else 'down' }}">
                        {{ prediction.predicted_price }}
                    </strong>
                </p>
                <p>Projected Change: <span class="{{ 'up' if prediction.is_up else 'down' }}">
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
    app.run(debug=True, port=8000)
