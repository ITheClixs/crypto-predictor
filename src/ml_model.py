import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class CryptoModel:
    def __init__(self):
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=150)
        self.scaler = MinMaxScaler()
        self.daily_return = 0.0
        self.is_trained = False

    def get_data(self, ticker='BTC-USD', days=60):
        """Get clean standardized data"""
        try:
            data = yf.download(ticker, period=f"{days}d", progress=False)
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data.dropna()
        except Exception as e:
            raise ValueError(f"Data download failed: {str(e)}")

    def add_features(self, data):
        """Create features without TA-Lib"""
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
            self.train_model()  # Ensure model is trained
            
        try:
            # Get recent data
            ticker = crypto if '-' in crypto else f"{crypto}-USD"
            data = self.get_data(ticker, 30)
            data = self.add_features(data)
            
            if len(data) < 7:
                raise ValueError("Not enough historical data")
            
            # Prepare latest features
            latest = data.iloc[-1:].drop('Close', axis=1)
            features = self.scaler.transform(latest)
            
            # Make base prediction
            base_price = self.model.predict(features)[0]
            
            # Apply trend projection
            projected_price = base_price * (1 + self.daily_return) ** days
            
            return round(projected_price, 2)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

# Create and export the predictor instance
crypto_model = CryptoModel()
crypto_model.train_model()  # Pre-train with BTC