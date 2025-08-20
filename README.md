# Crypto Price Predictor
    2
    3 A simple web application that predicts the future price of cryptocurrencies using
      a machine learning model.
    4
    5 ## How It Works
    6
    7 This application uses an XGBoost Regressor model to predict cryptocurrency
      prices. The model is trained on historical daily price data (Open, High, Low,
      Close, Volume) for the past year, fetched from Yahoo Finance.
    8
    9 The prediction is based on the following features:
   10 - 7-day Simple Moving Average (SMA)
   11 - 14-day Simple Moving Average (SMA)
   12 - 4-day Price Momentum
   13 - 7-day Price Volatility
   14
   15 The model makes a base prediction, which is then adjusted by projecting the
      recent daily return trend into the future.
   16
   17 ## Features
   18
   19 -   **Web Interface**: A simple Flask web interface to interact with the model.
   20 -   **Dynamic Data**: Fetches the latest cryptocurrency data from Yahoo Finance.
   21 -   **On-the-fly Training**: The model is trained with the latest data when the
      application starts.
   22 -   **Price Prediction**: Predicts the price for a given cryptocurrency for a
      specified number of days in the future (1-90).
   23
   24 ## Dependencies
   25
   26 The project's core dependencies are:
   27
   28 -   **Flask**: For the web application.
   29 -   **yfinance**: To download historical market data from Yahoo Finance.
   30 -   **pandas**: For data manipulation and analysis.
   31 -   **scikit-learn**: For data preprocessing (MinMaxScaler).
   32 -   **xgboost**: For the prediction model (XGBRegressor).
   33
   34 You can install these dependencies using the provided `requirements.txt` file.
   35
   36 ## Installation
   37
   38 1.  **Clone the repository:**

      git clone <repository-url>
      cd crypto-predictor


   1
   2 2.  **Create a virtual environment:**

      python3 -m venv venv
      source venv/bin/activate

   1
   2 3.  **Install the dependencies:**

      pip install -r requirements.txt


   1
   2 ## Usage
   3
   4 1.  **Run the Flask application:**

      python app.py

   1
   2 2.  **Open your web browser** and navigate to:

      http://127.0.0.1:5000


   1
   2 3.  **Enter a cryptocurrency symbol** (e.g., `BTC`, `ETH`, `SOL`) and the number
     of days you want to predict into the future.
   3
   4 4.  **Click "Predict Price"** to see the result.
   5
   6 ## Disclaimer
   7
   8 This project is for educational purposes only. The predictions are based on a
     simple model and historical data, and should not be considered financial advice.
     Cryptocurrency markets are highly volatile, and you should do your own research
     before making any investment decisions.