import requests

def get_crypto_price(crypto):
    """Get current price from CoinGecko API (fallback)"""
    try:
        crypto = crypto.lower().replace('-usd', '')
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data.get(crypto, {}).get('usd')
    except Exception as e:
        print(f"Error fetching from CoinGecko: {e}")
        return None