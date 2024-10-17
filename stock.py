import yfinance as yf

def get_historical_prices(ticker, start, end):
    data = yf.download(ticker, group_by="Ticker", start=start, end=end)
    data['Ticker'] = ticker
    data['Date'] = data.index
    return data

def get_historical_vix(start, end):
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(start=start, end=end)
    return vix_data