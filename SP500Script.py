import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np

def ATR(DF, n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    return df["ATR"]

def ADX(DF, n=20):
    "function to calculate ADX"
    df = DF.copy()
    df["ATR"] = ATR(DF, n)
    df["upmove"] = df["High"] - df["High"].shift(1)
    df["downmove"] = df["Low"].shift(1) - df["Low"]
    df["+dm"] = np.where((df["upmove"] > df["downmove"]) & (df["upmove"] > 0), df["upmove"], 0)
    df["-dm"] = np.where((df["downmove"] > df["upmove"]) & (df["downmove"] > 0), df["downmove"], 0)
    df["+di"] = 100 * (df["+dm"] / df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["-di"] = 100 * (df["-dm"] / df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["ADX"] = 100 * abs((df["+di"] - df["-di"]) / (df["+di"] + df["-di"])).ewm(alpha=1/n, min_periods=n).mean()
    return df["ADX"]

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod()
    n = len(df) / 252  # Assuming 252 trading days in a year
    CAGR_value = (df["cum_return"].iloc[-1])**(1/n) - 1
    return CAGR_value

def RSI(DF, n=14):
    "function to calculate RSI"
    df = DF.copy()
    df["change"] = df["Adj Close"] - df["Adj Close"].shift(1)
    df["gain"] = np.where(df["change"]>=0, df["change"], 0)
    df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/ (1 + df["rs"]))
    return df["rsi"]

def MACD(DF, a=12 ,b=26, c=9):
    """function to calculate MACD
       typical values a(fast moving average) = 12; 
                      b(slow moving average) =26; 
                      c(signal line ma window) =9"""
    df = DF.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=a, min_periods=a).mean()
    df["ma_slow"] = df["Adj Close"].ewm(span=b, min_periods=b).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    return df[["macd", "signal"]]

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1+df["return"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    return (df["drawdown"]/df["cum_roll_max"]).max()

def calmar(DF):
    "function to calculate calmar ratio"
    df = DF.copy()
    return CAGR(df)/max_dd(df)

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    vol = df["daily_ret"].std() * np.sqrt(252)
    return vol

def sharpe(DF, riskfree_rate):
    "function to calculate Sharpe Ratio of a trading strategy"
    df = DF.copy()
    return (CAGR(df) - riskfree_rate)/volatility(df)

def sortino(DF, rf):
    "function to calculate Sortino Ratio of a trading strategy"
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    neg_return = np.where(df["return"]>0,0,df["return"])
    #below you will see two ways to calculate the denominator (neg_vol), some people use the
    #standard deviation of negative returns while others use a downward deviation approach,
    #you can use either. However, downward deviation approach is more widely used
    neg_vol = np.sqrt((pd.Series(neg_return[neg_return != 0]) ** 2).mean() * 252)
    #neg_vol = pd.Series(neg_return[neg_return != 0]).std() * np.sqrt(252)
    return (CAGR(df) - rf)/neg_vol

# Define the stock and date range
stocks = ["^GSPC"]
start = dt.date.today() - dt.timedelta(days=10000)  # Correct usage
end = dt.date.today()

# start = dt.date(2024, 1, 1)
# end = dt.date.today()

# Initialize an empty dataframe for closing prices and dictionary for OHLCV data
cl_price = pd.DataFrame()
ohlcv_data = {}

# Downloading the OHLCV data and storing it in the dictionary
for ticker in stocks:
    ohlcv_data[ticker] = yf.download(ticker, start, end)

# Integrating indicators into the dataframe
# Define risk-free rate (using 10-year Treasury rate as approximation)
RF_RATE = 0.04  # 4% as example

# Modify the final loop to include all indicators and metrics
for ticker in stocks:
    df = ohlcv_data[ticker]
    
    # Technical Indicators
    df["ATR"] = ATR(df)
    df["ADX"] = ADX(df)
    df["RSI"] = RSI(df)
    macd_data = MACD(df)
    df["MACD"] = macd_data["macd"]
    df["MACD_Signal"] = macd_data["signal"]
    
    # Performance Metrics (as rolling calculations where applicable)
    df["Volatility"] = df["Adj Close"].rolling(window=252).apply(lambda x: volatility(pd.DataFrame({'Adj Close': x})))
    df["Max_Drawdown"] = df["Adj Close"].rolling(window=252).apply(lambda x: max_dd(pd.DataFrame({'Adj Close': x})))
    
    # Scalar Performance Metrics
    # df["CAGR"] = CAGR(df)
    # df["Sharpe_Ratio"] = sharpe(df, RF_RATE)
    # df["Sortino_Ratio"] = sortino(df, RF_RATE)
    # df["Calmar_Ratio"] = calmar(df)
    
    # Update the dictionary with the modified dataframe
    ohlcv_data[ticker] = df
    
    # Export to CSV
    df.to_csv(f"SP500_with_indicators_{ticker}.csv")