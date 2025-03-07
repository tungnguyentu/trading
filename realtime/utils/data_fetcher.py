import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.exceptions import BinanceAPIException
from ..utils.reporting import display_countdown

def get_market_data(client, symbol, timeframe, lookback_candles=500, short_window=20, long_window=50, atr_period=14):
    """Get latest market data"""
    try:
        print(f"Fetching latest data for {symbol}")
        
        # Ensure we have enough data for ML training
        min_candles = max(500, lookback_candles)
        
        # Calculate the start time based on number of candles
        # For 15-minute candles, we need to go back lookback_candles * 15 minutes
        start_time = int((datetime.now() - timedelta(minutes=min_candles * 15)).timestamp() * 1000)
        
        klines = client.get_historical_klines(
            symbol,
            timeframe,
            start_str=str(start_time)
        )
        
        print(f"Retrieved {len(klines)} candles from Binance")
        
        if len(klines) < 100:
            print("Warning: Not enough data points retrieved")
            
        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_volume', 'trades',
            'buy_base_volume', 'buy_quote_volume', 'ignore'
        ])
        
        # Convert numeric columns
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Calculate indicators
        df = calculate_indicators(df, short_window, long_window, atr_period)
        
        return df
        
    except BinanceAPIException as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_indicators(df, short_window=20, long_window=50, atr_period=14):
    """Calculate technical indicators for trading signals"""
    # Calculate SMAs
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_period).mean()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
    df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1
    
    return df