"""
Technical indicators utility module.
This module provides functions for calculating technical indicators,
with a fallback to pandas_ta or ta if TA-Lib is not available.
"""

import pandas as pd
import numpy as np

# Try to import TA-Lib, but provide alternatives if it's not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # Try to import alternative libraries
    try:
        import pandas_ta as pta
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        PANDAS_TA_AVAILABLE = False
        try:
            import ta
            TA_AVAILABLE = True
        except ImportError:
            TA_AVAILABLE = False
            print("WARNING: Neither TA-Lib, pandas_ta, nor ta is available. Installing pandas_ta...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta"])
            import pandas_ta as pta
            PANDAS_TA_AVAILABLE = True

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD indicator
    
    Args:
        df: DataFrame with 'close' column
        fast_period: Fast period
        slow_period: Slow period
        signal_period: Signal period
        
    Returns:
        tuple: (macd, macd_signal, macd_hist)
    """
    if TALIB_AVAILABLE:
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'], 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
        return macd, macd_signal, macd_hist
    elif PANDAS_TA_AVAILABLE:
        macd = pta.macd(
            df['close'], 
            fast=fast_period, 
            slow=slow_period, 
            signal=signal_period
        )
        return macd['MACD_12_26_9'], macd['MACDs_12_26_9'], macd['MACDh_12_26_9']
    elif TA_AVAILABLE:
        macd = ta.trend.MACD(
            df['close'], 
            window_fast=fast_period, 
            window_slow=slow_period, 
            window_sign=signal_period
        )
        return macd.macd(), macd.macd_signal(), macd.macd_diff()
    else:
        # Fallback implementation
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

def calculate_rsi(df, period=14):
    """
    Calculate RSI indicator
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period
        
    Returns:
        Series: RSI values
    """
    if TALIB_AVAILABLE:
        return talib.RSI(df['close'], timeperiod=period)
    elif PANDAS_TA_AVAILABLE:
        return pta.rsi(df['close'], length=period)
    elif TA_AVAILABLE:
        return ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    else:
        # Fallback implementation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        df: DataFrame with 'close' column
        period: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            df['close'], 
            timeperiod=period, 
            nbdevup=std_dev, 
            nbdevdn=std_dev, 
            matype=0
        )
        return upper, middle, lower
    elif PANDAS_TA_AVAILABLE:
        bbands = pta.bbands(df['close'], length=period, std=std_dev)
        return bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']
    elif TA_AVAILABLE:
        indicator_bb = ta.volatility.BollingerBands(
            df['close'], 
            window=period, 
            window_dev=std_dev
        )
        return indicator_bb.bollinger_hband(), indicator_bb.bollinger_mavg(), indicator_bb.bollinger_lband()
    else:
        # Fallback implementation
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

def calculate_sma(df, period):
    """
    Calculate Simple Moving Average
    
    Args:
        df: DataFrame with 'close' column
        period: SMA period
        
    Returns:
        Series: SMA values
    """
    if TALIB_AVAILABLE:
        return talib.SMA(df['close'], timeperiod=period)
    elif PANDAS_TA_AVAILABLE:
        return pta.sma(df['close'], length=period)
    elif TA_AVAILABLE:
        return ta.trend.sma_indicator(df['close'], window=period)
    else:
        # Fallback implementation
        return df['close'].rolling(window=period).mean()

def calculate_stochastic(df, k_period=14, d_period=3, slowing=3):
    """
    Calculate Stochastic Oscillator
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        k_period: K period
        d_period: D period
        slowing: Slowing period
        
    Returns:
        tuple: (stoch_k, stoch_d)
    """
    if TALIB_AVAILABLE:
        stoch_k, stoch_d = talib.STOCH(
            df['high'], 
            df['low'], 
            df['close'], 
            fastk_period=k_period, 
            slowk_period=slowing, 
            slowk_matype=0, 
            slowd_period=d_period, 
            slowd_matype=0
        )
        return stoch_k, stoch_d
    elif PANDAS_TA_AVAILABLE:
        stoch = pta.stoch(
            df['high'], 
            df['low'], 
            df['close'], 
            k=k_period, 
            d=d_period, 
            smooth_k=slowing
        )
        return stoch['STOCHk_14_3_3'], stoch['STOCHd_14_3_3']
    elif TA_AVAILABLE:
        stoch = ta.momentum.StochasticOscillator(
            df['high'], 
            df['low'], 
            df['close'], 
            window=k_period, 
            smooth_window=slowing
        )
        stoch_k = stoch.stoch()
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    else:
        # Fallback implementation
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_k = stoch_k.rolling(window=slowing).mean()
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ADX period
        
    Returns:
        Series: ADX values
    """
    if TALIB_AVAILABLE:
        return talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
    elif PANDAS_TA_AVAILABLE:
        adx = pta.adx(df['high'], df['low'], df['close'], length=period)
        return adx['ADX_14']
    elif TA_AVAILABLE:
        return ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=period).adx()
    else:
        # Fallback implementation (simplified)
        # This is a simplified version and not as accurate as the library implementations
        up_move = df['high'].diff()
        down_move = df['low'].diff(-1).abs()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr = np.maximum(
            df['high'] - df['low'], 
            np.maximum(
                abs(df['high'] - df['close'].shift(1)), 
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean()
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

def calculate_atr(df, period=14):
    """
    Calculate Average True Range
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
        
    Returns:
        Series: ATR values
    """
    if TALIB_AVAILABLE:
        return talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    elif PANDAS_TA_AVAILABLE:
        return pta.atr(df['high'], df['low'], df['close'], length=period)
    elif TA_AVAILABLE:
        return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    else:
        # Fallback implementation
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

def calculate_all_indicators(df):
    """
    Calculate all technical indicators for a DataFrame
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        
    Returns:
        DataFrame: Original DataFrame with added indicator columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # MACD
    if 'macd' not in df.columns:
        macd, macd_signal, macd_hist = calculate_macd(df)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
    
    # RSI
    if 'rsi' not in df.columns:
        df['rsi'] = calculate_rsi(df)
    
    # Bollinger Bands
    if 'bb_upper' not in df.columns:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
    
    # Moving Averages
    if 'sma_50' not in df.columns:
        df['sma_50'] = calculate_sma(df, 50)
    if 'sma_200' not in df.columns:
        df['sma_200'] = calculate_sma(df, 200)
    
    # Stochastic
    if 'stoch_k' not in df.columns:
        stoch_k, stoch_d = calculate_stochastic(df)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
    
    # ADX
    if 'adx' not in df.columns:
        df['adx'] = calculate_adx(df)
    
    # ATR
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)
    
    return df 