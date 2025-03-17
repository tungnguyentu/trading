"""
Advanced technical indicator calculations with error handling.
"""

import pandas as pd
import numpy as np
import talib
from .safe_indicators import safe_adx, safe_rsi, calculate_all_safe_indicators
import warnings
from datetime import datetime

def calculate_all_indicators(df):
    """
    Calculate a comprehensive set of technical indicators for trading
    
    Args:
        df: DataFrame with OHLC price data
        
    Returns:
        DataFrame with added indicator columns
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    try:
        # Suppress warnings during calculations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Make sure we have the required OHLC columns
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in result_df.columns:
                    print(f"Error: Missing required column '{col}' in data")
                    return df  # Return original dataframe
            
            # Moving Averages
            result_df['sma_5'] = result_df['close'].rolling(window=5).mean()
            result_df['sma_10'] = result_df['close'].rolling(window=10).mean()
            result_df['sma_20'] = result_df['close'].rolling(window=20).mean()
            result_df['sma_50'] = result_df['close'].rolling(window=50).mean()
            result_df['sma_100'] = result_df['close'].rolling(window=100).mean()
            result_df['sma_200'] = result_df['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            result_df['ema_5'] = result_df['close'].ewm(span=5, adjust=False).mean()
            result_df['ema_12'] = result_df['close'].ewm(span=12, adjust=False).mean()
            result_df['ema_26'] = result_df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            result_df['macd'] = result_df['ema_12'] - result_df['ema_26']
            result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
            result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
            
            # RSI
            result_df['rsi'] = safe_rsi(result_df['close'], window=14, fillna=True)
            
            # Bollinger Bands
            result_df['bb_middle'] = result_df['close'].rolling(window=20).mean()
            result_df['bb_std'] = result_df['close'].rolling(window=20).std()
            result_df['bb_upper'] = result_df['bb_middle'] + (result_df['bb_std'] * 2)
            result_df['bb_lower'] = result_df['bb_middle'] - (result_df['bb_std'] * 2)
            result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
            result_df['bb_pct'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
            
            # Stochastic Oscillator
            low_min = result_df['low'].rolling(window=14).min()
            high_max = result_df['high'].rolling(window=14).max()
            
            # Avoid division by zero in stochastic calculation
            k_divisor = high_max - low_min
            valid_k_divisor = k_divisor > 0
            
            result_df['stoch_k'] = pd.Series(np.zeros(len(result_df)), index=result_df.index)
            result_df.loc[valid_k_divisor, 'stoch_k'] = 100 * ((result_df.loc[valid_k_divisor, 'close'] - low_min[valid_k_divisor]) / k_divisor[valid_k_divisor])
            result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3).mean()
            
            # Fill NaN values in stochastic
            result_df['stoch_k'] = result_df['stoch_k'].fillna(50)
            result_df['stoch_d'] = result_df['stoch_d'].fillna(50)
            
            # ADX (Average Directional Index) - using our safe implementation
            adx_df = safe_adx(
                high=result_df['high'],
                low=result_df['low'],
                close=result_df['close'],
                window=14,
                fillna=True
            )
            
            result_df['adx'] = adx_df['adx']
            result_df['di_plus'] = adx_df['di_plus']
            result_df['di_minus'] = adx_df['di_minus']
            
            # ATR (Average True Range)
            tr1 = result_df['high'] - result_df['low']
            tr2 = abs(result_df['high'] - result_df['close'].shift())
            tr3 = abs(result_df['low'] - result_df['close'].shift())
            result_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result_df['atr'] = result_df['tr'].rolling(window=14).mean()
            
            # ATR percent
            result_df['atr_pct'] = (result_df['atr'] / result_df['close']) * 100
            
            # Volume indicators
            if 'volume' in result_df.columns:
                result_df['volume_sma'] = result_df['volume'].rolling(window=20).mean()
                result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma']
                
                # On-Balance Volume (OBV)
                result_df['obv'] = np.zeros(len(result_df))
                for i in range(1, len(result_df)):
                    if result_df['close'].iloc[i] > result_df['close'].iloc[i-1]:
                        result_df['obv'].iloc[i] = result_df['obv'].iloc[i-1] + result_df['volume'].iloc[i]
                    elif result_df['close'].iloc[i] < result_df['close'].iloc[i-1]:
                        result_df['obv'].iloc[i] = result_df['obv'].iloc[i-1] - result_df['volume'].iloc[i]
                    else:
                        result_df['obv'].iloc[i] = result_df['obv'].iloc[i-1]
            
            # Calculate returns and volatility
            result_df['returns'] = result_df['close'].pct_change()
            result_df['log_returns'] = np.log(result_df['close'] / result_df['close'].shift(1))
            result_df['volatility'] = result_df['log_returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Fill NaN values
            numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns
            result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
            
            return result_df
            
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return df  # Return original dataframe

def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with high, low, close price data
        window: Window size for calculation
        
    Returns:
        Series with ATR values
    """
    try:
        # True Range calculations
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
        
    except Exception as e:
        print(f"Error calculating ATR: {str(e)}")
        return pd.Series(np.nan, index=df.index)

def detect_support_resistance(df, window=20, sensitivity=0.03):
    """
    Detect support and resistance levels
    
    Args:
        df: DataFrame with price data
        window: Window size for detection
        sensitivity: Sensitivity threshold for level detection
        
    Returns:
        Tuple with (support_levels, resistance_levels)
    """
    try:
        # Get highs and lows
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Detect potential resistance levels
        resistance = []
        for i in range(window, len(df) - window):
            if highs.iloc[i] == df['high'].iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                resistance.append(df['high'].iloc[i])
        
        # Detect potential support levels
        support = []
        for i in range(window, len(df) - window):
            if lows.iloc[i] == df['low'].iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                support.append(df['low'].iloc[i])
        
        # Cluster similar levels
        resistance = cluster_levels(resistance, sensitivity)
        support = cluster_levels(support, sensitivity)
        
        return support, resistance
        
    except Exception as e:
        print(f"Error detecting support/resistance: {str(e)}")
        return [], []

def cluster_levels(levels, sensitivity):
    """Group similar price levels together"""
    if not levels:
        return []
        
    levels = sorted(levels)
    clustered = []
    
    current_cluster = [levels[0]]
    for i in range(1, len(levels)):
        # Calculate percentage difference
        if (levels[i] - current_cluster[0]) / current_cluster[0] <= sensitivity:
            current_cluster.append(levels[i])
        else:
            # Calculate average of cluster and add to result
            clustered.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [levels[i]]
    
    # Add the last cluster
    if current_cluster:
        clustered.append(sum(current_cluster) / len(current_cluster))
    
    return clustered
