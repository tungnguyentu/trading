"""
Safe implementations of technical indicators to avoid common warnings and errors.
"""

import numpy as np
import pandas as pd
import warnings
from ta.trend import ADXIndicator

def safe_adx(high, low, close, window=14, fillna=False):
    """
    Calculate ADX (Average Directional Index) with safeguards against division by zero warnings.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        window: Window size for calculations
        fillna: Whether to fill NA values
    
    Returns:
        DataFrame with ADX, +DI, and -DI columns
    """
    # Temporarily suppress specific RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        try:
            # Create a custom implementation to avoid ta-lib warnings
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window).mean()
            
            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            # Calculate +DM and -DM
            pdm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
            ndm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)
            
            # Smooth +DM and -DM
            pdm_smoothed = pdm.rolling(window).mean()
            ndm_smoothed = ndm.rolling(window).mean()
            
            # Calculate +DI and -DI with zero division protection
            di_plus = pd.Series(np.zeros(len(atr)), index=atr.index)
            di_minus = pd.Series(np.zeros(len(atr)), index=atr.index)
            
            # Calculate only where ATR is not zero
            valid_atr = atr > 0
            di_plus[valid_atr] = 100 * (pdm_smoothed[valid_atr] / atr[valid_atr])
            di_minus[valid_atr] = 100 * (ndm_smoothed[valid_atr] / atr[valid_atr])
            
            # Calculate DX - also with zero division protection
            dx = pd.Series(np.zeros(len(di_plus)), index=di_plus.index)
            di_sum = di_plus + di_minus
            valid_di_sum = di_sum > 0
            dx[valid_di_sum] = 100 * (abs(di_plus[valid_di_sum] - di_minus[valid_di_sum]) / di_sum[valid_di_sum])
            
            # Calculate ADX
            adx = dx.rolling(window).mean()
            
            # Handle NaN values
            if fillna:
                adx = adx.fillna(0)
                di_plus = di_plus.fillna(0)
                di_minus = di_minus.fillna(0)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'adx': adx,
                'di_plus': di_plus,
                'di_minus': di_minus
            }, index=close.index)
            
            return result
            
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            # Return DataFrame with NaN values in case of error
            return pd.DataFrame({
                'adx': np.nan,
                'di_plus': np.nan,
                'di_minus': np.nan
            }, index=close.index)

def calculate_adx(df, window=14, fillna=False):
    """
    Calculate ADX for a DataFrame that contains 'high', 'low', and 'close' columns.
    
    Args:
        df: DataFrame with high, low, and close prices
        window: Window size for calculations
        fillna: Whether to fill NA values
    
    Returns:
        DataFrame with added ADX columns
    """
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        print("Error: DataFrame must contain 'high', 'low', and 'close' columns")
        return df
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Calculate ADX indicators
    adx_result = safe_adx(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=window,
        fillna=fillna
    )
    
    # Add to the result DataFrame
    result_df['adx'] = adx_result['adx']
    result_df['di_plus'] = adx_result['di_plus']
    result_df['di_minus'] = adx_result['di_minus']
    
    return result_df

def safe_rsi(close, window=14, fillna=False):
    """
    Calculate RSI with proper handling of edge cases.
    
    Args:
        close: Series of close prices
        window: Window size for calculations
        fillna: Whether to fill NA values
    
    Returns:
        Series with RSI values
    """
    # Get price differences
    diff = close.diff(1)
    
    # Make a copy of the data
    diff_copy = diff.copy()
    
    # Separate gains and losses
    gains = diff_copy.where(diff_copy > 0, 0.0)
    losses = -diff_copy.where(diff_copy < 0, 0.0)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    
    # Handle division by zero
    avg_loss_nonzero = avg_loss.replace(0, np.nan)
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss_nonzero
    rsi = 100 - (100 / (1 + rs))
    
    # Replace NaN values with 50 (neutral)
    if fillna:
        rsi = rsi.fillna(50)
    
    return rsi

def calculate_all_safe_indicators(df, fillna=True):
    """
    Calculate all safe technical indicators for a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        fillna: Whether to fill NA values
    
    Returns:
        DataFrame with added indicator columns
    """
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        missing = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in df.columns]
        print(f"Warning: Missing required columns: {missing}")
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Calculate ADX indicators
    if all(col in df.columns for col in ['high', 'low', 'close']):
        adx_df = calculate_adx(df, window=14, fillna=fillna)
        result_df['adx'] = adx_df['adx']
        result_df['di_plus'] = adx_df['di_plus']
        result_df['di_minus'] = adx_df['di_minus']
    
    # Calculate RSI
    if 'close' in df.columns:
        result_df['rsi'] = safe_rsi(df['close'], window=14, fillna=fillna)
    
    return result_df
