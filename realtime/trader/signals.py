"""
Signal generation module for analyzing market data and generating trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..utils.indicators import calculate_all_indicators

class SignalGenerator:
    def __init__(self, trader):
        """
        Initialize the SignalGenerator with a reference to the trader
        
        Args:
            trader: The RealtimeTrader instance
        """
        self.trader = trader
        self.client = trader.client
        self.symbol = trader.symbol
        self.use_enhanced_signals = trader.use_enhanced_signals
        self.trend_following_mode = trader.trend_following_mode
        self.signal_confirmation_threshold = trader.signal_confirmation_threshold
        self.signal_cooldown_minutes = trader.signal_cooldown_minutes
        self.use_ml_signals = trader.use_ml_signals
        self.ml_confidence = trader.ml_confidence
    
    def generate_trading_signal(self):
        """
        Generate a trading signal based on market analysis
        
        Returns:
            str: "LONG", "SHORT", or "NEUTRAL"
        """
        # Check signal cooldown
        if self.trader.last_signal_time:
            time_since_last_signal = datetime.now() - self.trader.last_signal_time
            if time_since_last_signal < timedelta(minutes=self.signal_cooldown_minutes):
                print(f"Signal cooldown active. {self.signal_cooldown_minutes - time_since_last_signal.seconds // 60} minutes remaining.")
                return "NEUTRAL"
        
        # Get latest market data
        df = self.trader.account_manager.get_latest_data(lookback_candles=500)
        
        if df is None or len(df) < 100:
            print("Insufficient market data for signal generation")
            return "NEUTRAL"
        
        # Generate signals based on configuration
        if self.use_enhanced_signals:
            return self.generate_enhanced_signal(df)
        elif self.use_ml_signals and self.trader.ml_integration:
            return self.generate_ml_signal(df)
        else:
            return self.generate_basic_signal(df)
    
    def generate_basic_signal(self, df):
        """
        Generate a basic trading signal using simple indicators
        
        Args:
            df: DataFrame with market data
            
        Returns:
            str: "LONG", "SHORT", or "NEUTRAL"
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Simple MACD signal
        macd_signal = "NEUTRAL"
        if "macd" in df.columns and "macd_signal" in df.columns:
            macd = latest["macd"]
            macd_signal_line = latest["macd_signal"]
            
            if macd > macd_signal_line:
                macd_signal = "LONG"
            elif macd < macd_signal_line:
                macd_signal = "SHORT"
        
        # Simple RSI signal
        rsi_signal = "NEUTRAL"
        if "rsi" in df.columns:
            rsi = latest["rsi"]
            
            if rsi < 30:
                rsi_signal = "LONG"
            elif rsi > 70:
                rsi_signal = "SHORT"
        
        # Combine signals
        if macd_signal == rsi_signal and macd_signal != "NEUTRAL":
            return macd_signal
        
        return "NEUTRAL"
    
    def generate_enhanced_signal(self, df):
        """
        Generate an enhanced trading signal using multiple indicators
        
        Args:
            df: DataFrame with market data
            
        Returns:
            str: "LONG", "SHORT", or "NEUTRAL"
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Initialize signal counters
        long_signals = 0
        short_signals = 0
        
        print("\n=== SIGNAL ANALYSIS ===")
        
        # 1. MACD Analysis
        print("\n--- MACD Analysis ---")
        if all(col in df.columns for col in ["macd", "macd_signal", "macd_hist"]):
            macd = latest["macd"]
            macd_signal_line = latest["macd_signal"]
            macd_hist = latest["macd_hist"]
            macd_hist_prev = df.iloc[-2]["macd_hist"]
            
            print(f"DEBUG: MACD: {macd:.4f}, Signal: {macd_signal_line:.4f}, Hist: {macd_hist:.4f}, Prev Hist: {macd_hist_prev:.4f}")
            
            # MACD line crosses above signal line (bullish)
            if macd > macd_signal_line and df.iloc[-2]["macd"] <= df.iloc[-2]["macd_signal"]:
                long_signals += 1
                print("MACD: BULLISH (MACD crossed above signal line) ✅")
            # MACD line crosses below signal line (bearish)
            elif macd < macd_signal_line and df.iloc[-2]["macd"] >= df.iloc[-2]["macd_signal"]:
                short_signals += 1
                print("MACD: BEARISH (MACD crossed below signal line) ✅")
            # MACD histogram turns positive from negative (bullish)
            elif macd_hist > 0 and macd_hist_prev <= 0:
                long_signals += 0.5
                print("MACD: BULLISH (Histogram turned positive) ⚠️")
            # MACD histogram turns negative from positive (bearish)
            elif macd_hist < 0 and macd_hist_prev >= 0:
                short_signals += 0.5
                print("MACD: BEARISH (Histogram turned negative) ⚠️")
            # MACD and signal both above zero (bullish bias)
            elif macd > 0 and macd_signal_line > 0:
                long_signals += 0.3
                print("MACD: BULLISH BIAS (MACD and signal both positive) ℹ️")
            # MACD and signal both below zero (bearish bias)
            elif macd < 0 and macd_signal_line < 0:
                short_signals += 0.3
                print("MACD: BEARISH BIAS (MACD and signal both negative) ℹ️")
            else:
                print("MACD: NEUTRAL (no significant signal) ❌")
        else:
            print("DEBUG: MACD indicators not available in dataframe")
        
        # 2. RSI Analysis
        print("\n--- RSI Analysis ---")
        if "rsi" in df.columns:
            rsi = latest["rsi"]
            rsi_prev = df.iloc[-2]["rsi"]
            
            print(f"DEBUG: RSI: {rsi:.2f}, Previous: {rsi_prev:.2f}")
            
            # Oversold and rising
            if rsi < 30 and rsi > rsi_prev:
                long_signals += 1
                print(f"RSI: BULLISH (Oversold at {rsi:.1f} and rising) ✅")
            # Overbought and falling
            elif rsi > 70 and rsi < rsi_prev:
                short_signals += 1
                print(f"RSI: BEARISH (Overbought at {rsi:.1f} and falling) ✅")
            # Oversold condition
            elif rsi < 30:
                long_signals += 0.5
                print(f"RSI: BULLISH BIAS (Oversold at {rsi:.1f}) ⚠️")
            # Overbought condition
            elif rsi > 70:
                short_signals += 0.5
                print(f"RSI: BEARISH BIAS (Overbought at {rsi:.1f}) ⚠️")
            # Crossing above 50
            elif rsi > 50 and rsi_prev <= 50:
                long_signals += 0.3
                print("RSI: BULLISH BIAS (Crossed above 50) ℹ️")
            # Crossing below 50
            elif rsi < 50 and rsi_prev >= 50:
                short_signals += 0.3
                print("RSI: BEARISH BIAS (Crossed below 50) ℹ️")
            else:
                print(f"RSI: NEUTRAL at {rsi:.1f} ❌")
        else:
            print("DEBUG: RSI indicator not available in dataframe")
        
        # 3. Bollinger Bands Analysis
        print("\n--- Bollinger Bands Analysis ---")
        if all(col in df.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
            close = latest["close"]
            bb_upper = latest["bb_upper"]
            bb_middle = latest["bb_middle"]
            bb_lower = latest["bb_lower"]
            
            print(f"DEBUG: Close: {close:.2f}, Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}")
            
            # Price below lower band (potential bounce)
            if close < bb_lower:
                long_signals += 0.7
                print(f"BB: BULLISH (Price below lower band at {bb_lower:.2f}) ⚠️")
            # Price above upper band (potential reversal)
            elif close > bb_upper:
                short_signals += 0.7
                print(f"BB: BEARISH (Price above upper band at {bb_upper:.2f}) ⚠️")
            # Price crossing above middle band
            elif close > bb_middle and df.iloc[-2]["close"] <= df.iloc[-2]["bb_middle"]:
                long_signals += 0.3
                print("BB: BULLISH BIAS (Price crossed above middle band) ℹ️")
            # Price crossing below middle band
            elif close < bb_middle and df.iloc[-2]["close"] >= df.iloc[-2]["bb_middle"]:
                short_signals += 0.3
                print("BB: BEARISH BIAS (Price crossed below middle band) ℹ️")
            else:
                print("BB: NEUTRAL (Price within normal range) ❌")
        else:
            print("DEBUG: Bollinger Bands indicators not available in dataframe")
        
        # 4. Moving Average Analysis
        print("\n--- Moving Average Analysis ---")
        if all(col in df.columns for col in ["sma_50", "sma_200"]):
            close = latest["close"]
            sma_50 = latest["sma_50"]
            sma_200 = latest["sma_200"]
            
            print(f"DEBUG: Close: {close:.2f}, SMA50: {sma_50:.2f}, SMA200: {sma_200:.2f}")
            
            # Golden Cross (50 crosses above 200)
            if sma_50 > sma_200 and df.iloc[-2]["sma_50"] <= df.iloc[-2]["sma_200"]:
                long_signals += 1
                print("MA: BULLISH (Golden Cross - 50 SMA crossed above 200 SMA) ✅")
            # Death Cross (50 crosses below 200)
            elif sma_50 < sma_200 and df.iloc[-2]["sma_50"] >= df.iloc[-2]["sma_200"]:
                short_signals += 1
                print("MA: BEARISH (Death Cross - 50 SMA crossed below 200 SMA) ✅")
            # Price crosses above 50 SMA
            elif close > sma_50 and df.iloc[-2]["close"] <= df.iloc[-2]["sma_50"]:
                long_signals += 0.5
                print("MA: BULLISH (Price crossed above 50 SMA) ⚠️")
            # Price crosses below 50 SMA
            elif close < sma_50 and df.iloc[-2]["close"] >= df.iloc[-2]["sma_50"]:
                short_signals += 0.5
                print("MA: BEARISH (Price crossed below 50 SMA) ⚠️")
            # Bullish trend (50 > 200)
            elif sma_50 > sma_200:
                long_signals += 0.3
                print("MA: BULLISH BIAS (50 SMA above 200 SMA) ℹ️")
            # Bearish trend (50 < 200)
            elif sma_50 < sma_200:
                short_signals += 0.3
                print("MA: BEARISH BIAS (50 SMA below 200 SMA) ℹ️")
            else:
                print("MA: NEUTRAL ❌")
        else:
            print("DEBUG: Moving Average indicators not available in dataframe")
        
        # 5. Stochastic Analysis
        print("\n--- Stochastic Analysis ---")
        if all(col in df.columns for col in ["stoch_k", "stoch_d"]):
            k_current = latest["stoch_k"]
            d_current = latest["stoch_d"]
            k_prev = df.iloc[-2]["stoch_k"]
            d_prev = df.iloc[-2]["stoch_d"]
            
            print(f"DEBUG: K: {k_current:.2f}, D: {d_current:.2f}, Prev K: {k_prev:.2f}, Prev D: {d_prev:.2f}")
            
            # Bullish crossover in oversold territory
            if k_current > d_current and k_prev <= d_prev and k_current < 20:
                long_signals += 1
                print(f"Stochastic: BULLISH (K crossed above D in oversold territory) ✅")
            # Bearish crossover in overbought territory
            elif k_current < d_current and k_prev >= d_prev and k_current > 80:
                short_signals += 1
                print(f"Stochastic: BEARISH (K crossed below D in overbought territory) ✅")
            else:
                # Add smaller weight for overbought/oversold conditions
                if k_current < 20:
                    print(f"Stochastic: NEUTRAL but OVERSOLD (K at {k_current:.1f}) ⚠️")
                    long_signals += 0.3
                elif k_current > 80:
                    print(f"Stochastic: NEUTRAL but OVERBOUGHT (K at {k_current:.1f}) ⚠️")
                    short_signals += 0.3
                else:
                    print(f"Stochastic: NEUTRAL (no significant signal) ❌")
        else:
            print("DEBUG: Stochastic indicators not available in dataframe")
        
        # 6. ADX for trend strength
        print("\n--- ADX Trend Strength Analysis ---")
        if "adx" in df.columns:
            adx = latest["adx"]
            print(f"DEBUG: ADX value: {adx:.2f}")
            print(f"ADX: {adx:.1f} - {'Strong' if adx > 25 else 'Weak'} trend")
            
            # ADX doesn't give direction, just confirms strength of other signals
            if adx > 25:
                # Strong trend - boost existing signals
                old_long = long_signals
                old_short = short_signals
                long_signals = long_signals * 1.2 if long_signals > short_signals else long_signals
                short_signals = short_signals * 1.2 if short_signals > long_signals else short_signals
                print(f"ADX: Boosting dominant signal due to strong trend (Long: {old_long:.1f} → {long_signals:.1f}, Short: {old_short:.1f} → {short_signals:.1f})")
            else:
                print("ADX: Weak trend, no signal boost")
        else:
            print("DEBUG: ADX indicator not available in dataframe")
        
        # 7. Volume Analysis
        print("\n--- Volume Analysis ---")
        if "volume" in df.columns:
            current_volume = latest["volume"]
            avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            print(f"DEBUG: Current Volume: {current_volume:.2f}, Avg Volume: {avg_volume:.2f}, Ratio: {volume_ratio:.2f}")
            
            # High volume confirms signals
            if volume_ratio > 1.5:
                # Boost the stronger signal
                if long_signals > short_signals:
                    old_long = long_signals
                    long_signals *= 1.2
                    print(f"Volume: High volume confirming BULLISH signal (Long: {old_long:.1f} → {long_signals:.1f})")
                elif short_signals > long_signals:
                    old_short = short_signals
                    short_signals *= 1.2
                    print(f"Volume: High volume confirming BEARISH signal (Short: {old_short:.1f} → {short_signals:.1f})")
                else:
                    print("Volume: High volume but no dominant signal to confirm")
            else:
                print(f"Volume: Normal volume ({volume_ratio:.2f}x average), no signal boost")
        else:
            print("DEBUG: Volume data not available in dataframe")
        
        # 8. Market Trend Analysis
        if self.trend_following_mode:
            print("\n--- Market Trend Analysis ---")
            trend = self.analyze_market_trend(df)
            
            if trend == "BULLISH":
                long_signals += 0.5
                print("Trend: BULLISH - Adding weight to long signals")
            elif trend == "BEARISH":
                short_signals += 0.5
                print("Trend: BEARISH - Adding weight to short signals")
            else:
                print("Trend: NEUTRAL - No additional signal weight")
        
        # Final signal determination
        print("\n=== SIGNAL SUMMARY ===")
        print(f"Long Signals: {long_signals:.2f}")
        print(f"Short Signals: {short_signals:.2f}")
        print(f"Confirmation Threshold: {self.signal_confirmation_threshold}")
        
        if long_signals >= self.signal_confirmation_threshold and long_signals > short_signals:
            print("FINAL SIGNAL: LONG ✅")
            self.trader.last_signal_time = datetime.now()
            return "LONG"
        elif short_signals >= self.signal_confirmation_threshold and short_signals > long_signals:
            print("FINAL SIGNAL: SHORT ✅")
            self.trader.last_signal_time = datetime.now()
            return "SHORT"
        else:
            print("FINAL SIGNAL: NEUTRAL ❌")
            return "NEUTRAL"
    
    def generate_ml_signal(self, df):
        """
        Generate a trading signal using ML predictions
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            str: "LONG", "SHORT", or "NEUTRAL"
        """
        # Calculate indicators for ML
        df = self.calculate_indicators(df)
        
        # Get ML prediction
        prediction, confidence = self.trader.ml_integration.predict(df)
        
        print(f"\n=== ML SIGNAL ANALYSIS ===")
        print(f"ML Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Confidence Threshold: {self.ml_confidence}")
        
        # Check confidence threshold
        if confidence >= self.ml_confidence:
            if prediction > 0:
                print("FINAL SIGNAL: LONG ✅")
                self.trader.last_signal_time = datetime.now()
                return "LONG"
            elif prediction < 0:
                print("FINAL SIGNAL: SHORT ✅")
                self.trader.last_signal_time = datetime.now()
                return "SHORT"
        
        print("FINAL SIGNAL: NEUTRAL ❌")
        return "NEUTRAL"
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for the given DataFrame
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added indicators
        """
        # Use our indicators utility module to calculate all indicators
        return calculate_all_indicators(df)
    
    def analyze_market_trend(self, df=None):
        """
        Analyze the overall market trend
        
        Args:
            df: DataFrame with market data (optional)
            
        Returns:
            str: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if df is None:
            df = self.trader.account_manager.get_latest_data(lookback_candles=500)
            
            if df is None or len(df) < 100:
                return "NEUTRAL"
        
        # Calculate indicators if not already present
        df = self.calculate_indicators(df)
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Trend indicators
        trend_score = 0
        
        # 1. Moving Average Relationship
        if "sma_50" in df.columns and "sma_200" in df.columns:
            if latest["sma_50"] > latest["sma_200"]:
                trend_score += 1
            elif latest["sma_50"] < latest["sma_200"]:
                trend_score -= 1
        
        # 2. Price vs Moving Averages
        if "sma_50" in df.columns:
            if latest["close"] > latest["sma_50"]:
                trend_score += 0.5
            elif latest["close"] < latest["sma_50"]:
                trend_score -= 0.5
        
        # 3. MACD Histogram
        if "macd_hist" in df.columns:
            if latest["macd_hist"] > 0:
                trend_score += 0.5
            elif latest["macd_hist"] < 0:
                trend_score -= 0.5
        
        # 4. RSI
        if "rsi" in df.columns:
            if latest["rsi"] > 50:
                trend_score += 0.3
            elif latest["rsi"] < 50:
                trend_score -= 0.3
        
        # 5. Higher Highs and Higher Lows (for bullish)
        last_10_highs = df["high"].rolling(window=5).max().tail(10)
        last_10_lows = df["low"].rolling(window=5).min().tail(10)
        
        if last_10_highs.is_monotonic_increasing:
            trend_score += 0.7
        if last_10_lows.is_monotonic_increasing:
            trend_score += 0.7
        
        # 6. Lower Highs and Lower Lows (for bearish)
        if last_10_highs.is_monotonic_decreasing:
            trend_score -= 0.7
        if last_10_lows.is_monotonic_decreasing:
            trend_score -= 0.7
        
        # Determine trend based on score
        if trend_score >= 1.5:
            return "BULLISH"
        elif trend_score <= -1.5:
            return "BEARISH"
        else:
            return "NEUTRAL"