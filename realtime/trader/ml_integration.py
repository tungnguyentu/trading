"""
ML integration module for using machine learning models in trading.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from ml_models import MLManager
from ..utils.indicators import calculate_all_indicators

class MLIntegration:
    def __init__(self, trader):
        """
        Initialize the MLIntegration with a reference to the trader
        
        Args:
            trader: The RealtimeTrader instance
        """
        self.trader = trader
        self.symbol = trader.symbol
        self.ml_confidence = trader.ml_confidence
        
        # Initialize ML manager
        self.ml_manager = MLManager()
        
        # Load model if it exists
        model, scaler = self.ml_manager.load_model(self.symbol)
        self.model_loaded = model is not None and scaler is not None
        
        if not self.model_loaded and trader.train_ml:
            print("No pre-trained model found. Will train a new model.")
        elif self.model_loaded:
            print(f"ML model loaded for {self.symbol}")
    
    def predict(self, df):
        """
        Generate predictions using the ML model
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            tuple: (prediction, confidence)
                prediction: 1 for LONG, -1 for SHORT, 0 for NEUTRAL
                confidence: Confidence score between 0 and 1
        """
        if not self.model_loaded:
            print("No ML model loaded. Cannot make predictions.")
            # Automatically train a model if one doesn't exist
            print("Attempting to train ML model...")
            success = self.train_ml_model()
            if not success:
                print("Failed to train model. Cannot make predictions.")
                return 0, 0.0
            print("Model trained successfully!")
            
        # Prepare features
        features = self.prepare_features(df)
            
        if features is None:
            print("Failed to prepare features for prediction")
            return 0, 0.0
            
        # Add symbol column to features for ML model
        features['symbol'] = self.symbol
        
        # Get prediction
        try:
            # Use the get_ml_signal method which is designed for prediction
            signal, confidence = self.ml_manager.get_ml_signal(self.symbol, features)
            
            # Convert signal to numeric prediction
            prediction = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
            
            print(f"ML Prediction: {prediction} with confidence {confidence:.2f}")
            return prediction, confidence
            
        except Exception as e:
            print(f"Error making ML prediction: {e}")
            # Try one more time if prediction fails
            print("Attempting to retrain the model...")
            if self.train_ml_model():
                try:
                    signal, confidence = self.ml_manager.get_ml_signal(self.symbol, features)
                    prediction = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
                    print(f"ML Prediction after retraining: {prediction} with confidence {confidence:.2f}")
                    return prediction, confidence
                except Exception as retry_e:
                    print(f"Error making ML prediction after retraining: {retry_e}")
            return 0, 0.0
    
    def train_ml_model(self):
        """
        Train or retrain the ML model
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        print("\n=== Training ML Model ===")
        
        # Get historical data
        df = self.get_training_data()
        
        if df is None or len(df) < 1000:
            print(f"Insufficient data for ML training (needed 1000, got {len(df) if df is not None else 0} records)")
            # Try with less data if that's all we have
            if df is not None and len(df) >= 200:
                print(f"Will attempt training with {len(df)} records instead")
            else:
                return False
        
        # Add symbol column to the dataframe
        df['symbol'] = self.symbol
        
        try:
            # Train the model using the MLManager's train_ml_model method
            model, scaler = self.ml_manager.train_ml_model(self.symbol, df, force_retrain=True)
            
            if model is not None and scaler is not None:
                self.model_loaded = True
                print("ML model training completed")
                return True
            else:
                print("ML model training failed")
                return False
            
        except Exception as e:
            print(f"Error training ML model: {e}")
            return False
    
    def get_training_data(self):
        """
        Get historical data for training
        
        Returns:
            DataFrame: Historical market data
        """
        try:
            # Get data from Binance
            klines = self.trader.client.futures_klines(
                symbol=self.symbol,
                interval="1h",
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ]
            )
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate indicators
            df = calculate_all_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting training data: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            DataFrame: Features for prediction
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure all indicators are calculated
            df = calculate_all_indicators(df)
            
            # Select features
            feature_columns = [
                "macd", "macd_signal", "macd_hist",
                "rsi", "bb_upper", "bb_middle", "bb_lower",
                "sma_50", "sma_200", "stoch_k", "stoch_d",
                "adx", "atr"
            ]
            
            # Add price-based features
            df["close_pct_change"] = df["close"].pct_change()
            df["high_low_diff"] = (df["high"] - df["low"]) / df["close"]
            df["volume_change"] = df["volume"].pct_change()
            
            # Add moving average crossover features
            df["sma_50_200_ratio"] = df["sma_50"] / df["sma_200"]
            df["price_sma_50_ratio"] = df["close"] / df["sma_50"]
            
            # Add these to feature columns
            feature_columns.extend([
                "close_pct_change", "high_low_diff", "volume_change",
                "sma_50_200_ratio", "price_sma_50_ratio"
            ])
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Get the latest data point
            features = df.iloc[-1:][feature_columns]
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None