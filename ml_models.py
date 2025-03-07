import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import joblib
import traceback

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pickle

logger = logging.getLogger(__name__)


class MLManager:
    def __init__(self, models_dir=os.path.join(os.getcwd(), "models")):
        """Initialize ML Manager with directory for saving models"""
        self.models_dir = models_dir

        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Dictionary to store models and scalers in memory
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.model_metrics = {}

        # Load any existing models
        self.load_all_models()

    def model_path(self, symbol):
        """Get path for model file"""
        return os.path.join(self.models_dir, f"{symbol}_model.pkl")

    def scaler_path(self, symbol):
        """Get path for scaler file"""
        return os.path.join(self.models_dir, f"{symbol}_scaler.pkl")

    def metrics_path(self, symbol):
        """Get path for model metrics file"""
        return os.path.join(self.models_dir, f"{symbol}_metrics.pkl")

    def save_model(self, symbol, model, scaler, metrics=None):
        """Save model and scaler to disk"""
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)

            # Save model
            with open(self.model_path(symbol), "wb") as f:
                pickle.dump(model, f)

            # Save scaler
            with open(self.scaler_path(symbol), "wb") as f:
                pickle.dump(scaler, f)

            # Save metrics if provided
            if metrics:
                with open(self.metrics_path(symbol), "wb") as f:
                    pickle.dump(metrics, f)

            print(f"Model and scaler saved for {symbol}")

            # Store feature importances if available
            if hasattr(model, "feature_importances_"):
                self.feature_importances[symbol] = model.feature_importances_

        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, symbol):
        """Load model and scaler from disk"""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path(symbol)) or not os.path.exists(
                self.scaler_path(symbol)
            ):
                print(f"Model or scaler not found for {symbol}")
                return None, None

            # Load model
            with open(self.model_path(symbol), "rb") as f:
                model = pickle.load(f)

            # Load scaler
            with open(self.scaler_path(symbol), "rb") as f:
                scaler = pickle.load(f)

            # Load metrics if available
            if os.path.exists(self.metrics_path(symbol)):
                with open(self.metrics_path(symbol), "rb") as f:
                    self.model_metrics[symbol] = pickle.load(f)

            # Store in memory
            self.models[symbol] = model
            self.scalers[symbol] = scaler

            # Store feature importances if available
            if hasattr(model, "feature_importances_"):
                self.feature_importances[symbol] = model.feature_importances_

            print(f"Model and scaler loaded for {symbol}")
            return model, scaler

        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def load_all_models(self):
        """Load all models from disk"""
        try:
            # Get all model files
            model_files = [
                f for f in os.listdir(self.models_dir) if f.endswith("_model.pkl")
            ]

            # Extract symbols
            symbols = [f.split("_model.pkl")[0] for f in model_files]

            # Load each model
            for symbol in symbols:
                self.load_model(symbol)

            print(f"Loaded {len(symbols)} models")

        except Exception as e:
            print(f"Error loading models: {e}")

    def add_features(self, df):
        """Add technical indicators and features to the dataframe"""
        try:
            # Make a copy to avoid modifying the original dataframe
            df_feat = df.copy()

            # Ensure all required columns exist
            required_columns = ["open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df_feat.columns:
                    print(f"Warning: Required column '{col}' not found in dataframe")

            # Ensure all basic SMAs are calculated
            df_feat = self._ensure_basic_smas(df_feat)

            # Calculate returns if not already present
            if "returns" not in df_feat.columns and "close" in df_feat.columns:
                df_feat["returns"] = df_feat["close"].pct_change()

            # Calculate log returns if not already present
            if "log_returns" not in df_feat.columns and "close" in df_feat.columns:
                df_feat["log_returns"] = np.log(
                    df_feat["close"] / df_feat["close"].shift(1)
                )

            # Calculate ATR if not already present
            if "atr" not in df_feat.columns and all(
                col in df_feat.columns for col in ["high", "low", "close"]
            ):
                df_feat["atr"] = self._calculate_atr(df_feat)
                if "close" in df_feat.columns:
                    df_feat["atr_ratio"] = df_feat["atr"] / df_feat["close"]

            # Calculate RSI if not already present
            if "rsi_14" not in df_feat.columns and "close" in df_feat.columns:
                df_feat["rsi_14"] = self._calculate_rsi(df_feat, 14)

            # Calculate MACD if not already present
            if "macd" not in df_feat.columns and "close" in df_feat.columns:
                ema_12 = df_feat["close"].ewm(span=12, adjust=False).mean()
                ema_26 = df_feat["close"].ewm(span=26, adjust=False).mean()
                df_feat["macd"] = ema_12 - ema_26
                df_feat["macd_signal"] = (
                    df_feat["macd"].ewm(span=9, adjust=False).mean()
                )
                df_feat["macd_hist"] = df_feat["macd"] - df_feat["macd_signal"]

            # Calculate Bollinger Bands if not already present
            if "bb_width_20" not in df_feat.columns and "close" in df_feat.columns:
                if "sma_20" not in df_feat.columns:
                    df_feat["sma_20"] = df_feat["close"].rolling(window=20).mean()

                std_20 = df_feat["close"].rolling(window=20).std()
                df_feat["bb_upper_20"] = df_feat["sma_20"] + (std_20 * 2)
                df_feat["bb_lower_20"] = df_feat["sma_20"] - (std_20 * 2)
                df_feat["bb_width_20"] = (
                    df_feat["bb_upper_20"] - df_feat["bb_lower_20"]
                ) / df_feat["sma_20"]
                df_feat["bb_position_20"] = (
                    df_feat["close"] - df_feat["bb_lower_20"]
                ) / (df_feat["bb_upper_20"] - df_feat["bb_lower_20"])

            # Calculate volume indicators if not already present
            if "volume_ratio" not in df_feat.columns and "volume" in df_feat.columns:
                df_feat["volume_ma_5"] = (
                    df_feat["volume"].astype(float).rolling(window=5).mean()
                )
                df_feat["volume_ratio"] = (
                    df_feat["volume"].astype(float) / df_feat["volume_ma_5"]
                )

            # Drop NaN values
            df_feat = df_feat.dropna()

            return df_feat

        except Exception as e:
            print(f"Error adding features: {str(e)}")
            # Return original dataframe if there's an error
            return df

    def _ensure_basic_smas(self, df_feat):
        """Ensure all basic SMA features are calculated"""
        # Print available columns for debugging
        print(f"Available columns: {df_feat.columns.tolist()}")

        # Check if we have SMA_short and SMA_long and convert them to standard format
        if "SMA_short" in df_feat.columns and "sma_10" not in df_feat.columns:
            print("Converting SMA_short to sma_10")
            df_feat["sma_10"] = df_feat["SMA_short"]

            # Calculate sma_ratio_10
            if "close" in df_feat.columns:
                df_feat["sma_ratio_10"] = df_feat["close"] / df_feat["sma_10"]

        if "SMA_long" in df_feat.columns and "sma_20" not in df_feat.columns:
            print("Converting SMA_long to sma_20")
            df_feat["sma_20"] = df_feat["SMA_long"]

            # Calculate sma_ratio_20
            if "close" in df_feat.columns:
                df_feat["sma_ratio_20"] = df_feat["close"] / df_feat["sma_20"]

        # Calculate basic SMAs if they don't exist
        if "close" in df_feat.columns:
            sma_periods = [5, 10, 20, 50, 100, 200]
            for period in sma_periods:
                sma_col = f"sma_{period}"
                ratio_col = f"sma_ratio_{period}"

                if sma_col not in df_feat.columns:
                    print(f"Calculating {sma_col}")
                    df_feat[sma_col] = df_feat["close"].rolling(window=period).mean()

                if ratio_col not in df_feat.columns and sma_col in df_feat.columns:
                    print(f"Calculating {ratio_col}")
                    df_feat[ratio_col] = df_feat["close"] / df_feat[sma_col]

        # Copy ATR to atr if it exists
        if "ATR" in df_feat.columns and "atr" not in df_feat.columns:
            print("Converting ATR to atr")
            df_feat["atr"] = df_feat["ATR"]

            # Calculate atr_ratio
            if "close" in df_feat.columns:
                df_feat["atr_ratio"] = df_feat["atr"] / df_feat["close"]

        # Calculate returns and log_returns if they don't exist
        if "close" in df_feat.columns:
            if "returns" not in df_feat.columns:
                print("Calculating returns")
                df_feat["returns"] = df_feat["close"].pct_change()

            if "log_returns" not in df_feat.columns:
                print("Calculating log_returns")
                df_feat["log_returns"] = np.log(
                    df_feat["close"] / df_feat["close"].shift(1)
                )

        # Calculate volume_ratio if it doesn't exist
        if "volume" in df_feat.columns and "volume_ratio" not in df_feat.columns:
            print("Calculating volume_ratio")
            df_feat["volume_ma_5"] = (
                df_feat["volume"].astype(float).rolling(window=5).mean()
            )
            df_feat["volume_ratio"] = (
                df_feat["volume"].astype(float) / df_feat["volume_ma_5"]
            )

        # Calculate RSI if it doesn't exist
        if "close" in df_feat.columns and "rsi_14" not in df_feat.columns:
            print("Calculating RSI")
            delta = df_feat["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_feat["rsi_14"] = 100 - (100 / (1 + rs))

        # Calculate MACD if it doesn't exist
        if "close" in df_feat.columns and "macd" not in df_feat.columns:
            print("Calculating MACD")
            ema12 = df_feat["close"].ewm(span=12, adjust=False).mean()
            ema26 = df_feat["close"].ewm(span=26, adjust=False).mean()
            df_feat["macd"] = ema12 - ema26
            df_feat["macd_signal"] = df_feat["macd"].ewm(span=9, adjust=False).mean()
            df_feat["macd_hist"] = df_feat["macd"] - df_feat["macd_signal"]

        # Calculate Bollinger Bands if they don't exist
        if "close" in df_feat.columns and "bb_upper_20" not in df_feat.columns:
            print("Calculating Bollinger Bands")
            if "sma_20" not in df_feat.columns:
                df_feat["sma_20"] = df_feat["close"].rolling(window=20).mean()

            std_20 = df_feat["close"].rolling(window=20).std()
            df_feat["bb_upper_20"] = df_feat["sma_20"] + (std_20 * 2)
            df_feat["bb_lower_20"] = df_feat["sma_20"] - (std_20 * 2)
            df_feat["bb_width_20"] = (
                df_feat["bb_upper_20"] - df_feat["bb_lower_20"]
            ) / df_feat["sma_20"]
            df_feat["bb_position_20"] = (df_feat["close"] - df_feat["bb_lower_20"]) / (
                df_feat["bb_upper_20"] - df_feat["bb_lower_20"]
            )

        return df_feat

    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr

    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] - df["volume"].iloc[i]
            else:
                obv[i] = obv[i - 1]

        return pd.Series(obv, index=df.index)

    def _calculate_rsi(self, df, window=14):
        """Calculate Relative Strength Index"""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Handle division by zero
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rs = rs.replace(np.nan, 0)

        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, df, window=14):
        """Calculate Stochastic Oscillator %K"""
        low_min = df["low"].rolling(window=window).min()
        high_max = df["high"].rolling(window=window).max()

        # Handle division by zero
        denom = high_max - low_min
        denom = denom.replace(0, np.nan)

        stoch_k = 100 * ((df["close"] - low_min) / denom)
        stoch_k = stoch_k.replace(np.nan, 50)  # Default to middle value when undefined

        return stoch_k

    def _add_ichimoku(self, df):
        """Add Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = df["high"].rolling(window=9).max()
        nine_period_low = df["low"].rolling(window=9).min()
        df["tenkan_sen"] = (nine_period_high + nine_period_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df["high"].rolling(window=26).max()
        period26_low = df["low"].rolling(window=26).min()
        df["kijun_sen"] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df["high"].rolling(window=52).max()
        period52_low = df["low"].rolling(window=52).min()
        df["senkou_span_b"] = ((period52_high + period52_low) / 2).shift(26)

        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        df["chikou_span"] = df["close"].shift(-26)

        # Cloud strength and direction
        df["cloud_strength"] = abs(df["senkou_span_a"] - df["senkou_span_b"]) / (
            (df["senkou_span_a"] + df["senkou_span_b"]) / 2
        )
        df["above_cloud"] = (
            (df["close"] > df["senkou_span_a"]) & (df["close"] > df["senkou_span_b"])
        ).astype(int)
        df["below_cloud"] = (
            (df["close"] < df["senkou_span_a"]) & (df["close"] < df["senkou_span_b"])
        ).astype(int)
        df["in_cloud"] = (
            ~df["above_cloud"].astype(bool) & ~df["below_cloud"].astype(bool)
        ).astype(int)

        return df

    def _add_adx(self, df, window=14):
        """Add Average Directional Index"""
        # True Range
        df["tr1"] = abs(df["high"] - df["low"])
        df["tr2"] = abs(df["high"] - df["close"].shift())
        df["tr3"] = abs(df["low"] - df["close"].shift())
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr_adx"] = df["tr"].rolling(window=window).mean()

        # Plus Directional Movement (+DM)
        df["plus_dm"] = 0.0
        df.loc[
            (df["high"] - df["high"].shift() > df["low"].shift() - df["low"])
            & (df["high"] - df["high"].shift() > 0),
            "plus_dm",
        ] = (
            df["high"] - df["high"].shift()
        )

        # Minus Directional Movement (-DM)
        df["minus_dm"] = 0.0
        df.loc[
            (df["low"].shift() - df["low"] > df["high"] - df["high"].shift())
            & (df["low"].shift() - df["low"] > 0),
            "minus_dm",
        ] = (
            df["low"].shift() - df["low"]
        )

        # Smooth +DM and -DM with handling for zero ATR
        df["plus_di"] = 0.0  # Initialize with zeros
        df["minus_di"] = 0.0  # Initialize with zeros
        
        # Calculate DI+ and DI- only where ATR is non-zero
        mask = df["atr_adx"] > 0
        df.loc[mask, "plus_di"] = 100 * (
            df.loc[mask, "plus_dm"].rolling(window=window).mean() / df.loc[mask, "atr_adx"]
        )
        df.loc[mask, "minus_di"] = 100 * (
            df.loc[mask, "minus_dm"].rolling(window=window).mean() / df.loc[mask, "atr_adx"]
        )

        # Directional Movement Index (DX) with handling for zero denominator
        df["dx"] = 0.0  # Initialize with zeros
        di_sum = df["plus_di"] + df["minus_di"]
        mask = di_sum > 0  # Only calculate where sum is non-zero
        df.loc[mask, "dx"] = 100 * (
            abs(df.loc[mask, "plus_di"] - df.loc[mask, "minus_di"]) / di_sum[mask]
        )

        # Average Directional Index (ADX)
        df["adx"] = df["dx"].rolling(window=window).mean()

        return df

    def train_ml_model(self, symbol, df, force_retrain=False):
        """Train ML model for a specific symbol"""
        try:
            print(f"\n--- Training ML model for {symbol} ---")

            # Check if model already exists and we're not forcing a retrain
            if not force_retrain and symbol in self.models and symbol in self.scalers:
                print(
                    f"Model for {symbol} already exists. Use force_retrain=True to retrain."
                )
                return self.models[symbol], self.scalers[symbol]

            # Add features
            df_features = self.add_features(df)

            # Print available columns after feature addition
            print(
                f"Available columns after feature addition: {df_features.columns.tolist()}"
            )

            # Check if we have enough data
            if len(df_features) < 100:
                print(
                    f"Not enough data to train model for {symbol}. Need at least 50 rows, got {len(df_features)}."
                )
                return None, None

            # Define target - predict if next candle will be up (1) or down (0)
            if "target" not in df_features.columns:
                print("Creating target variable (next candle direction)")
                df_features["target"] = (
                    df_features["close"].shift(-1) > df_features["close"]
                ).astype(int)

            # Drop rows with NaN values
            df_features = df_features.dropna()

            # Define features - use what's available
            available_columns = df_features.columns.tolist()

            # Basic feature set - prioritize these if available
            basic_features = [
                "returns",
                "log_returns",
                "sma_ratio_10",
                "sma_ratio_20",
                "sma_ratio_50",
                "atr_ratio",
                "volume_ratio",
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_width_20",
                "bb_position_20",
            ]

            # Filter to only include columns that exist in the dataframe
            feature_columns = [
                col for col in basic_features if col in available_columns
            ]

            # If we don't have enough features, use whatever is available
            if len(feature_columns) < 5:
                print(
                    f"Warning: Only {len(feature_columns)} basic features available. Adding additional columns."
                )

                # Add any numeric columns that aren't the target or timestamp
                exclude_cols = [
                    "target",
                    "timestamp",
                    "date",
                    "time",
                    "symbol",
                    "target_2day",
                    "target_3day",
                    "target_5day",
                ]

                additional_features = [
                    col
                    for col in available_columns
                    if col not in exclude_cols
                    and col not in feature_columns
                    and df_features[col].dtype in ["float64", "int64"]
                ]

                # Add up to 10 additional features
                feature_columns.extend(additional_features[:10])
                print(f"Added {len(additional_features[:10])} additional features")

            print(f"Using {len(feature_columns)} features for training")
            print(f"Features: {feature_columns}")

            # Check if we have any features with inconsistent naming
            # This helps prevent the "Feature names unseen at fit time" error
            standardized_features = []
            feature_mapping = {}

            for feature in feature_columns:
                # Standardize feature names to lowercase with underscores
                std_feature = feature.lower().replace(" ", "_")

                # Map original feature to standardized name
                feature_mapping[feature] = std_feature
                standardized_features.append(std_feature)

                # Create a copy of the feature with standardized name if it doesn't match
                if std_feature != feature:
                    print(f"Standardizing feature name: {feature} -> {std_feature}")
                    df_features[std_feature] = df_features[feature]

            # Use standardized feature names for training
            X = df_features[feature_columns].copy()
            X.columns = standardized_features
            y = df_features["target"]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            print("Training RandomForestClassifier...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Model evaluation:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

            # Save model metrics
            self.model_metrics[symbol] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "feature_names": standardized_features,  # Store standardized feature names
                "feature_mapping": feature_mapping,  # Store mapping for future reference
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_points": len(df_features),
            }

            # Save model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler

            # Save model to disk
            self.save_model(symbol, model, scaler)

            return model, scaler

        except Exception as e:
            print(f"Error training ML model: {str(e)}")
            traceback.print_exc()
            return None, None

    def train_advanced_model(self, symbol, df, force_retrain=False, model_type="ensemble", optimize_hyperparams=True):
        """
        Train an advanced ML model with optional hyperparameter optimization
        
        Args:
            symbol: Trading symbol
            df: DataFrame with historical data
            force_retrain: Whether to force retraining even if model exists
            model_type: Type of model to train ("rf", "gb", "ensemble")
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            tuple: (model, scaler) - The trained model and feature scaler
        """
        try:
            print(f"\n--- Training Advanced ML model for {symbol} ---")
            
            # Check if model already exists and we're not forcing a retrain
            if not force_retrain and symbol in self.models and symbol in self.scalers:
                print(f"Model for {symbol} already exists. Use force_retrain=True to retrain.")
                return self.models[symbol], self.scalers[symbol]
                
            # Add features
            df_features = self.add_features(df)
            
            # Create target variable - future price direction
            # For more comprehensive training, we'll use multiple timeframes
            print("Creating target variables for multiple timeframes")
            
            # 1-candle ahead prediction (next candle direction)
            df_features['target_1'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
            
            # 3-candles ahead prediction
            df_features['target_3'] = (df_features['close'].shift(-3) > df_features['close']).astype(int)
            
            # 5-candles ahead prediction
            df_features['target_5'] = (df_features['close'].shift(-5) > df_features['close']).astype(int)
            
            # Primary target - we'll use the 1-candle prediction as default
            df_features['target'] = df_features['target_1']
            
            # Drop NaN values
            df_features = df_features.dropna()
            print(f"Data shape after feature engineering: {df_features.shape}")
            
            # Define features
            excluded_columns = ['target', 'target_1', 'target_3', 'target_5', 'timestamp', 
                              'date', 'time', 'symbol', 'open_time', 'close_time']
            feature_columns = [col for col in df_features.columns 
                             if col not in excluded_columns 
                             and df_features[col].dtype in ['float64', 'int64']]
            
            print(f"Using {len(feature_columns)} features")
            
            # Split data into training and testing sets
            # Use time series split to respect chronological order
            X = df_features[feature_columns]
            y = df_features['target']
            
            # Use the last 20% of data for testing
            train_size = int(len(df_features) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model selection based on type
            if model_type == "rf":
                print("Training Random Forest model...")
                
                if optimize_hyperparams:
                    # Hyperparameter optimization for Random Forest
                    print("Optimizing hyperparameters...")
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    
                    grid_search = GridSearchCV(
                        RandomForestClassifier(random_state=42),
                        param_grid,
                        cv=TimeSeriesSplit(n_splits=3),
                        scoring='f1',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    print(f"Best parameters: {grid_search.best_params_}")
                    model = grid_search.best_estimator_
                else:
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
            elif model_type == "gb":
                print("Training Gradient Boosting model...")
                
                if optimize_hyperparams:
                    # Hyperparameter optimization for Gradient Boosting
                    print("Optimizing hyperparameters...")
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 4, 5],
                        'min_samples_split': [2, 5],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    
                    grid_search = GridSearchCV(
                        GradientBoostingClassifier(random_state=42),
                        param_grid,
                        cv=TimeSeriesSplit(n_splits=3),
                        scoring='f1',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    print(f"Best parameters: {grid_search.best_params_}")
                    model = grid_search.best_estimator_
                else:
                    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
                    model.fit(X_train_scaled, y_train)
            
            else:  # ensemble
                print("Training Ensemble model (Voting Classifier)...")
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
                
                model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb)],
                    voting='soft'
                )
                model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"Model performance on test set:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # Classification report
            report = classification_report(y_test, y_pred)
            print("\nClassification Report:")
            print(report)
            
            # Run backtest evaluation
            backtest_result = self.backtest_model(
                model, 
                scaler, 
                df_features.iloc[train_size:].copy(), 
                feature_columns
            )
            
            # Create and save metrics
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'feature_names': feature_columns,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_points': len(df_features),
                'model_type': model_type,
                'backtest_results': backtest_result
            }
            
            # Store model and metrics
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.model_metrics[symbol] = metrics
            
            # Save model to disk
            self.save_model(symbol, model, scaler, metrics)
            
            # Plot feature importance if available
            if hasattr(model, 'feature_importances_'):
                self._plot_feature_importance(symbol, model, feature_columns)
            elif hasattr(model, 'estimators_'):
                for i, estimator in enumerate(model.estimators_):
                    if hasattr(estimator, 'feature_importances_'):
                        self._plot_feature_importance(
                            f"{symbol}_{estimator.__class__.__name__}_{i}", 
                            estimator, 
                            feature_columns
                        )
            
            return model, scaler
            
        except Exception as e:
            print(f"Error training advanced ML model: {str(e)}")
            traceback.print_exc()
            return None, None

    def backtest_model(self, model, scaler, df, feature_columns, initial_balance=10000, position_size_pct=0.2, retrain_interval=0, retrain_func=None):
        """
        Backtest a trained model
        
        Args:
            model: Trained ML model
            scaler: Feature scaler
            df: DataFrame with features
            feature_columns: List of feature columns
            initial_balance: Initial balance for backtesting
            position_size_pct: Position size as percentage of balance
            retrain_interval: Interval in hours to retrain the model (0 to disable)
            retrain_func: Function to call for retraining the model
            
        Returns:
            dict: Backtest results
        """
        print("Backtesting model...")
        
        # Initialize variables
        balance = initial_balance
        position = None
        entry_price = 0
        position_size = 0
        trades = []
        equity = [initial_balance]
        equity_timestamps = [df.index[0]]
        last_retrain_time = None
        retrain_count = 0
        
        # Ensure df is sorted by date
        df = df.sort_index()
        
        # Get expected number of features from scaler
        n_expected_features = scaler.n_features_in_
        print(f"Model expects {n_expected_features} features, found {len(feature_columns)} valid feature columns")
        
        # Validate feature columns - make sure all features exist in the dataframe
        valid_feature_columns = [col for col in feature_columns if col in df.columns]
        if len(valid_feature_columns) != len(feature_columns):
            print(f"Warning: {len(feature_columns) - len(valid_feature_columns)} feature columns not found in dataframe")
            print(f"Missing: {[col for col in feature_columns if col not in df.columns]}")
        
        # Get the actual feature names used during training
        feature_names_at_fit = None
        
        # Try to get the feature names from the scaler
        if hasattr(scaler, 'feature_names_in_'):
            feature_names_at_fit = scaler.feature_names_in_
            print(f"Feature names from scaler: {feature_names_at_fit}")
            
            # Create a mapping between model features and available columns
            feature_name_mapping = self._create_feature_name_mapping(feature_names_at_fit, df.columns)
            
            # Check how many features we were able to map
            if len(feature_name_mapping) < len(feature_names_at_fit):
                print(f"Warning: Only {len(feature_name_mapping)} out of {len(feature_names_at_fit)} features could be mapped")
                print(f"Missing features: {[f for f in feature_names_at_fit if f not in feature_name_mapping]}")
        else:
            print("Warning: Scaler does not have feature_names_in_ attribute. Using original feature columns.")
            if len(valid_feature_columns) > n_expected_features:
                valid_feature_columns = valid_feature_columns[:n_expected_features]
            feature_name_mapping = {col: col for col in valid_feature_columns}
        
        # Slice df to have the necessary minimum lookback
        start_idx = 20  # Default to a reasonable lookback
        
        # Get predictions for each candle
        for i in range(start_idx, len(df)):
            current_time = df.index[i]
            current_row = df.iloc[i]
            
            # Get current price
            current_price = current_row['close']
            
            # Handle retraining if enabled
            if retrain_interval > 0 and retrain_func is not None:
                if last_retrain_time is None or (current_time - last_retrain_time).total_seconds() / 3600 >= retrain_interval:
                    print(f"Retraining model at {current_time}")
                    model, scaler = retrain_func()
                    last_retrain_time = current_time
                    retrain_count += 1
                    
                    # Update feature names if they changed after retraining
                    if hasattr(scaler, 'feature_names_in_'):
                        feature_names_at_fit = scaler.feature_names_in_
                        feature_name_mapping = self._create_feature_name_mapping(feature_names_at_fit, df.columns)
            
            try:
                if feature_names_at_fit is not None:
                    # Use the exact feature names the model was trained with
                    features_to_use = {}
                    for name in feature_names_at_fit:
                        if name in feature_name_mapping:
                            mapped_name = feature_name_mapping[name]
                            features_to_use[name] = current_row[mapped_name]
                        else:
                            # If a feature is missing, use a reasonable default value (0)
                            features_to_use[name] = 0.0
                            
                    # Create a DataFrame with the exact feature names expected by the model
                    X = pd.DataFrame([features_to_use], columns=feature_names_at_fit)
                else:
                    # Fallback to using a numpy array if feature names aren't available
                    X = current_row[valid_feature_columns].values.reshape(1, -1)
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Get prediction
                pred = model.predict(X_scaled)[0]
                # Get prediction probability
                pred_proba = model.predict_proba(X_scaled)[0]
                
                # Extract confidence score
                if pred == 1:  # Buy signal
                    confidence = pred_proba[1]  # Probability of class 1
                else:  # Sell signal (0 or -1)
                    confidence = pred_proba[0]  # Probability of class 0
            except Exception as e:
                print(f"Error during prediction at index {i}: {e}")
                continue
            
            # Check if we should close position based on prediction
            if position == 'long' and pred == 0:
                # Close long position
                profit = position_size * (current_price - entry_price)
                balance += profit
                trades.append({
                    'type': 'long',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'entry_date': entry_date,
                    'exit_date': current_time,
                    'position_size': position_size,
                    'profit_amount': profit,
                    'profit_pct': (current_price / entry_price - 1) * 100,
                    'balance': balance
                })
                position = None
                
            elif position == 'short' and pred == 1:
                # Close short position
                profit = position_size * (entry_price - current_price)
                balance += profit
                trades.append({
                    'type': 'short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'entry_date': entry_date,
                    'exit_date': current_time,
                    'position_size': position_size,
                    'profit_amount': profit,
                    'profit_pct': (entry_price / current_price - 1) * 100,
                    'balance': balance
                })
                position = None
            
            # Check if we should open a new position
            if position is None and confidence >= 0.55:  # Only enter with sufficient confidence
                if pred == 1:  # Buy signal
                    # Open long position
                    position = 'long'
                    entry_price = current_price
                    position_size = balance * position_size_pct / current_price
                    entry_date = current_time
                    
                elif pred == 0:  # Sell signal
                    # Open short position
                    position = 'short'
                    entry_price = current_price
                    position_size = balance * position_size_pct / current_price
                    entry_date = current_time
            
            # Update equity curve
            equity.append(balance)
            equity_timestamps.append(current_time)
        
        # Close any open position at the end
        if position == 'long':
            profit = position_size * (current_price - entry_price)
            balance += profit
            trades.append({
                'type': 'long',
                'entry_price': entry_price,
                'exit_price': current_price,
                'entry_date': entry_date,
                'exit_date': current_time,
                'position_size': position_size,
                'profit_amount': profit,
                'profit_pct': (current_price / entry_price - 1) * 100,
                'balance': balance
            })
            
        elif position == 'short':
            profit = position_size * (entry_price - current_price)
            balance += profit
            trades.append({
                'type': 'short',
                'entry_price': entry_price,
                'exit_price': current_price,
                'entry_date': entry_date,
                'exit_date': current_time,
                'position_size': position_size,
                'profit_amount': profit,
                'profit_pct': (entry_price / current_price - 1) * 100,
                'balance': balance
            })
        
        # Calculate results
        win_trades = sum(1 for trade in trades if trade['profit_amount'] > 0)
        loss_trades = sum(1 for trade in trades if trade['profit_amount'] < 0)
        
        if len(trades) > 0:
            win_rate = win_trades / len(trades) * 100
        else:
            win_rate = 0
        
        # Calculate profit factor
        total_profit = sum(trade['profit_amount'] for trade in trades if trade['profit_amount'] > 0)
        total_loss = sum(abs(trade['profit_amount']) for trade in trades if trade['profit_amount'] < 0)
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = float('inf') if total_profit > 0 else 0
        
        # Calculate maximum drawdown
        max_balance = initial_balance
        max_drawdown = 0
        
        for i, trade_balance in enumerate(equity):
            if trade_balance > max_balance:
                max_balance = trade_balance
            drawdown = (max_balance - trade_balance) / max_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        # Convert to percentage
        max_drawdown_pct = max_drawdown * 100
        
        # Calculate total return
        if initial_balance > 0:
            total_return_pct = (balance / initial_balance - 1) * 100
        else:
            total_return_pct = 0
        
        # Save results
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return_pct,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades),
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'trades': trades
        }
        
        # Add retraining information if used
        if retrain_interval > 0:
            results['retrain_interval'] = retrain_interval
            results['retrain_count'] = retrain_count
        
        return results

    def _plot_equity_curve(self, equity_curve, trades):
        """Plot equity curve and trades"""
        try:
            dates, equity = zip(*equity_curve)
            plt.figure(figsize=(14, 7))
            plt.plot(dates, equity, label='Equity Curve')
            # Mark trades on equity curve
            for trade in trades:
                if trade['profit_amount'] > 0:
                    color = 'green'
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'v'
                
                # Find index of trade exit in equity curve
                exit_idx = next((i for i, (date, _) in enumerate(equity_curve) if date >= trade['exit_date']), None)
                if exit_idx is not None:
                    plt.scatter(dates[exit_idx], equity[exit_idx], color=color, marker=marker, s=50)
            
            plt.title('Backtest Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            plt.savefig('backtest_equity_curve.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting equity curve: {e}")

    def _plot_feature_importance(self, symbol, model, feature_names):
        """Plot feature importance"""
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance for {symbol}")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.models_dir, f"{symbol}_feature_importance.png"))
        plt.close()

    def get_ml_signal(self, symbol, df):
        """Get trading signal from ML model"""
        try:
            print(f"\n--- ML Signal Analysis for {symbol} ---")

            # Check if model exists in memory, try to load from disk if not
            if symbol not in self.models or symbol not in self.scalers:
                model, scaler = self.load_model(symbol)

                # If still not available, train a new model
                if model is None or scaler is None:
                    print("No existing model found. Training new model...")
                    model, scaler = self.train_ml_model(symbol, df)

            # Add features
            df_features = self.add_features(df)

            # Get latest data point
            latest_data = df_features.iloc[-1:]

            # Get the feature names used during training
            if hasattr(self.scalers[symbol], 'feature_names_in_'):
                feature_names_at_fit = self.scalers[symbol].feature_names_in_
                print(f"Using feature names from scaler: {feature_names_at_fit}")
                
                # Create mapping between model features and available columns
                feature_name_mapping = self._create_feature_name_mapping(feature_names_at_fit, df_features.columns)
                
                # Check if we have enough mapped features
                if len(feature_name_mapping) >= len(feature_names_at_fit) * 0.7:  # At least 70% of features should be mapped
                    print(f"{len(feature_name_mapping)} out of {len(feature_names_at_fit)} features could be mapped")
                    
                    # Create X with mapped features
                    X = pd.DataFrame(index=latest_data.index)
                    for name in feature_names_at_fit:
                        if name in feature_name_mapping:
                            mapped_name = feature_name_mapping[name]
                            X[name] = latest_data[mapped_name]
                        else:
                            # Use 0 as default for missing features
                            X[name] = 0.0
                    
                    # Scale features
                    X_scaled = self.scalers[symbol].transform(X)
                    
                    # Predict
                    prediction = self.models[symbol].predict(X_scaled)[0]
                    probability = self.models[symbol].predict_proba(X_scaled)[0]
                    
                    # Print prediction details
                    print(f"Prediction: {'BUY' if prediction else 'SELL'}")
                    print(f"Confidence: {max(probability):.2f}")
                    
                    # Print feature values used for prediction
                    print("\nFeature values used:")
                    for name in feature_names_at_fit:
                        if name in feature_name_mapping:
                            value = latest_data[feature_name_mapping[name]].values[0]
                            print(f"  {name} (mapped from {feature_name_mapping[name]}): {value:.4f}")
                        else:
                            print(f"  {name}: 0.0000 (default value)")
                    
                    # Convert to signal
                    if prediction:
                        return 1, probability[1]  # Buy signal with probability
                    else:
                        return -1, probability[0]  # Sell signal with probability
            
            # Fall back to original method if feature_names_in_ is not available or mapping failed
            # Define features - must match those used in training
            if symbol in self.model_metrics and "feature_names" in self.model_metrics[symbol]:
                # Use the same features that were used during training
                feature_columns = self.model_metrics[symbol]["feature_names"]
                print(f"Using feature names from trained model: {feature_columns}")

                # Check if we have a feature mapping
                if "feature_mapping" in self.model_metrics[symbol]:
                    feature_mapping = self.model_metrics[symbol]["feature_mapping"]

                    # Apply feature mapping to ensure consistent naming
                    for orig_feature, std_feature in feature_mapping.items():
                        if (orig_feature in df_features.columns and std_feature not in df_features.columns):
                            print(f"Mapping feature: {orig_feature} -> {std_feature}")
                            df_features[std_feature] = df_features[orig_feature]
            else:
                # Use default features
                feature_columns = [
                    "returns", "log_returns", "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
                    "atr_ratio", "volume_ratio", "rsi_14", "macd", "macd_signal", "macd_hist",
                    "bb_width_20", "bb_position_20"
                ]

            # Filter to only include columns that exist in the dataframe
            available_features = [col for col in feature_columns if col in df_features.columns]

            # Check if we have enough features
            if len(available_features) < 5:
                print(f"Warning: Only {len(available_features)} features available. Need at least 5 for reliable prediction.")
                
                # Try simplified feature set
                simplified_features = ["returns", "log_returns", "sma_ratio_10", "sma_ratio_20", 
                                      "atr_ratio", "volume_ratio", "rsi_14"]
                simplified_available = [col for col in simplified_features if col in df_features.columns]
                
                if len(simplified_available) >= 3:
                    print(f"Using simplified feature set with {len(simplified_available)} features")
                    available_features = simplified_available
                else:
                    # Last resort: retrain
                    print("Not enough features even with simplified set. Retraining model...")
                    model, scaler = self.train_ml_model(symbol, df, force_retrain=True)
                    
                    # Update available features after retraining
                    if symbol in self.model_metrics and "feature_names" in self.model_metrics[symbol]:
                        feature_columns = self.model_metrics[symbol]["feature_names"]
                        available_features = [col for col in feature_columns if col in df_features.columns]
                        
                        if len(available_features) < 3:
                            print("Critical error: Still not enough features after retraining")
                            return 0, 0.5  # Neutral signal with 50% confidence

            print(f"Using {len(available_features)} features for prediction")

            # Create a DataFrame with only the required features
            X = pd.DataFrame(index=latest_data.index)
            for feature in available_features:
                X[feature] = latest_data[feature]

            # Scale features
            X_scaled = self.scalers[symbol].transform(X)

            # Predict
            prediction = self.models[symbol].predict(X_scaled)[0]
            probability = self.models[symbol].predict_proba(X_scaled)[0]

            # Print prediction details
            print(f"Prediction: {'BUY' if prediction else 'SELL'}")
            print(f"Confidence: {max(probability):.2f}")

            # Print feature values used for prediction
            print("\nFeature values used:")
            for feature in available_features:
                feature_value = latest_data[feature].values[0]
                print(f"  {feature}: {feature_value:.4f}")

            # Top influential features if available
            if hasattr(self.models[symbol], "feature_importances_"):
                importances = self.models[symbol].feature_importances_
                feature_importance = list(zip(available_features, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print("\nTop influential features:")
                for feature, importance in feature_importance[:5]:
                    feature_value = latest_data[feature].values[0]
                    print(
                        f"  {feature}: {feature_value:.4f} (importance: {importance:.4f})"
                    )

            # Convert to signal
            if prediction:
                return 1, probability[1]  # Buy signal with probability
            else:
                return -1, probability[0]  # Sell signal with probability

        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            print("Attempting to retrain model with current data...")
            traceback.print_exc()
            
            try:
                # Retrain model
                model, scaler = self.train_ml_model(symbol, df, force_retrain=True)
                if model is None or scaler is None:
                    return 0, 0.5  # Neutral signal with 50% confidence
                    
                # Try prediction with new model
                # ...implementation similar to above...
                return 0, 0.5  # Placeholder - in real code we'd make a prediction
            except Exception as retry_error:
                print(f"Error in model retraining: {str(retry_error)}")
                traceback.print_exc()
                return 0, 0.5  # Neutral signal with 50% confidence

    def predict(self, df):
        """Make a prediction using the ML model"""
        try:
            # Add features
            df_features = self.add_features(df)

            # Get latest data point
            latest_data = df_features.iloc[-1:]

            # Get symbol from dataframe if available
            symbol = (
                latest_data["symbol"].iloc[0]
                if "symbol" in latest_data.columns
                else "unknown"
            )

            # Check if model exists
            if symbol not in self.models or symbol not in self.scalers:
                model, scaler = self.load_model(symbol)

                # If still not available, train a new model
                if model is None or scaler is None:
                    model, scaler = self.train_ml_model(symbol, df)

            # Get feature columns used during training
            if (
                symbol in self.model_metrics
                and "feature_names" in self.model_metrics[symbol]
            ):
                feature_columns = self.model_metrics[symbol]["feature_names"]

                # Apply feature mapping if available
                if "feature_mapping" in self.model_metrics[symbol]:
                    feature_mapping = self.model_metrics[symbol]["feature_mapping"]
                    for orig_feature, std_feature in feature_mapping.items():
                        if (
                            orig_feature in df_features.columns
                            and std_feature not in df_features.columns
                        ):
                            df_features[std_feature] = df_features[orig_feature]
            else:
                # Use default features
                feature_columns = [
                    "returns",
                    "log_returns",
                    "sma_ratio_10",
                    "sma_ratio_20",
                    "sma_ratio_50",
                    "atr_ratio",
                    "volume_ratio",
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_width_20",
                    "bb_position_20",
                ]

            # Filter to only include columns that exist in the dataframe
            available_features = [
                col for col in feature_columns if col in df_features.columns
            ]

            # Create a DataFrame with only the required features in the correct order
            X = pd.DataFrame(index=latest_data.index)
            for feature in available_features:
                X[feature] = latest_data[feature]

            # Scale features
            X_scaled = self.scalers[symbol].transform(X)

            # Predict
            prediction = self.models[symbol].predict(X_scaled)[0]
            probability = self.models[symbol].predict_proba(X_scaled)[0]

            # Return probability of upward movement (second class)
            # For binary classification: True = bullish, False = bearish
            if prediction:
                return probability[1]  # Probability of bullish movement
            else:
                return (
                    1 - probability[0]
                )  # Convert bearish probability to bullish scale

        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return 0.5  # Return neutral prediction on error

    def _create_feature_name_mapping(self, model_features, available_columns):
        """
        Create a mapping between model feature names and available columns
        
        Args:
            model_features: Feature names used during model training
            available_columns: Available columns in the current dataframe
            
        Returns:
            dict: Mapping from model features to available columns
        """
        mapping = {}
        available_cols_set = set(available_columns)
        
        # Direct mapping if column names match
        for feature in model_features:
            if feature in available_cols_set:
                mapping[feature] = feature
                continue
                
            # Check for common feature name variations
            if feature == 'rsi' and 'rsi_14' in available_cols_set:
                mapping[feature] = 'rsi_14'
            elif feature == 'bb_width' and 'bb_width_20' in available_cols_set:
                mapping[feature] = 'bb_width_20'
            elif feature == 'bb_position' and 'bb_position_20' in available_cols_set:
                mapping[feature] = 'bb_position_20'
            elif feature == 'macd' and 'macd' in available_cols_set:
                mapping[feature] = 'macd'
            elif feature == 'macd_signal' and 'macd_signal' in available_cols_set:
                mapping[feature] = 'macd_signal'
            elif feature == 'macd_hist' and 'macd_hist' in available_cols_set:
                mapping[feature] = 'macd_hist'
        
        print(f"Feature mapping created: {mapping}")
        return mapping