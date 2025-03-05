import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import joblib
import traceback

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
            with open(self.model_path(symbol), 'wb') as f:
                pickle.dump(model, f)
                
            # Save scaler
            with open(self.scaler_path(symbol), 'wb') as f:
                pickle.dump(scaler, f)
                
            # Save metrics if provided
            if metrics:
                with open(self.metrics_path(symbol), 'wb') as f:
                    pickle.dump(metrics, f)
                
            print(f"Model and scaler saved for {symbol}")
            
            # Store feature importances if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[symbol] = model.feature_importances_
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, symbol):
        """Load model and scaler from disk"""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path(symbol)) or not os.path.exists(self.scaler_path(symbol)):
                print(f"Model or scaler not found for {symbol}")
                return None, None
                
            # Load model
            with open(self.model_path(symbol), 'rb') as f:
                model = pickle.load(f)
                
            # Load scaler
            with open(self.scaler_path(symbol), 'rb') as f:
                scaler = pickle.load(f)
                
            # Load metrics if available
            if os.path.exists(self.metrics_path(symbol)):
                with open(self.metrics_path(symbol), 'rb') as f:
                    self.model_metrics[symbol] = pickle.load(f)
                
            # Store in memory
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            # Store feature importances if available
            if hasattr(model, 'feature_importances_'):
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
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]
            
            # Extract symbols
            symbols = [f.split('_model.pkl')[0] for f in model_files]
            
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
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_feat.columns:
                    print(f"Warning: Required column '{col}' not found in dataframe")
            
            # Ensure all basic SMAs are calculated
            df_feat = self._ensure_basic_smas(df_feat)
            
            # Calculate returns if not already present
            if 'returns' not in df_feat.columns and 'close' in df_feat.columns:
                df_feat['returns'] = df_feat['close'].pct_change()
            
            # Calculate log returns if not already present
            if 'log_returns' not in df_feat.columns and 'close' in df_feat.columns:
                df_feat['log_returns'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
            
            # Calculate ATR if not already present
            if 'atr' not in df_feat.columns and all(col in df_feat.columns for col in ['high', 'low', 'close']):
                df_feat['atr'] = self._calculate_atr(df_feat)
                if 'close' in df_feat.columns:
                    df_feat['atr_ratio'] = df_feat['atr'] / df_feat['close']
            
            # Calculate RSI if not already present
            if 'rsi_14' not in df_feat.columns and 'close' in df_feat.columns:
                df_feat['rsi_14'] = self._calculate_rsi(df_feat, 14)
            
            # Calculate MACD if not already present
            if 'macd' not in df_feat.columns and 'close' in df_feat.columns:
                ema_12 = df_feat['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df_feat['close'].ewm(span=26, adjust=False).mean()
                df_feat['macd'] = ema_12 - ema_26
                df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
                df_feat['macd_hist'] = df_feat['macd'] - df_feat['macd_signal']
            
            # Calculate Bollinger Bands if not already present
            if 'bb_width_20' not in df_feat.columns and 'close' in df_feat.columns:
                if 'sma_20' not in df_feat.columns:
                    df_feat['sma_20'] = df_feat['close'].rolling(window=20).mean()
                
                std_20 = df_feat['close'].rolling(window=20).std()
                df_feat['bb_upper_20'] = df_feat['sma_20'] + (std_20 * 2)
                df_feat['bb_lower_20'] = df_feat['sma_20'] - (std_20 * 2)
                df_feat['bb_width_20'] = (df_feat['bb_upper_20'] - df_feat['bb_lower_20']) / df_feat['sma_20']
                df_feat['bb_position_20'] = (df_feat['close'] - df_feat['bb_lower_20']) / (df_feat['bb_upper_20'] - df_feat['bb_lower_20'])
            
            # Calculate volume indicators if not already present
            if 'volume_ratio' not in df_feat.columns and 'volume' in df_feat.columns:
                df_feat['volume_ma_5'] = df_feat['volume'].astype(float).rolling(window=5).mean()
                df_feat['volume_ratio'] = df_feat['volume'].astype(float) / df_feat['volume_ma_5']
            
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
        if 'SMA_short' in df_feat.columns and 'sma_10' not in df_feat.columns:
            print("Converting SMA_short to sma_10")
            df_feat['sma_10'] = df_feat['SMA_short']
            
            # Calculate sma_ratio_10
            if 'close' in df_feat.columns:
                df_feat['sma_ratio_10'] = df_feat['close'] / df_feat['sma_10']
        
        if 'SMA_long' in df_feat.columns and 'sma_20' not in df_feat.columns:
            print("Converting SMA_long to sma_20")
            df_feat['sma_20'] = df_feat['SMA_long']
            
            # Calculate sma_ratio_20
            if 'close' in df_feat.columns:
                df_feat['sma_ratio_20'] = df_feat['close'] / df_feat['sma_20']
        
        # Calculate basic SMAs if they don't exist
        if 'close' in df_feat.columns:
            sma_periods = [5, 10, 20, 50, 100, 200]
            for period in sma_periods:
                sma_col = f'sma_{period}'
                ratio_col = f'sma_ratio_{period}'
                
                if sma_col not in df_feat.columns:
                    print(f"Calculating {sma_col}")
                    df_feat[sma_col] = df_feat['close'].rolling(window=period).mean()
                
                if ratio_col not in df_feat.columns and sma_col in df_feat.columns:
                    print(f"Calculating {ratio_col}")
                    df_feat[ratio_col] = df_feat['close'] / df_feat[sma_col]
        
        # Copy ATR to atr if it exists
        if 'ATR' in df_feat.columns and 'atr' not in df_feat.columns:
            print("Converting ATR to atr")
            df_feat['atr'] = df_feat['ATR']
            
            # Calculate atr_ratio
            if 'close' in df_feat.columns:
                df_feat['atr_ratio'] = df_feat['atr'] / df_feat['close']
        
        # Calculate returns and log_returns if they don't exist
        if 'close' in df_feat.columns:
            if 'returns' not in df_feat.columns:
                print("Calculating returns")
                df_feat['returns'] = df_feat['close'].pct_change()
            
            if 'log_returns' not in df_feat.columns:
                print("Calculating log_returns")
                df_feat['log_returns'] = np.log(df_feat['close']/df_feat['close'].shift(1))
        
        # Calculate volume_ratio if it doesn't exist
        if 'volume' in df_feat.columns and 'volume_ratio' not in df_feat.columns:
            print("Calculating volume_ratio")
            df_feat['volume_ma_5'] = df_feat['volume'].astype(float).rolling(window=5).mean()
            df_feat['volume_ratio'] = df_feat['volume'].astype(float) / df_feat['volume_ma_5']
        
        # Calculate RSI if it doesn't exist
        if 'close' in df_feat.columns and 'rsi_14' not in df_feat.columns:
            print("Calculating RSI")
            delta = df_feat['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_feat['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD if it doesn't exist
        if 'close' in df_feat.columns and 'macd' not in df_feat.columns:
            print("Calculating MACD")
            ema12 = df_feat['close'].ewm(span=12, adjust=False).mean()
            ema26 = df_feat['close'].ewm(span=26, adjust=False).mean()
            df_feat['macd'] = ema12 - ema26
            df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
            df_feat['macd_hist'] = df_feat['macd'] - df_feat['macd_signal']
        
        # Calculate Bollinger Bands if they don't exist
        if 'close' in df_feat.columns and 'bb_upper_20' not in df_feat.columns:
            print("Calculating Bollinger Bands")
            if 'sma_20' not in df_feat.columns:
                df_feat['sma_20'] = df_feat['close'].rolling(window=20).mean()
            
            std_20 = df_feat['close'].rolling(window=20).std()
            df_feat['bb_upper_20'] = df_feat['sma_20'] + (std_20 * 2)
            df_feat['bb_lower_20'] = df_feat['sma_20'] - (std_20 * 2)
            df_feat['bb_width_20'] = (df_feat['bb_upper_20'] - df_feat['bb_lower_20']) / df_feat['sma_20']
            df_feat['bb_position_20'] = (df_feat['close'] - df_feat['bb_lower_20']) / (df_feat['bb_upper_20'] - df_feat['bb_lower_20'])
        
        return df_feat
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    def _calculate_rsi(self, df, window=14):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
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
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        
        # Handle division by zero
        denom = high_max - low_min
        denom = denom.replace(0, np.nan)
        
        stoch_k = 100 * ((df['close'] - low_min) / denom)
        stoch_k = stoch_k.replace(np.nan, 50)  # Default to middle value when undefined
        
        return stoch_k
    
    def _add_ichimoku(self, df):
        """Add Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = df['high'].rolling(window=9).max()
        nine_period_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        df['chikou_span'] = df['close'].shift(-26)
        
        # Cloud strength and direction
        df['cloud_strength'] = abs(df['senkou_span_a'] - df['senkou_span_b']) / ((df['senkou_span_a'] + df['senkou_span_b']) / 2)
        df['above_cloud'] = ((df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])).astype(int)
        df['below_cloud'] = ((df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])).astype(int)
        df['in_cloud'] = (~df['above_cloud'].astype(bool) & ~df['below_cloud'].astype(bool)).astype(int)
        
        return df
    
    def _add_adx(self, df, window=14):
        """Add Average Directional Index"""
        # True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_adx'] = df['tr'].rolling(window=window).mean()
        
        # Plus Directional Movement (+DM)
        df['plus_dm'] = 0.0
        df.loc[(df['high'] - df['high'].shift() > df['low'].shift() - df['low']) & 
               (df['high'] - df['high'].shift() > 0), 'plus_dm'] = df['high'] - df['high'].shift()
        
        # Minus Directional Movement (-DM)
        df['minus_dm'] = 0.0
        df.loc[(df['low'].shift() - df['low'] > df['high'] - df['high'].shift()) & 
               (df['low'].shift() - df['low'] > 0), 'minus_dm'] = df['low'].shift() - df['low']
        
        # Smooth +DM and -DM
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=window).mean() / df['atr_adx'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=window).mean() / df['atr_adx'])
        
        # Directional Movement Index (DX)
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        
        # Average Directional Index (ADX)
        df['adx'] = df['dx'].rolling(window=window).mean()
        
        return df
    
    def train_ml_model(self, symbol, df, force_retrain=False):
        """Train ML model for a specific symbol"""
        try:
            print(f"\n--- Training ML model for {symbol} ---")
            
            # Check if model already exists and we're not forcing a retrain
            if not force_retrain and symbol in self.models and symbol in self.scalers:
                print(f"Model for {symbol} already exists. Use force_retrain=True to retrain.")
                return self.models[symbol], self.scalers[symbol]
            
            # Add features
            df_features = self.add_features(df)
            
            # Print available columns after feature addition
            print(f"Available columns after feature addition: {df_features.columns.tolist()}")
            
            # Check if we have enough data
            if len(df_features) < 100:
                print(f"Not enough data to train model for {symbol}. Need at least 100 rows, got {len(df_features)}.")
                return None, None
            
            # Define target - predict if next candle will be up (1) or down (0)
            if 'target' not in df_features.columns:
                print("Creating target variable (next candle direction)")
                df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
            
            # Drop rows with NaN values
            df_features = df_features.dropna()
            
            # Define features - use what's available
            available_columns = df_features.columns.tolist()
            
            # Basic feature set - prioritize these if available
            basic_features = [
                'returns', 'log_returns', 
                'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                'atr_ratio', 'volume_ratio', 'rsi_14', 
                'macd', 'macd_signal', 'macd_hist',
                'bb_width_20', 'bb_position_20'
            ]
            
            # Filter to only include columns that exist in the dataframe
            feature_columns = [col for col in basic_features if col in available_columns]
            
            # If we don't have enough features, use whatever is available
            if len(feature_columns) < 5:
                print(f"Warning: Only {len(feature_columns)} basic features available. Adding additional columns.")
                
                # Add any numeric columns that aren't the target or timestamp
                exclude_cols = ['target', 'timestamp', 'date', 'time', 'symbol', 
                               'target_2day', 'target_3day', 'target_5day']
                
                additional_features = [col for col in available_columns 
                                      if col not in exclude_cols 
                                      and col not in feature_columns
                                      and df_features[col].dtype in ['float64', 'int64']]
                
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
                std_feature = feature.lower().replace(' ', '_')
                
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
            y = df_features['target']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
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
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'feature_names': standardized_features,  # Store standardized feature names
                'feature_mapping': feature_mapping,      # Store mapping for future reference
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(df_features)
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
    
    def _plot_feature_importance(self, symbol, model, feature_names):
        """Plot feature importance"""
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance for {symbol}')
        plt.bar(range(len(indices)), importances[indices], align='center')
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
            
            # Define features - must match those used in training
            if symbol in self.model_metrics and 'feature_names' in self.model_metrics[symbol]:
                # Use the same features that were used during training
                feature_columns = self.model_metrics[symbol]['feature_names']
                print(f"Using feature names from trained model: {feature_columns}")
                
                # Check if we have a feature mapping
                if 'feature_mapping' in self.model_metrics[symbol]:
                    feature_mapping = self.model_metrics[symbol]['feature_mapping']
                    
                    # Apply feature mapping to ensure consistent naming
                    for orig_feature, std_feature in feature_mapping.items():
                        if orig_feature in df_features.columns and std_feature not in df_features.columns:
                            print(f"Mapping feature: {orig_feature} -> {std_feature}")
                            df_features[std_feature] = df_features[orig_feature]
            else:
                # Fallback to default features
                feature_columns = [
                    'returns', 'log_returns', 
                    'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                    'atr_ratio', 'volume_ratio', 'rsi_14', 
                    'macd', 'macd_signal', 'macd_hist',
                    'bb_width_20', 'bb_position_20'
                ]
            
            # Filter to only include columns that exist in the dataframe
            available_features = [col for col in feature_columns if col in df_features.columns]
            
            # Check if we have enough features
            if len(available_features) < 5:
                print(f"Warning: Only {len(available_features)} features available. Need at least 5 for reliable prediction.")
                print(f"Available features: {available_features}")
                print(f"Missing features: {[col for col in feature_columns if col not in df_features.columns]}")
                
                # Try to use a simplified feature set
                simplified_features = [
                    'returns', 'log_returns', 'sma_ratio_10', 'sma_ratio_20', 
                    'atr_ratio', 'volume_ratio', 'rsi_14'
                ]
                
                # Filter to only include columns that exist
                simplified_available = [col for col in simplified_features if col in df_features.columns]
                
                if len(simplified_available) >= 3:
                    print(f"Using simplified feature set with {len(simplified_available)} features")
                    available_features = simplified_available
                else:
                    print("Not enough features even with simplified set. Retraining model...")
                    # Force retrain with current data structure
                    model, scaler = self.train_ml_model(symbol, df, force_retrain=True)
                    
                    # Get updated feature list after retraining
                    if symbol in self.model_metrics and 'feature_names' in self.model_metrics[symbol]:
                        feature_columns = self.model_metrics[symbol]['feature_names']
                        available_features = [col for col in feature_columns if col in df_features.columns]
                        
                        # If we still don't have enough features, something is wrong
                        if len(available_features) < 3:
                            print("Critical error: Still not enough features after retraining")
                            return 0, 0.5  # Neutral signal with 50% confidence
            
            print(f"Using {len(available_features)} features for prediction")
            
            # Create a DataFrame with only the required features in the correct order
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
            
            # Print top influential features if available
            if hasattr(self.models[symbol], 'feature_importances_'):
                importances = self.models[symbol].feature_importances_
                feature_importance = list(zip(available_features, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print("\nTop influential features:")
                for feature, importance in feature_importance[:5]:
                    feature_value = latest_data[feature].values[0]
                    print(f"  {feature}: {feature_value:.4f} (importance: {importance:.4f})")
            
            # Convert to signal
            if prediction:
                return 1, probability[1]  # Buy signal with probability
            else:
                return -1, probability[0]  # Sell signal with probability
                
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            print("Attempting to retrain model with current data...")
            
            try:
                # Retrain model with current data and force it to use current features
                model, scaler = self.train_ml_model(symbol, df, force_retrain=True)
                
                if model is None or scaler is None:
                    print("Failed to retrain model")
                    return 0, 0.5  # Neutral signal with 50% confidence
                
                # Try prediction again with new model
                df_features = self.add_features(df)
                latest_data = df_features.iloc[-1:]
                
                # Get updated feature list
                if symbol in self.model_metrics and 'feature_names' in self.model_metrics[symbol]:
                    feature_columns = self.model_metrics[symbol]['feature_names']
                    
                    # Apply feature mapping if available
                    if 'feature_mapping' in self.model_metrics[symbol]:
                        feature_mapping = self.model_metrics[symbol]['feature_mapping']
                        for orig_feature, std_feature in feature_mapping.items():
                            if orig_feature in df_features.columns and std_feature not in df_features.columns:
                                df_features[std_feature] = df_features[orig_feature]
                    
                    available_features = [col for col in feature_columns if col in df_features.columns]
                    
                    # Double check we have all features
                    if len(available_features) < len(feature_columns):
                        print(f"Warning: Some features are still missing after retraining")
                        print(f"Missing: {[col for col in feature_columns if col not in df_features.columns]}")
                
                # Create a DataFrame with only the required features in the correct order
                X = pd.DataFrame(index=latest_data.index)
                for feature in available_features:
                    X[feature] = latest_data[feature]
                
                # Scale features
                X_scaled = self.scalers[symbol].transform(X)
                
                # Predict
                prediction = self.models[symbol].predict(X_scaled)[0]
                probability = self.models[symbol].predict_proba(X_scaled)[0]
                
                print(f"Retrained model prediction: {'BUY' if prediction else 'SELL'} with {max(probability):.2f} confidence")
                
                # Convert to signal
                if prediction:
                    return 1, probability[1]  # Buy signal with probability
                else:
                    return -1, probability[0]  # Sell signal with probability
                    
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
            symbol = latest_data['symbol'].iloc[0] if 'symbol' in latest_data.columns else 'unknown'
            
            # Check if model exists
            if symbol not in self.models or symbol not in self.scalers:
                model, scaler = self.load_model(symbol)
                
                # If still not available, train a new model
                if model is None or scaler is None:
                    model, scaler = self.train_ml_model(symbol, df)
            
            # Get feature columns used during training
            if symbol in self.model_metrics and 'feature_names' in self.model_metrics[symbol]:
                feature_columns = self.model_metrics[symbol]['feature_names']
                
                # Apply feature mapping if available
                if 'feature_mapping' in self.model_metrics[symbol]:
                    feature_mapping = self.model_metrics[symbol]['feature_mapping']
                    for orig_feature, std_feature in feature_mapping.items():
                        if orig_feature in df_features.columns and std_feature not in df_features.columns:
                            df_features[std_feature] = df_features[orig_feature]
            else:
                # Use default features
                feature_columns = [
                    'returns', 'log_returns', 
                    'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                    'atr_ratio', 'volume_ratio', 'rsi_14', 
                    'macd', 'macd_signal', 'macd_hist',
                    'bb_width_20', 'bb_position_20'
                ]
            
            # Filter to only include columns that exist in the dataframe
            available_features = [col for col in feature_columns if col in df_features.columns]
            
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
                return 1 - probability[0]  # Convert bearish probability to bullish scale
                
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return 0.5  # Return neutral prediction on error