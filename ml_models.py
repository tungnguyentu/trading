import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import logging

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
        
        # Load any existing models
        self.load_all_models()
    
    def model_path(self, symbol):
        """Get path for model file"""
        return os.path.join(self.models_dir, f"{symbol}_model.pkl")
    
    def scaler_path(self, symbol):
        """Get path for scaler file"""
        return os.path.join(self.models_dir, f"{symbol}_scaler.pkl")
    
    def save_model(self, symbol, model, scaler):
        """Save model and scaler to disk"""
        try:
            with open(self.model_path(symbol), 'wb') as f:
                pickle.dump(model, f)
            
            with open(self.scaler_path(symbol), 'wb') as f:
                pickle.dump(scaler, f)
                
            print(f"Model and scaler for {symbol} saved successfully")
            return True
        except Exception as e:
            print(f"Error saving model for {symbol}: {e}")
            return False
    
    def load_model(self, symbol):
        """Load model and scaler from disk"""
        try:
            model_file = self.model_path(symbol)
            scaler_file = self.scaler_path(symbol)
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                
                self.models[symbol] = model
                self.scalers[symbol] = scaler
                
                print(f"Model and scaler for {symbol} loaded successfully")
                return model, scaler
            else:
                print(f"No saved model found for {symbol}")
                return None, None
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            return None, None
    
    def load_all_models(self):
        """Load all models from the models directory"""
        try:
            # Get all model files
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                # Extract symbol from filename
                symbol = model_file.replace('_model.pkl', '')
                self.load_model(symbol)
                
            print(f"Loaded {len(self.models)} models from disk")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def add_features(self, df):
        """Add technical indicators as features for ML model"""
        # Copy the dataframe to avoid modifying the original
        df_feat = df.copy()
        
        # Basic price features
        df_feat['returns'] = df_feat['close'].pct_change()
        df_feat['log_returns'] = np.log(df_feat['close']/df_feat['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df_feat[f'sma_{window}'] = df_feat['close'].rolling(window=window).mean()
            df_feat[f'sma_ratio_{window}'] = df_feat['close'] / df_feat[f'sma_{window}']
        
        # Volatility indicators
        df_feat['atr'] = df_feat['ATR']
        df_feat['atr_ratio'] = df_feat['atr'] / df_feat['close']
        
        # Volume indicators
        df_feat['volume_ma_5'] = df_feat['volume'].astype(float).rolling(window=5).mean()
        df_feat['volume_ratio'] = df_feat['volume'].astype(float) / df_feat['volume_ma_5']
        
        # RSI
        delta = df_feat['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_feat['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df_feat['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_feat['close'].ewm(span=26, adjust=False).mean()
        df_feat['macd'] = ema_12 - ema_26
        df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
        df_feat['macd_hist'] = df_feat['macd'] - df_feat['macd_signal']
        
        # Bollinger Bands
        df_feat['bb_middle'] = df_feat['close'].rolling(window=20).mean()
        df_feat['bb_std'] = df_feat['close'].rolling(window=20).std()
        df_feat['bb_upper'] = df_feat['bb_middle'] + 2 * df_feat['bb_std']
        df_feat['bb_lower'] = df_feat['bb_middle'] - 2 * df_feat['bb_std']
        df_feat['bb_width'] = (df_feat['bb_upper'] - df_feat['bb_lower']) / df_feat['bb_middle']
        df_feat['bb_position'] = (df_feat['close'] - df_feat['bb_lower']) / (df_feat['bb_upper'] - df_feat['bb_lower'])
        
        # Target variable (future returns)
        df_feat['target'] = df_feat['returns'].shift(-1) > 0
        
        # Drop NaN values
        df_feat = df_feat.dropna()
        
        return df_feat
    
    def train_ml_model(self, symbol, df):
        """Train a machine learning model for the given symbol"""
        print(f"Training ML model for {symbol}")
        
        # Add features
        df_features = self.add_features(df)
        
        # Define features and target
        feature_columns = [
            'returns', 'log_returns', 
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'atr_ratio', 'volume_ratio', 'rsi', 
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position'
        ]
        
        X = df_features[feature_columns]
        y = df_features['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy for {symbol}: {accuracy:.4f}")
        
        # Store model and scaler in memory
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Save to disk
        self.save_model(symbol, model, scaler)
        
        return model, scaler
    
    def get_ml_signal(self, symbol, df):
        """Get trading signal from ML model"""
        # Check if model exists in memory, try to load from disk if not
        if symbol not in self.models or symbol not in self.scalers:
            model, scaler = self.load_model(symbol)
            
            # If still not available, train a new model
            if model is None or scaler is None:
                model, scaler = self.train_ml_model(symbol, df)
        
        # Add features
        df_features = self.add_features(df)
        
        # Get latest data point
        latest_data = df_features.iloc[-1:]
        
        # Define features
        feature_columns = [
            'returns', 'log_returns', 
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'atr_ratio', 'volume_ratio', 'rsi', 
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position'
        ]
        
        # Scale features
        X = latest_data[feature_columns]
        X_scaled = self.scalers[symbol].transform(X)
        
        # Predict
        prediction = self.models[symbol].predict(X_scaled)[0]
        probability = self.models[symbol].predict_proba(X_scaled)[0]
        
        # Convert to signal
        if prediction:
            return 1, probability[1]  # Buy signal with probability
        else:
            return -1, probability[0]  # Sell signal with probability