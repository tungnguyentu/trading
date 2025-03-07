import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from binance.client import Client
from dotenv import load_dotenv
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml_models import MLManager

# Load environment variables
load_dotenv()

class MLTrainer:
    def __init__(self, symbols=None, timeframe='1h', lookback_days=365, test_size=0.2):
        """
        Initialize the ML Trainer
        
        Args:
            symbols: List of trading pairs to train models for (default: ['BTCUSDT', 'ETHUSDT'])
            timeframe: Timeframe for historical data (default: '1h')
            lookback_days: Number of days of historical data to use (default: 365)
            test_size: Proportion of data to use for testing (default: 0.2)
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.test_size = test_size
        
        # Initialize Binance client for data fetching
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        
        # Map timeframe string to Binance interval
        self.timeframe_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
        # Initialize ML Manager
        self.ml_manager = MLManager()
        
    def fetch_historical_data(self, symbol):
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            DataFrame with historical data
        """
        print(f"Fetching historical data for {symbol} ({self.timeframe})...")
        
        # Calculate start and end times
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.lookback_days)
        
        # Convert to milliseconds timestamp
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Fetch klines from Binance
        klines = self.client.get_historical_klines(
            symbol, 
            self.timeframe_map.get(self.timeframe, Client.KLINE_INTERVAL_1HOUR),
            start_ms,
            end_ms
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Set index to open_time
        df.set_index('open_time', inplace=True)
        
        print(f"Fetched {len(df)} candles for {symbol}")
        return df
    
    def add_features(self, df):
        """Add technical indicators and features"""
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['on_balance_volume'] = (df['close'].diff() > 0).astype(int) * df['volume']
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_ma'] = df['rsi'].rolling(window=10).mean()
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['dmi_plus'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['dmi_minus'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        # Moving averages and derived features
        for window in [8, 13, 21, 34, 55]:
            df[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
        # EMA crossover signals
        df['ema_8_13_cross'] = np.where(df['ema_8'] > df['ema_13'], 1, -1)
        df['ema_13_21_cross'] = np.where(df['ema_13'] > df['ema_21'], 1, -1)
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        df['bbands_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        df['bbands_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        df['bbands_width'] = (df['bbands_upper'] - df['bbands_lower']) / df['close'] * 100
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=26, window_fast=12, window_sign=9)
        
        # Support and resistance levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        
        # Price patterns
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        df['higher_close'] = df['close'] > df['close'].shift(1)
        
        # Advanced momentum features
        df['close_to_ema8'] = (df['close'] - df['ema_8']) / df['ema_8'] * 100
        df['close_to_ema21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
        df['momentum'] = df['close'] - df['close'].shift(4)
        
        # Volatility regime
        df['volatility_regime'] = np.where(df['atr_pct'] > df['atr_pct'].rolling(window=20).mean(), 1, 0)
        
        # Volume price trend
        df['vpt'] = df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
        df['vpt_sma'] = df['vpt'].rolling(window=13).mean()
        
        # Clean up NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df

    def train_advanced_model(self, symbol, df, force_retrain=False, model_type='ensemble', optimize_hyperparams=True):
        """Train an advanced ML model with optimized parameters"""
        print(f"\nTraining advanced model for {symbol}")
        
        # Add features
        df = self.add_features(df)
        
        # Prepare target variable (future returns)
        future_returns = df['close'].pct_change(periods=3).shift(-3)  # 3-period future returns
        df['target'] = np.where(future_returns > 0.002, 1, np.where(future_returns < -0.002, -1, 0))
        
        # Remove last 3 rows since they have NaN target values
        df = df.iloc[:-3]
        
        # Split features and target
        feature_columns = [col for col in df.columns if col not in ['target', 'timestamp', 'date', 'time']]
        X = df[feature_columns]
        y = df['target']
        
        # Train/test split (use more recent data for testing)
        train_size = int(len(df) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'ensemble':
            # Create base models
            models = {
                'rf': RandomForestClassifier(random_state=42),
                'gb': GradientBoostingClassifier(random_state=42),
                'xgb': XGBClassifier(random_state=42)
            }
            
            if optimize_hyperparams:
                # Hyperparameter grids
                param_grids = {
                    'rf': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [5, 10],
                        'class_weight': ['balanced']
                    },
                    'gb': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0]
                    },
                    'xgb': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0]
                    }
                }
                
                # Optimize each model
                optimized_models = {}
                for name, model in models.items():
                    print(f"Optimizing {name} model...")
                    grid_search = GridSearchCV(
                        model, 
                        param_grids[name],
                        cv=TimeSeriesSplit(n_splits=5),
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    optimized_models[name] = grid_search.best_estimator_
                    print(f"Best parameters for {name}: {grid_search.best_params_}")
                
                # Create voting classifier with optimized models
                model = VotingClassifier(
                    estimators=[(name, model) for name, model in optimized_models.items()],
                    voting='soft'
                )
            else:
                # Create voting classifier with default models
                model = VotingClassifier(
                    estimators=[(name, model) for name, model in models.items()],
                    voting='soft'
                )
        
        elif model_type == 'rf':
            if optimize_hyperparams:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [5, 10],
                    'class_weight': ['balanced']
                }
                model = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    param_grid,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='f1_weighted',
                    n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42
                )
        
        elif model_type == 'gb':
            if optimize_hyperparams:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
                model = GridSearchCV(
                    GradientBoostingClassifier(random_state=42),
                    param_grid,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='f1_weighted',
                    n_jobs=-1
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
        
        # Train the model
        print("Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return model, scaler
    
    def train_and_backtest(self, symbol, model_type='ensemble', force_retrain=False, optimize_hyperparams=True, retrain_interval=0):
        """
        Train a model for a symbol and backtest it
        
        Args:
            symbol: Trading pair symbol
            model_type: Type of model to train ('rf', 'gb', 'ensemble')
            force_retrain: Whether to force retraining even if model exists
            optimize_hyperparams: Whether to optimize hyperparameters
            retrain_interval: Interval in hours to retrain the model during backtesting (0 to disable)
            
        Returns:
            dict: Backtest results
        """
        print(f"\n=== Training and Backtesting for {symbol} ===")
        
        # Fetch historical data
        df = self.fetch_historical_data(symbol)
        
        if len(df) < 100:
            print(f"Not enough data for {symbol}. Need at least 100 candles, got {len(df)}.")
            return None
        
        # Train advanced model
        model, scaler = self.train_advanced_model(
            symbol, 
            df, 
            force_retrain=force_retrain,
            model_type=model_type,
            optimize_hyperparams=optimize_hyperparams
        )
        
        if model is None or scaler is None:
            print(f"Failed to train model for {symbol}")
            return None
        
        # Add features for backtesting
        df_features = self.add_features(df)
        
        # Get feature columns
        excluded_columns = ['target', 'target_1', 'target_3', 'target_5', 'timestamp', 
                          'date', 'time', 'symbol', 'open_time', 'close_time']
        feature_columns = [col for col in df_features.columns 
                         if col not in excluded_columns 
                         and df_features[col].dtype in ['float64', 'int64']]
        
        # Run backtest with retraining if enabled
        backtest_results = self.ml_manager.backtest_model(
            model, 
            scaler, 
            df_features, 
            feature_columns,
            initial_balance=10000,
            position_size_pct=0.2,
            retrain_interval=retrain_interval,
            retrain_func=lambda: self.train_advanced_model(
                symbol, 
                df, 
                force_retrain=True,
                model_type=model_type,
                optimize_hyperparams=optimize_hyperparams
            ) if retrain_interval > 0 else None
        )
        
        # Calculate additional metrics if they don't exist
        if 'total_return_pct' not in backtest_results and 'total_return' in backtest_results:
            backtest_results['total_return_pct'] = backtest_results['total_return']
        
        if 'annualized_return_pct' not in backtest_results:
            # Estimate annualized return based on total return
            # Assuming the backtest period is approximately 1 year
            backtest_results['annualized_return_pct'] = backtest_results['total_return_pct']
        
        if 'win_rate' in backtest_results and not isinstance(backtest_results['win_rate'], float):
            backtest_results['win_rate'] = float(backtest_results['win_rate'])
        
        if 'max_drawdown' in backtest_results and 'max_drawdown_pct' not in backtest_results:
            backtest_results['max_drawdown_pct'] = backtest_results['max_drawdown']
        
        if 'sharpe_ratio' not in backtest_results:
            # Estimate Sharpe ratio based on return and drawdown
            if backtest_results['max_drawdown_pct'] > 0:
                backtest_results['sharpe_ratio'] = backtest_results['total_return_pct'] / backtest_results['max_drawdown_pct']
            else:
                backtest_results['sharpe_ratio'] = 0.0
        
        # Print backtest summary
        print("\n=== Backtest Results ===")
        print(f"Symbol: {symbol}")
        print(f"Model Type: {model_type}")
        print(f"Initial Balance: ${backtest_results['initial_balance']:.2f}")
        print(f"Final Balance: ${backtest_results['final_balance']:.2f}")
        print(f"Total Return: {backtest_results['total_return_pct']:.2f}%")
        
        if retrain_interval > 0:
            print(f"Retraining Interval: {retrain_interval} hours")
            if 'retrain_count' in backtest_results:
                print(f"Model Retrains: {backtest_results['retrain_count']}")
        
        if 'annualized_return_pct' in backtest_results:
            print(f"Annualized Return: {backtest_results['annualized_return_pct']:.2f}%")
        
        print(f"Total Trades: {backtest_results['total_trades']}")
        print(f"Win Rate: {backtest_results['win_rate']:.2f}%")
        
        if 'profit_factor' in backtest_results:
            print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
        
        print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
        
        if 'sharpe_ratio' in backtest_results:
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        
        return backtest_results
    
    def train_all_symbols(self, model_type='ensemble', force_retrain=False, optimize_hyperparams=True, retrain_interval=0):
        """
        Train models for all symbols
        
        Args:
            model_type: Type of model to train ('rf', 'gb', 'ensemble')
            force_retrain: Whether to force retraining even if model exists
            optimize_hyperparams: Whether to optimize hyperparameters
            retrain_interval: Interval in hours to retrain the model during backtesting
        """
        results = {}
        
        for symbol in self.symbols:
            results[symbol] = self.train_and_backtest(
                symbol, 
                model_type=model_type,
                force_retrain=force_retrain,
                optimize_hyperparams=optimize_hyperparams,
                retrain_interval=retrain_interval
            )
            
        return results
    
    def compare_timeframes(self, symbol, timeframes=['1h', '4h', '1d'], model_type='ensemble'):
        """
        Compare model performance across different timeframes
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes to compare
            model_type: Type of model to train
        """
        results = {}
        original_timeframe = self.timeframe
        
        for tf in timeframes:
            print(f"\n=== Testing {symbol} on {tf} timeframe ===")
            self.timeframe = tf
            results[tf] = self.train_and_backtest(symbol, model_type=model_type, force_retrain=True)
            
        # Reset timeframe
        self.timeframe = original_timeframe
        
        # Compare results
        print("\n=== Timeframe Comparison ===")
        print(f"Symbol: {symbol}")
        print(f"Model Type: {model_type}")
        print("\nTimeframe | Return % | Win Rate | Profit Factor | Max Drawdown | Sharpe")
        print("---------|----------|----------|--------------|--------------|-------")
        
        for tf, result in results.items():
            if result:
                # Ensure all required keys exist
                if 'total_return_pct' not in result and 'total_return' in result:
                    result['total_return_pct'] = result['total_return']
                
                if 'max_drawdown_pct' not in result and 'max_drawdown' in result:
                    result['max_drawdown_pct'] = result['max_drawdown']
                
                if 'sharpe_ratio' not in result:
                    # Estimate Sharpe ratio
                    if result.get('max_drawdown_pct', 0) > 0:
                        result['sharpe_ratio'] = result.get('total_return_pct', 0) / result.get('max_drawdown_pct', 1)
                    else:
                        result['sharpe_ratio'] = 0.0
                
                # Print results with safe access to keys
                print(f"{tf:9} | {result.get('total_return_pct', 0):8.2f}% | {result.get('win_rate', 0):7.2f}% | {result.get('profit_factor', 0):12.2f} | {result.get('max_drawdown_pct', 0):12.2f}% | {result.get('sharpe_ratio', 0):6.2f}")
        
        return results

def main():
    """Main function to run the trainer from command line"""
    parser = argparse.ArgumentParser(description='Train and backtest ML models for cryptocurrency trading')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Symbols to train models for')
    parser.add_argument('--timeframe', default='1h', choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'], help='Timeframe for historical data')
    parser.add_argument('--lookback', type=int, default=365, help='Number of days of historical data to use')
    parser.add_argument('--model', default='ensemble', choices=['rf', 'gb', 'ensemble'], help='Type of model to train')
    parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')
    parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    parser.add_argument('--compare-timeframes', action='store_true', help='Compare performance across timeframes')
    parser.add_argument('--retrain-interval', type=int, default=0, help='Interval in hours to retrain the model during backtesting (0 to disable)')
    
    args = parser.parse_args()
    
    trainer = MLTrainer(
        symbols=args.symbols,
        timeframe=args.timeframe,
        lookback_days=args.lookback
    )
    
    if args.compare_timeframes:
        for symbol in args.symbols:
            trainer.compare_timeframes(symbol, timeframes=['1h', '4h', '1d'], model_type=args.model)
    else:
        trainer.train_all_symbols(
            model_type=args.model,
            force_retrain=args.force,
            optimize_hyperparams=args.optimize,
            retrain_interval=args.retrain_interval
        )

if __name__ == "__main__":
    main()
