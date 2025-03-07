import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from binance.client import Client
from dotenv import load_dotenv

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
    
    def train_and_backtest(self, symbol, model_type='ensemble', force_retrain=False, optimize_hyperparams=True):
        """
        Train a model for a symbol and backtest it
        
        Args:
            symbol: Trading pair symbol
            model_type: Type of model to train ('rf', 'gb', 'ensemble')
            force_retrain: Whether to force retraining even if model exists
            optimize_hyperparams: Whether to optimize hyperparameters
            
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
        model, scaler = self.ml_manager.train_advanced_model(
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
        df_features = self.ml_manager.add_features(df)
        
        # Get feature columns
        excluded_columns = ['target', 'target_1', 'target_3', 'target_5', 'timestamp', 
                          'date', 'time', 'symbol', 'open_time', 'close_time']
        feature_columns = [col for col in df_features.columns 
                         if col not in excluded_columns 
                         and df_features[col].dtype in ['float64', 'int64']]
        
        # Run backtest
        backtest_results = self.ml_manager.backtest_model(
            model, 
            scaler, 
            df_features, 
            feature_columns,
            initial_balance=10000,
            position_size_pct=0.2
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
    
    def train_all_symbols(self, model_type='ensemble', force_retrain=False, optimize_hyperparams=True):
        """
        Train models for all symbols
        
        Args:
            model_type: Type of model to train ('rf', 'gb', 'ensemble')
            force_retrain: Whether to force retraining even if model exists
            optimize_hyperparams: Whether to optimize hyperparameters
        """
        results = {}
        
        for symbol in self.symbols:
            results[symbol] = self.train_and_backtest(
                symbol, 
                model_type=model_type,
                force_retrain=force_retrain,
                optimize_hyperparams=optimize_hyperparams
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
            optimize_hyperparams=args.optimize
        )

if __name__ == "__main__":
    main()
