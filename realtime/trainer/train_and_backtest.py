#!/usr/bin/env python3
import os
import sys
import argparse
from ml_trainer import MLTrainer

def main():
    """
    Main function to run training and backtesting
    """
    parser = argparse.ArgumentParser(description='Train and backtest ML models for cryptocurrency trading')
    
    # Basic configuration
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], 
                        help='Symbols to train models for')
    parser.add_argument('--timeframe', default='1h', 
                        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'], 
                        help='Timeframe for historical data')
    parser.add_argument('--lookback', type=int, default=365, 
                        help='Number of days of historical data to use')
    
    # Model options
    parser.add_argument('--model', default='ensemble', 
                        choices=['rf', 'gb', 'ensemble'], 
                        help='Type of model to train (rf=Random Forest, gb=Gradient Boosting, ensemble=Ensemble of models)')
    parser.add_argument('--force', action='store_true', 
                        help='Force retraining even if model exists')
    parser.add_argument('--optimize', action='store_true', 
                        help='Optimize hyperparameters (takes longer but may improve performance)')
    parser.add_argument('--retrain-interval', type=int, default=0,
                        help='Retraining interval in hours during backtesting (0 to disable)')
    
    # Analysis options
    parser.add_argument('--compare-timeframes', action='store_true', 
                        help='Compare performance across different timeframes')
    parser.add_argument('--initial-balance', type=float, default=10000, 
                        help='Initial balance for backtesting')
    parser.add_argument('--position-size', type=float, default=0.2, 
                        help='Position size as percentage of balance (0.2 = 20%)')
    
    args = parser.parse_args()
    
    print("=== ML Model Training and Backtesting ===")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Lookback period: {args.lookback} days")
    print(f"Model type: {args.model}")
    print(f"Optimize hyperparameters: {args.optimize}")
    print(f"Force retrain: {args.force}")
    print(f"Retrain interval: {args.retrain_interval} hours {'(disabled)' if args.retrain_interval == 0 else ''}")
    print(f"Initial balance: ${args.initial_balance}")
    print(f"Position size: {args.position_size * 100}%")
    print()
    
    # Initialize trainer
    trainer = MLTrainer(
        symbols=args.symbols,
        timeframe=args.timeframe,
        lookback_days=args.lookback
    )
    
    if args.compare_timeframes:
        print("=== Comparing Performance Across Timeframes ===")
        for symbol in args.symbols:
            trainer.compare_timeframes(
                symbol, 
                timeframes=['1h', '4h', '1d'], 
                model_type=args.model
            )
    else:
        # Train and backtest for all symbols
        results = trainer.train_all_symbols(
            model_type=args.model,
            force_retrain=args.force,
            optimize_hyperparams=args.optimize,
            retrain_interval=args.retrain_interval
        )
        
        # Print summary of all results
        print("\n=== Summary of All Results ===")
        print("Symbol | Return % | Win Rate | Profit Factor | Max Drawdown | Sharpe")
        print("-------|----------|----------|--------------|--------------|-------")
        
        for symbol, result in results.items():
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
                print(f"{symbol:6} | {result.get('total_return_pct', 0):8.2f}% | {result.get('win_rate', 0):7.2f}% | {result.get('profit_factor', 0):12.2f} | {result.get('max_drawdown_pct', 0):12.2f}% | {result.get('sharpe_ratio', 0):6.2f}")

if __name__ == "__main__":
    main()