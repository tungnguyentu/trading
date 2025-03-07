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
            optimize_hyperparams=args.optimize
        )
        
        # Print summary of all results
        print("\n=== Summary of All Results ===")
        print("Symbol | Return % | Win Rate | Profit Factor | Max Drawdown | Sharpe")
        print("-------|----------|----------|--------------|--------------|-------")
        
        for symbol, result in results.items():
            if result:
                print(f"{symbol:6} | {result['total_return_pct']:8.2f}% | {result['win_rate']:7.2f}% | {result['profit_factor']:12.2f} | {result['max_drawdown_pct']:12.2f}% | {result['sharpe_ratio']:6.2f}")

if __name__ == "__main__":
    main() 