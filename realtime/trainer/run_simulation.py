#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_trainer import MLTrainer

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation_runner import run_simulation
from simulator import RealtimeSimulator

def main():
    """
    Main function to run a simulation with a trained model
    """
    parser = argparse.ArgumentParser(description='Run a simulation with a trained ML model')
    
    # Basic configuration
    parser.add_argument('--symbol', default='BTCUSDT', 
                        help='Symbol to run simulation for')
    parser.add_argument('--timeframe', default='1h', 
                        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'], 
                        help='Timeframe for simulation')
    parser.add_argument('--duration', type=int, default=24, 
                        help='Duration of simulation in hours')
    parser.add_argument('--update-interval', type=int, default=15, 
                        help='Update interval in minutes')
    
    # Trading parameters
    parser.add_argument('--initial-investment', type=float, default=1000, 
                        help='Initial investment amount in USD')
    parser.add_argument('--leverage', type=float, default=1, 
                        help='Leverage to use (1-20)')
    parser.add_argument('--profit-target', type=float, default=15, 
                        help='Daily profit target in USD')
    
    # Model options
    parser.add_argument('--train-first', action='store_true', 
                        help='Train model before running simulation')
    parser.add_argument('--model-type', default='ensemble', 
                        choices=['rf', 'gb', 'ensemble'], 
                        help='Type of model to train')
    parser.add_argument('--optimize', action='store_true', 
                        help='Optimize hyperparameters during training')
    
    args = parser.parse_args()
    
    print("=== ML Model Simulation ===")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Duration: {args.duration} hours")
    print(f"Update interval: {args.update_interval} minutes")
    print(f"Initial investment: ${args.initial_investment}")
    print(f"Leverage: {args.leverage}x")
    print(f"Daily profit target: ${args.profit_target}")
    print()
    
    # Train model if requested
    if args.train_first:
        print("=== Training Model First ===")
        trainer = MLTrainer(
            symbols=[args.symbol],
            timeframe=args.timeframe,
            lookback_days=365
        )
        
        results = trainer.train_and_backtest(
            args.symbol,
            model_type=args.model_type,
            force_retrain=True,
            optimize_hyperparams=args.optimize
        )
        
        if not results:
            print("Failed to train model. Exiting.")
            return
    
    # Initialize simulator
    simulator = RealtimeSimulator(
        symbol=args.symbol,
        initial_investment=args.initial_investment,
        daily_profit_target=args.profit_target,
        leverage=args.leverage
    )
    
    # Run simulation
    print("\n=== Starting Simulation ===")
    run_simulation(
        simulator,
        duration_hours=args.duration,
        update_interval_minutes=args.update_interval
    )
    
    # Save results
    simulator.save_realtime_results()
    
    print("\n=== Simulation Complete ===")
    print(f"Final balance: ${simulator.simulator.balance:.2f}")
    print(f"Total return: {(simulator.simulator.balance / args.initial_investment - 1) * 100:.2f}%")
    print(f"Total trades: {len(simulator.simulator.trades)}")
    
    # Calculate win rate
    if simulator.simulator.trades:
        winning_trades = sum(1 for trade in simulator.simulator.trades if trade['profit_amount'] > 0)
        win_rate = winning_trades / len(simulator.simulator.trades) * 100
        print(f"Win rate: {win_rate:.2f}%")
    
    # Plot equity curve
    equity_curve = [(trade['exit_date'], trade['balance']) for trade in simulator.simulator.trades]
    if equity_curve:
        dates, balances = zip(*equity_curve)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, balances)
        plt.title(f"Equity Curve - {args.symbol} ({args.timeframe})")
        plt.xlabel("Date")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.getcwd(), f"equity_curve_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        print(f"Equity curve saved to: {plot_path}")

if __name__ == "__main__":
    main() 