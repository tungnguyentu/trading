import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ml_models import MLManager
from trading_bot import TradingBot
from simulation import TradingSimulator
import argparse
import telegram
import asyncio

# Add Telegram notification functionality
async def send_telegram_notification(bot_token, chat_id, message):
    """Send notification to Telegram"""
    try:
        bot = telegram.Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message)
        print(f"Telegram notification sent: {message}")
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")

def send_telegram_message(bot_token, chat_id, message):
    """Synchronous wrapper for send_telegram_notification"""
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(send_telegram_notification(bot_token, chat_id, message))
    except Exception as e:
        print(f"Error in send_telegram_message: {e}")

def run_simulation(symbol='BTCUSDT', days=30, investment=50, daily_target=15, use_ml=True):
    """
    Run a trading simulation
    
    Args:
        symbol: Trading pair to simulate
        days: Number of days of historical data to use
        investment: Initial investment amount
        daily_target: Daily profit target
        use_ml: Whether to use ML models for signals
    """
    print(f"Starting simulation for {symbol} with ${investment} investment")
    print(f"Daily profit target: ${daily_target}")
    
    # Initialize trading bot to get historical data
    bot = TradingBot()
    
    # Get historical data
    # For simulation we need more data than usual
    original_lookback = bot.lookback_period
    bot.lookback_period = max(2000, int(days * 24 * 4))  # 4 candles per hour * 24 hours * days
    
    print(f"Fetching {bot.lookback_period} 15-minute candles for simulation...")
    df = bot.get_historical_data(symbol)
    
    # Restore original lookback
    bot.lookback_period = original_lookback
    
    if df is None or len(df) < 100:
        print("Error: Not enough historical data for simulation")
        return
    
    print(f"Retrieved {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize ML Manager if needed
    ml_manager = None
    if use_ml:
        ml_manager = MLManager()
        
        # Check if model exists, train if not
        if symbol not in ml_manager.models:
            print(f"Training ML model for {symbol}...")
            ml_manager.train_ml_model(symbol, df)
    
    # Initialize simulator
    simulator = TradingSimulator(
        historical_data=df,
        initial_investment=investment,
        daily_profit_target=daily_target
    )
    
    # Run simulation
    final_balance, trades, daily_profits = simulator.run_simulation(ml_manager)
    
    # Print summary
    print("\n=== FINAL RESULTS ===")
    print(f"Initial Investment: ${investment:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    profit = final_balance - investment
    print(f"Total Profit/Loss: ${profit:.2f} ({(profit/investment)*100:.2f}%)")
    
    # Calculate average daily profit
    if daily_profits:
        avg_daily = sum(daily_profits.values()) / len(daily_profits)
        print(f"Average Daily Profit: ${avg_daily:.2f}")
        
        # Check if we met our target
        days_met = sum(1 for p in daily_profits.values() if p >= daily_target)
        print(f"Days Meeting ${daily_target} Target: {days_met}/{len(daily_profits)} ({days_met/len(daily_profits)*100:.2f}%)")
    
    print("\nSimulation results saved to:", simulator.results_dir)

def optimize_parameters(symbol='BTCUSDT', days=30, investment=50, daily_target=15):
    """
    Optimize trading parameters to meet daily profit target
    """
    print(f"Optimizing parameters for {symbol} to meet ${daily_target}/day target")
    
    # Initialize trading bot to get historical data
    bot = TradingBot()
    
    # Get historical data
    original_lookback = bot.lookback_period
    bot.lookback_period = max(2000, int(days * 24 * 4))
    df = bot.get_historical_data(symbol)
    bot.lookback_period = original_lookback
    
    if df is None or len(df) < 100:
        print("Error: Not enough historical data for optimization")
        return
    
    # Parameter ranges to test
    stop_loss_pcts = [0.01, 0.02, 0.03, 0.05]
    take_profit_pcts = [0.03, 0.04, 0.05, 0.07, 0.1]
    
    best_params = None
    best_profit = 0
    best_days_met = 0
    
    # Test combinations
    for sl in stop_loss_pcts:
        for tp in take_profit_pcts:
            print(f"Testing SL: {sl:.2f}, TP: {tp:.2f}")
            
            # Initialize simulator with these parameters
            simulator = TradingSimulator(
                historical_data=df,
                initial_investment=investment,
                daily_profit_target=daily_target
            )
            
            # Set parameters
            simulator.stop_loss_pct = sl
            simulator.take_profit_pct = tp
            
            # Run simulation
            final_balance, trades, daily_profits = simulator.run_simulation()
            
            # Calculate metrics
            profit = final_balance - investment
            days_met = sum(1 for p in daily_profits.values() if p >= daily_target)
            
            # Check if better than current best
            if days_met > best_days_met or (days_met == best_days_met and profit > best_profit):
                best_params = (sl, tp)
                best_profit = profit
                best_days_met = days_met
                
                print(f"New best parameters: SL={sl:.2f}, TP={tp:.2f}")
                print(f"Profit: ${best_profit:.2f}, Days met target: {best_days_met}/{len(daily_profits)}")
    
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Best Parameters: Stop Loss = {best_params[0]:.2f}, Take Profit = {best_params[1]:.2f}")
    print(f"Total Profit: ${best_profit:.2f}")
    print(f"Days Meeting ${daily_target} Target: {best_days_met}")
    
    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crypto Trading Simulator')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair to simulate')
    parser.add_argument('--days', type=int, default=30, help='Number of days to simulate')
    parser.add_argument('--investment', type=float, default=50.0, help='Initial investment amount')
    parser.add_argument('--target', type=float, default=15.0, help='Daily profit target')
    parser.add_argument('--optimize', action='store_true', help='Optimize parameters')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML signals')
    
    args = parser.parse_args()
    
    if args.optimize:
        best_params = optimize_parameters(
            symbol=args.symbol,
            days=args.days,
            investment=args.investment,
            daily_target=args.target
        )
    else:
        run_simulation(
            symbol=args.symbol,
            days=args.days,
            investment=args.investment,
            daily_target=args.target,
            use_ml=not args.no_ml
        )