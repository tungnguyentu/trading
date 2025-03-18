#!/usr/bin/env python3
"""
Main script to run the real-time trading bot.
"""

import argparse
import os
from dotenv import load_dotenv
from realtime.real_trader import RealtimeTrader

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run real-time trading bot")
    
    # Trading parameters
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--investment", type=float, default=50.0, help="Initial investment amount in USDT")
    parser.add_argument("--profit-target", type=float, default=15.0, help="Daily profit target percentage")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage for futures trading")
    
    # Duration parameters
    parser.add_argument("--duration", type=int, default=24, help="Trading duration in hours")
    parser.add_argument("--interval-minutes", type=int, default=15, help="Update interval in minutes")
    parser.add_argument("--interval-seconds", type=int, default=0, help="Additional update interval in seconds")
    
    # Trading modes
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode without real trades")
    parser.add_argument("--full-investment", action="store_true", help="Use full initial investment for each trade")
    parser.add_argument("--full-margin", action="store_true", help="Use full available margin for each trade")
    parser.add_argument("--compound", action="store_true", help="Enable compound interest")
    
    # Pyramiding settings
    parser.add_argument("--pyramiding", action="store_true", help="Enable pyramiding strategy")
    parser.add_argument("--max-pyramid", type=int, default=2, help="Maximum pyramid entries")
    parser.add_argument("--pyramid-threshold", type=float, default=0.01, help="Pyramid entry threshold percentage")
    
    # Take profit and stop loss settings
    parser.add_argument("--dynamic-tp", action="store_true", help="Use dynamic take profit based on market conditions")
    parser.add_argument("--fixed-tp", type=float, default=0, help="Fixed take profit percentage (0 to disable)")
    parser.add_argument("--fixed-sl", type=float, default=0, help="Fixed stop loss percentage (0 to disable)")
    
    # Signal settings
    parser.add_argument("--trend-following", action="store_true", help="Enable trend following mode")
    parser.add_argument("--enhanced-signals", action="store_true", help="Use enhanced signal generation")
    parser.add_argument("--signal-threshold", type=float, default=2, help="Signal confirmation threshold")
    parser.add_argument("--signal-cooldown", type=int, default=15, help="Signal cooldown in minutes")
    
    # Scalping settings
    parser.add_argument("--scalping", action="store_true", help="Enable scalping mode")
    parser.add_argument("--scalping-tp", type=float, default=0.5, help="Scalping take profit factor")
    parser.add_argument("--scalping-sl", type=float, default=0.8, help="Scalping stop loss factor")
    
    # ML settings
    parser.add_argument("--ml-signals", action="store_true", help="Use ML for signal generation")
    parser.add_argument("--ml-confidence", type=float, default=0.6, help="ML confidence threshold")
    parser.add_argument("--train-ml", action="store_true", help="Train ML model before trading")
    parser.add_argument("--retrain-interval", type=int, default=0, help="ML model retraining interval in hours (0 to disable)")
    
    # Position management
    parser.add_argument("--reassess", action="store_true", help="Periodically reassess positions")
    
    return parser.parse_args()

def main():
    """Main function to run the trading bot"""
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        print("Error: Binance API credentials not found in environment variables")
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file")
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create the trader instance
    trader = RealtimeTrader(
        symbol=args.symbol,
        initial_investment=args.investment,
        daily_profit_target=args.profit_target,
        leverage=args.leverage,
        test_mode=args.test_mode,
        use_full_investment=args.full_investment,
        use_full_margin=args.full_margin,
        compound_interest=args.compound,
        enable_pyramiding=args.pyramiding,
        max_pyramid_entries=args.max_pyramid,
        pyramid_threshold_pct=args.pyramid_threshold,
        use_dynamic_take_profit=args.dynamic_tp,
        trend_following_mode=args.trend_following,
        use_enhanced_signals=args.enhanced_signals,
        signal_confirmation_threshold=args.signal_threshold,
        signal_cooldown_minutes=args.signal_cooldown,
        use_scalping_mode=args.scalping,
        scalping_tp_factor=args.scalping_tp,
        scalping_sl_factor=args.scalping_sl,
        use_ml_signals=args.ml_signals,
        ml_confidence=args.ml_confidence,
        train_ml=args.train_ml,
        retrain_interval=args.retrain_interval,
        reassess_positions=args.reassess,
        fixed_tp=args.fixed_tp,
        fixed_sl=args.fixed_sl,
    )
    
    # Print configuration
    print("\n=== Trading Configuration ===")
    print(f"Symbol: {args.symbol}")
    print(f"Initial Investment: {args.investment} USDT")
    print(f"Leverage: {args.leverage}x")
    print(f"Duration: {args.duration} hours")
    print(f"Update Interval: {args.interval_minutes}m {args.interval_seconds}s")
    
    # Print trading modes
    modes = []
    if args.test_mode:
        modes.append("Test Mode")
    if args.full_investment:
        modes.append("Full Investment")
    if args.full_margin:
        modes.append("Full Margin")
    if args.compound:
        modes.append("Compound Interest")
    if args.pyramiding:
        modes.append(f"Pyramiding (max: {args.max_pyramid})")
    if args.dynamic_tp:
        modes.append("Dynamic Take Profit")
    if args.trend_following:
        modes.append("Trend Following")
    if args.enhanced_signals:
        modes.append("Enhanced Signals")
    if args.scalping:
        modes.append("Scalping")
    if args.ml_signals:
        modes.append("ML Signals")
    if args.train_ml:
        modes.append("ML Training")
    if args.reassess:
        modes.append("Position Reassessment")
    
    if modes:
        print(f"Enabled Modes: {', '.join(modes)}")
    
    # Print take profit and stop loss settings
    if args.fixed_tp > 0:
        print(f"Fixed Take Profit: {args.fixed_tp}%")
    if args.fixed_sl > 0:
        print(f"Fixed Stop Loss: {args.fixed_sl}%")
    
    # Confirm and start trading
    confirmation = input("\nStart trading with these settings? (y/n): ")
    if confirmation.lower() == "y":
        # Run the trading bot
        trader.run_real_trading(
            duration_hours=args.duration,
            update_interval_minutes=args.interval_minutes,
            update_interval_seconds=args.interval_seconds
        )
    else:
        print("Trading cancelled")

if __name__ == "__main__":
    main()
