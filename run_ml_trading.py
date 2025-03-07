#!/usr/bin/env python3
"""
Script to run real-time trading using trained ML models.
This script integrates the ML training system with the real-time trading system.
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import ML trainer
from realtime.trainer.ml_trainer import MLTrainer

# Import trading components
from realtime.trader import RealtimeTrader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ML-based real-time trading")

    # Trading parameters
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--investment",
        type=float,
        default=50.0,
        help="Initial investment amount in USDT",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        default=15.0,
        help="Daily profit target percentage",
    )
    parser.add_argument(
        "--leverage", type=int, default=5, help="Leverage for futures trading"
    )

    # Duration parameters
    parser.add_argument(
        "--duration", type=int, default=24, help="Trading duration in hours"
    )
    parser.add_argument(
        "--interval-minutes", type=int, default=15, help="Update interval in minutes"
    )

    # ML parameters
    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["rf", "gb", "ensemble"],
        help="ML model type (rf=Random Forest, gb=Gradient Boosting, ensemble=Ensemble)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Timeframe for ML model training",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Number of days of historical data to use for training",
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Optimize ML model hyperparameters"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining of ML model even if it exists",
    )
    parser.add_argument(
        "--ml-confidence",
        type=float,
        default=0.6,
        help="ML confidence threshold for trading signals",
    )
    parser.add_argument(
        "--retrain-interval",
        type=int,
        default=0,
        help="ML model retraining interval in hours (0 to disable)",
    )
    parser.add_argument(
        "--signal-cooldown",
        type=int,
        default=15,
        help="Signal cooldown in minutes",
    )
    # Trading modes
    parser.add_argument(
        "--test-mode", action="store_true", help="Run in test mode without real trades"
    )
    parser.add_argument(
        "--backtest-first", action="store_true", help="Run backtest before live trading"
    )
    parser.add_argument(
        "--compound", action="store_true", help="Enable compound interest"
    )
    parser.add_argument(
        "--full-margin", action="store_true", help="Use full margin for trading"
    )

    # Take profit and stop loss settings
    parser.add_argument(
        "--dynamic-tp",
        action="store_true",
        help="Use dynamic take profit based on market conditions",
    )
    parser.add_argument(
        "--fixed-tp",
        type=float,
        default=0,
        help="Fixed take profit percentage (0 to disable)",
    )
    parser.add_argument(
        "--fixed-sl",
        type=float,
        default=0,
        help="Fixed stop loss percentage (0 to disable)",
    )

    return parser.parse_args()


def train_and_backtest(args):
    """
    Train ML model and run backtest

    Args:
        args: Command line arguments

    Returns:
        tuple: (model, scaler, backtest_results)
    """
    print("\n=== Training and Backtesting ML Model ===")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Model Type: {args.model_type}")
    print(f"Lookback Period: {args.lookback_days} days")
    print(f"Optimize Hyperparameters: {args.optimize}")
    print(f"Force Retrain: {args.force_retrain}")

    # Initialize ML trainer
    trainer = MLTrainer(
        symbols=[args.symbol],
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
    )

    # Train and backtest model
    backtest_results = trainer.train_and_backtest(
        args.symbol,
        model_type=args.model_type,
        force_retrain=args.force_retrain,
        optimize_hyperparams=args.optimize,
    )

    if backtest_results is None:
        print("Failed to train and backtest model")
        return None, None, None

    # Get trained model and scaler
    model = trainer.ml_manager.models.get(args.symbol)
    scaler = trainer.ml_manager.scalers.get(args.symbol)

    return model, scaler, backtest_results


def main():
    """Main function to run ML-based trading"""
    # Load environment variables
    load_dotenv()

    # Check for API keys
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        print("Error: Binance API credentials not found in environment variables")
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file")
        return

    # Parse command line arguments
    args = parse_arguments()

    # Train and backtest if requested
    if args.backtest_first or args.force_retrain:
        model, scaler, backtest_results = train_and_backtest(args)

        if backtest_results:
            print("\n=== Backtest Results ===")

            # Ensure all required keys exist
            if (
                "total_return_pct" not in backtest_results
                and "total_return" in backtest_results
            ):
                backtest_results["total_return_pct"] = backtest_results["total_return"]

            if (
                "max_drawdown_pct" not in backtest_results
                and "max_drawdown" in backtest_results
            ):
                backtest_results["max_drawdown_pct"] = backtest_results["max_drawdown"]

            if "sharpe_ratio" not in backtest_results:
                # Estimate Sharpe ratio
                if backtest_results.get("max_drawdown_pct", 0) > 0:
                    backtest_results["sharpe_ratio"] = backtest_results.get(
                        "total_return_pct", 0
                    ) / backtest_results.get("max_drawdown_pct", 1)
                else:
                    backtest_results["sharpe_ratio"] = 0.0

            # Print results with safe access to keys
            print(f"Total Return: {backtest_results.get('total_return_pct', 0):.2f}%")
            print(f"Win Rate: {backtest_results.get('win_rate', 0):.2f}%")
            print(f"Profit Factor: {backtest_results.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {backtest_results.get('max_drawdown_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")

            # Ask for confirmation to proceed with live trading
            if not args.test_mode:
                confirmation = input(
                    "\nProceed with live trading based on these results? (y/n): "
                )
                if confirmation.lower() != "y":
                    print("Trading cancelled")
                    return

    # Create the trader instance
    trader = RealtimeTrader(
        symbol=args.symbol,
        initial_investment=args.investment,
        daily_profit_target=args.profit_target,
        leverage=args.leverage,
        test_mode=args.test_mode,
        use_full_investment=False,
        use_full_margin=args.full_margin,
        compound_interest=args.compound,
        enable_pyramiding=False,
        max_pyramid_entries=2,
        pyramid_threshold_pct=0.01,
        use_dynamic_take_profit=args.dynamic_tp,
        trend_following_mode=True,
        use_enhanced_signals=True,
        signal_confirmation_threshold=2,
        signal_cooldown_minutes=args.signal_cooldown,
        use_scalping_mode=False,
        scalping_tp_factor=0.5,
        scalping_sl_factor=0.8,
        use_ml_signals=True,  # Always use ML signals in this script
        ml_confidence=args.ml_confidence,
        train_ml=args.force_retrain,
        retrain_interval=args.retrain_interval,
        reassess_positions=True,
        fixed_tp=args.fixed_tp,
        fixed_sl=args.fixed_sl,
    )

    # Print configuration
    print("\n=== Trading Configuration ===")
    print(f"Symbol: {args.symbol}")
    print(f"Initial Investment: {args.investment} USDT")
    print(f"Leverage: {args.leverage}x")
    print(f"Duration: {args.duration} hours")
    print(f"Update Interval: {args.interval_minutes}m")
    print(f"ML Confidence Threshold: {args.ml_confidence}")

    # Print trading modes
    modes = ["ML Signals"]  # Always using ML signals
    if args.test_mode:
        modes.append("Test Mode")
    if args.compound:
        modes.append("Compound Interest")
    if args.dynamic_tp:
        modes.append("Dynamic Take Profit")
    if args.force_retrain:
        modes.append("ML Training")
    if args.retrain_interval > 0:
        modes.append(f"ML Retraining ({args.retrain_interval}h)")

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
            update_interval_seconds=0,
        )
    else:
        print("Trading cancelled")


if __name__ == "__main__":
    main()
