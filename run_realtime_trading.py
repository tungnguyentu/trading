#!/usr/bin/env python3
"""
Real-time Trading Runner

This script runs real-time trading using the Binance API.
"""

import os
import argparse
from dotenv import load_dotenv
from realtime.real_trader import RealtimeTrader
from datetime import datetime


def main():
    """Main function to run real-time trading"""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time Trading")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading pair to trade"
    )
    parser.add_argument(
        "--investment", type=float, default=50.0, help="Initial investment amount"
    )
    parser.add_argument(
        "--target", type=float, default=15.0, help="Daily profit target"
    )
    parser.add_argument("--hours", type=int, default=24, help="Duration in hours")
    parser.add_argument(
        "--interval", type=int, default=15, help="Update interval in minutes"
    )
    parser.add_argument(
        "--leverage", type=int, default=15, help="Margin trading leverage (15-20x)"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode with fake balance")

    args = parser.parse_args()

    # Check if API keys are set for real trading mode
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not args.test and (not api_key or not api_secret):
        print("ERROR: Binance API key and secret are required for real trading")
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        print("Or use --test flag to run in test mode with fake balance")
        return

    # Display warning and confirmation for real trading
    if not args.test:
        print("=" * 80)
        print("WARNING: You are about to start REAL TRADING with real money!")
        print(f"Symbol: {args.symbol}")
        print(f"Initial Investment: ${args.investment}")
        print(f"Leverage: {args.leverage}x")
        print(f"Duration: {args.hours} hours")
        print(f"Update Interval: {args.interval} minutes")
        print("=" * 80)

        confirmation = input(
            "Are you sure you want to proceed with real trading? (yes/no): "
        )

        if confirmation.lower() != "yes":
            print("Trading cancelled")
            return
    else:
        print("=" * 80)
        print("Starting in TEST MODE with fake balance")
        print(f"Symbol: {args.symbol}")
        print(f"Initial Investment: ${args.investment}")
        print(f"Leverage: {args.leverage}x")
        print(f"Duration: {args.hours} hours")
        print(f"Update Interval: {args.interval} minutes")
        print("=" * 80)

    # Initialize real-time trader
    trader = RealtimeTrader(
        symbol=args.symbol,
        initial_investment=args.investment,
        daily_profit_target=args.target,
        leverage=args.leverage,
        test_mode=args.test
    )

    # Run real-time trading
    try:
        print(f"Starting {'test' if args.test else 'real'} trading...")
        result = trader.run_real_trading(
            duration_hours=args.hours, update_interval_minutes=args.interval
        )

        # Display final results
        if result:
            print("\n=== Final Results ===")
            print(f"Final Balance: ${result['final_balance']:.2f}")
            print(f"Profit/Loss: ${result['profit_loss']:.2f}")
            print(f"Return: {result['return_pct']:.2f}%")

    except KeyboardInterrupt:
        print("\nTrading interrupted by user")

        # Close any open positions
        if trader.has_open_position():
            print("Closing open position...")
            current_price = trader.get_current_price()
            close_result = trader.close_position(current_price, datetime.now(), "user_interrupt")

            if close_result:
                print(f"Position closed: {close_result}")

    except Exception as e:
        print(f"Error in trading: {e}")

    finally:
        print("Trading session ended")


if __name__ == "__main__":
    main()
