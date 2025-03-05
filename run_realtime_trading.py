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
from binance.exceptions import BinanceAPIException


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
    parser.add_argument("--full-investment", action="store_true", help="Use full investment amount for each trade (higher risk)")
    parser.add_argument("--full-margin", action="store_true", help="Use full investment amount as margin (EXTREME RISK)")
    parser.add_argument("--compound", action="store_true", help="Enable compound interest (use profits to increase position sizes)")
    
    # Add new advanced strategy arguments
    parser.add_argument("--enhanced-signals", action="store_true", help="Use enhanced signal generation with multiple indicators")
    parser.add_argument("--signal-threshold", type=int, default=2, help="Number of indicators required to confirm a signal (1-8)")
    parser.add_argument("--signal-cooldown", type=int, default=15, help="Minimum time between signals in minutes (5-60)")
    parser.add_argument("--trend-following", action="store_true", help="Only trade in the direction of the overall market trend")
    parser.add_argument("--pyramiding", action="store_true", help="Enable pyramiding (adding to winning positions)")
    parser.add_argument("--pyramid-entries", type=int, default=2, help="Maximum number of pyramid entries (1-5)")
    parser.add_argument("--pyramid-threshold", type=float, default=1.0, help="Profit percentage required before pyramiding (0.5-5.0)")
    parser.add_argument("--dynamic-tp", action="store_true", help="Use dynamic take profit targets based on market conditions")

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
        if args.full_margin:
            print("⚠️ EXTREME RISK: Using FULL MARGIN mode - Your entire investment will be used as margin!")
            print("⚠️ This mode can lead to rapid liquidation of your position!")
        elif args.full_investment:
            print("CAUTION: Using FULL INVESTMENT mode - HIGHER RISK!")
        if args.compound:
            print("📈 Compound interest enabled - Position sizes will increase as profits grow")
        if args.enhanced_signals:
            print(f"🎯 Enhanced signals enabled - Requiring {args.signal_threshold} confirming indicators")
            print(f"⏱️ Signal cooldown set to {args.signal_cooldown} minutes between trades")
        if args.trend_following:
            print("📊 Trend following enabled - Only trading with the market trend")
        if args.pyramiding:
            print(f"🔺 Pyramiding enabled - Up to {args.pyramid_entries} additional entries at {args.pyramid_threshold}% profit")
        if args.dynamic_tp:
            print("💰 Dynamic take profit enabled - Adjusting targets based on market conditions")
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
        if args.full_margin:
            print("⚠️ EXTREME RISK: Using FULL MARGIN mode - Your entire investment will be used as margin!")
            print("⚠️ This mode can lead to rapid liquidation of your position!")
        elif args.full_investment:
            print("CAUTION: Using FULL INVESTMENT mode - HIGHER RISK!")
        if args.compound:
            print("📈 Compound interest enabled - Position sizes will increase as profits grow")
        if args.enhanced_signals:
            print(f"🎯 Enhanced signals enabled - Requiring {args.signal_threshold} confirming indicators")
            print(f"⏱️ Signal cooldown set to {args.signal_cooldown} minutes between trades")
        if args.trend_following:
            print("📊 Trend following enabled - Only trading with the market trend")
        if args.pyramiding:
            print(f"🔺 Pyramiding enabled - Up to {args.pyramid_entries} additional entries at {args.pyramid_threshold}% profit")
        if args.dynamic_tp:
            print("💰 Dynamic take profit enabled - Adjusting targets based on market conditions")
        print("=" * 80)

    # Initialize real-time trader
    trader = RealtimeTrader(
        symbol=args.symbol,
        initial_investment=args.investment,
        daily_profit_target=args.target,
        leverage=args.leverage,
        test_mode=args.test,
        use_full_investment=args.full_investment,
        use_full_margin=args.full_margin,
        compound_interest=args.compound,
        enable_pyramiding=args.pyramiding,
        max_pyramid_entries=args.pyramid_entries,
        pyramid_threshold_pct=args.pyramid_threshold/100,  # Convert percentage to decimal
        use_dynamic_take_profit=args.dynamic_tp,
        trend_following_mode=args.trend_following,
        use_enhanced_signals=args.enhanced_signals,
        signal_confirmation_threshold=args.signal_threshold,
        signal_cooldown_minutes=args.signal_cooldown
    )
    # Run real-time trading
    try:
        print(f"Starting {'test' if args.test else 'real'} trading...")
        result = trader.run_real_trading(
            duration_hours=args.hours,
            update_interval_minutes=args.interval
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
    
    except BinanceAPIException as e:
        if "Precision is over the maximum defined for this asset" in str(e):
            print("\nERROR: Precision is over the maximum defined for this asset.")
            print("This error occurs when the quantity precision is too high for the asset.")
            print("Try increasing your investment amount or reducing the risk per trade.")
            print("You can also try using the --test flag to run in test mode first.")
        else:
            print(f"\nBinance API Error: {e}")
    
    except Exception as e:
        print(f"Error in trading: {e}")
    
    finally:
        print("Trading session ended")


if __name__ == "__main__":
    main()
