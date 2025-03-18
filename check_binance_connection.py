#!/usr/bin/env python3
"""
Utility script to check Binance API connectivity.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from realtime.trader.utils import test_binance_connection

def main():
    """Main function to test Binance API connectivity"""
    parser = argparse.ArgumentParser(description="Check Binance API connectivity")
    parser.add_argument("--testnet", action="store_true", help="Test connection to Binance Testnet")
    parser.add_argument("--key", type=str, help="Binance API key (overrides .env)")
    parser.add_argument("--secret", type=str, help="Binance API secret (overrides .env)")
    args = parser.parse_args()
    
    # Load API credentials from .env file
    load_dotenv()
    
    # Use command line args if provided, otherwise use .env
    api_key = args.key if args.key else os.getenv("BINANCE_API_KEY")
    api_secret = args.secret if args.secret else os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Binance API credentials not found.")
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file or provide them as arguments.")
        sys.exit(1)
    
    print(f"Testing connection to Binance {'Testnet' if args.testnet else 'Production'} API...")
    
    # Test connection
    results = test_binance_connection(api_key, api_secret, is_testnet=args.testnet)
    
    # Output results
    print("\n=== CONNECTION TEST RESULTS ===")
    if results['success']:
        print("✅ Connection test SUCCESSFUL")
    else:
        print("❌ Connection test FAILED")
    
    # Print errors
    if results['errors']:
        print("\n❌ ERRORS:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Print warnings
    if results['warnings']:
        print("\n⚠️ WARNINGS:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    # Print info
    if results['info']:
        print("\nℹ️ INFO:")
        for info in results['info']:
            print(f"  - {info}")
    
    # Print account data summary if available
    if results['account_data']:
        print("\n=== ACCOUNT SUMMARY ===")
        account = results['account_data']
        
        # Show futures wallet balance
        for asset in account.get('assets', []):
            if asset['asset'] == 'USDT':
                print(f"USDT Wallet Balance: {float(asset['walletBalance'])}")
                print(f"USDT Available Balance: {float(asset['availableBalance'])}")
                print(f"USDT Position Margin: {float(asset['positionInitialMargin'])}")
                break
        
        # Show positions
        positions = [pos for pos in account.get('positions', []) if float(pos['positionAmt']) != 0]
        if positions:
            print("\nOpen Positions:")
            for pos in positions:
                symbol = pos['symbol']
                amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                direction = "LONG" if amt > 0 else "SHORT"
                leverage = pos['leverage']
                print(f"  {symbol}: {direction} {abs(amt)} at {entry_price} (Leverage: {leverage}x)")
        else:
            print("\nNo open positions")
    
    # Return exit code
    return 0 if results['success'] else 1

if __name__ == "__main__":
    sys.exit(main())
