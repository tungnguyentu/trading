#!/usr/bin/env python3
"""
Migration script to help users migrate from the old monolithic structure to the new modular structure.
This script doesn't actually move any files, but provides guidance on how to use the new structure.
"""

import os
import sys

def print_header(text):
    """Print a header with decoration"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_section(text):
    """Print a section header"""
    print("\n" + "-" * 80)
    print(f" {text} ".center(80, "-"))
    print("-" * 80)

def check_files():
    """Check if the necessary files exist"""
    required_files = [
        "realtime/trader/__init__.py",
        "realtime/trader/core.py",
        "realtime/trader/account.py",
        "realtime/trader/signals.py",
        "realtime/trader/position.py",
        "realtime/trader/orders.py",
        "realtime/trader/ml_integration.py",
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def main():
    """Main function"""
    print_header("Migration Guide: Monolithic to Modular Structure")
    
    # Check if the new structure files exist
    missing_files = check_files()
    if missing_files:
        print("\n⚠️  Some files from the new structure are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease make sure all the necessary files are in place before continuing.")
        return
    
    print("\n🎉 All necessary files for the new modular structure are in place!")
    
    print_section("Why Migrate?")
    print("""
The new modular structure offers several advantages:

1. Better code organization and maintainability
2. Easier to understand and modify specific components
3. Improved testability
4. More flexible for future extensions
5. Reduced file sizes for better readability
""")
    
    print_section("How to Use the New Structure")
    print("""
Instead of importing from 'realtime.real_trader', you should now import from 'realtime.trader':

OLD: from realtime.real_trader import RealtimeTrader
NEW: from realtime.trader import RealtimeTrader

The RealtimeTrader class has the same interface, so your existing code should work
with minimal changes.
""")
    
    print_section("Module Overview")
    print("""
The functionality has been split into these modules:

- core.py: Main RealtimeTrader class that orchestrates everything
- account.py: Account management (balances, market data, symbol info)
- signals.py: Signal generation and technical analysis
- position.py: Position management (open/close positions, TP/SL)
- orders.py: Order execution (placing orders, calculating position sizes)
- ml_integration.py: Machine learning integration
""")
    
    print_section("Example Usage")
    print("""
# Basic usage remains the same
from realtime.trader import RealtimeTrader

trader = RealtimeTrader(
    symbol="BTCUSDT",
    initial_investment=100.0,
    leverage=5,
    # ... other parameters ...
)

# Run trading
trader.run_real_trading(
    duration_hours=24,
    update_interval_minutes=15
)
""")
    
    print_section("Advanced Usage")
    print("""
# You can also access the individual components directly
from realtime.trader import RealtimeTrader

trader = RealtimeTrader(...)

# Get account information
balance = trader.account_manager.get_account_balance()
current_price = trader.account_manager.get_current_price()

# Generate signals
signal = trader.signal_generator.generate_trading_signal()

# Check positions
has_position = trader.position_manager.has_open_position()
position_info = trader.position_manager.get_position_info()

# Execute trades
trader.order_executor.execute_trade("LONG", current_price, "2023-01-01 12:00:00")
""")
    
    print_section("Next Steps")
    print("""
1. Update your imports to use the new structure
2. Run your existing scripts with the new imports
3. Explore the individual modules to understand the functionality better
4. Consider using the component classes directly for more advanced use cases

The old 'real_trader.py' file is still available for backward compatibility,
but it's recommended to migrate to the new structure for future development.
""")
    
    print("\nHappy trading! 🚀")

if __name__ == "__main__":
    main() 