"""
Utilities for reporting, logging, and displaying trading information.
"""

import time
import sys
from datetime import datetime, timedelta

def format_time_remaining(seconds):
    """
    Format time remaining as HH:MM:SS
    
    Args:
        seconds: Time remaining in seconds
        
    Returns:
        Formatted time string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
    else:
        return f"{int(minutes):02d}:{int(seconds):02d}"

def display_countdown(next_run_time, interval_seconds, prefix_text="Next update in"):
    """
    Display a countdown timer
    
    Args:
        next_run_time: Datetime when the next event will occur
        interval_seconds: Interval between updates in seconds
        prefix_text: Text to display before the countdown
        
    Returns:
        datetime: Updated next run time
    """
    now = datetime.now()
    
    # If we've passed the next run time, update it
    if now >= next_run_time:
        next_run_time = now + timedelta(seconds=interval_seconds)
    
    # Calculate time remaining
    time_remaining = (next_run_time - now).total_seconds()
    
    # Format the countdown
    formatted_time = format_time_remaining(time_remaining)
    
    # Display the countdown
    sys.stdout.write(f"\r{prefix_text} {formatted_time}     ")
    sys.stdout.flush()
    
    # Small sleep to prevent CPU overuse
    time.sleep(0.1)
    
    return next_run_time

def print_trade_summary(trade):
    """
    Print a formatted trade summary
    
    Args:
        trade: Dictionary containing trade information
    """
    profit_color = "🟢" if trade.get('profit_amount', 0) >= 0 else "🔴"
    
    print("\n=== TRADE SUMMARY ===")
    print(f"Type: {trade.get('type', 'Unknown').upper()}")
    print(f"Entry Date: {trade.get('entry_date', 'Unknown')}")
    print(f"Exit Date: {trade.get('exit_date', 'Unknown')}")
    print(f"Entry Price: {trade.get('entry_price', 0):.2f}")
    print(f"Exit Price: {trade.get('exit_price', 0):.2f}")
    print(f"Position Size: {trade.get('position_size', 0):.6f}")
    print(f"{profit_color} Profit/Loss: {trade.get('profit_amount', 0):.2f} ({trade.get('profit_pct', 0):.2f}%)")
    print(f"Balance: {trade.get('balance', 0):.2f}")

def format_trades_summary(trades):
    """
    Format a summary of all trades
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        str: Formatted summary
    """
    if not trades:
        return "No trades executed"
    
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade.get('profit_amount', 0) > 0)
    losing_trades = sum(1 for trade in trades if trade.get('profit_amount', 0) <= 0)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = sum(trade.get('profit_amount', 0) for trade in trades if trade.get('profit_amount', 0) > 0)
    total_loss = sum(trade.get('profit_amount', 0) for trade in trades if trade.get('profit_amount', 0) <= 0)
    
    profit_factor = (total_profit / abs(total_loss)) if abs(total_loss) > 0 else float('inf')
    
    first_trade_date = trades[0].get('entry_date', 'Unknown') if trades else 'N/A'
    last_trade_date = trades[-1].get('exit_date', 'Unknown') if trades else 'N/A'
    
    summary = f"""
=== TRADING SUMMARY ===
Period: {first_trade_date} to {last_trade_date}
Total Trades: {total_trades}
Winning Trades: {winning_trades} ({win_rate:.2f}%)
Losing Trades: {losing_trades}
Total Profit: {total_profit:.2f}
Total Loss: {total_loss:.2f}
Net P/L: {total_profit + total_loss:.2f}
Profit Factor: {profit_factor:.2f}
"""
    return summary
