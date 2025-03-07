import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import time

def save_results(simulator, results_dir, symbol, initial_investment):
    """Save simulation results to files"""
    try:
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save trades to CSV
        trades_df = pd.DataFrame(simulator.trades)
        trades_file = os.path.join(results_dir, f"{symbol}_trades.csv")
        trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve to CSV
        equity_data = []
        for trade in simulator.trades:
            equity_data.append({
                'timestamp': trade['exit_time'],
                'balance': trade['balance_after']
            })
        
        if equity_data:
            equity_df = pd.DataFrame(equity_data)
            equity_file = os.path.join(results_dir, f"{symbol}_equity.csv")
            equity_df.to_csv(equity_file, index=False)
        
        # Generate performance chart
        if len(simulator.trades) > 0:
            generate_performance_chart(simulator, results_dir, symbol, initial_investment)
        
        # Save summary statistics
        save_summary_stats(simulator, results_dir, symbol, initial_investment)
        
        print(f"Results saved to {results_dir}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def generate_performance_chart(simulator, results_dir, symbol, initial_investment):
    """Generate performance chart"""
    try:
        # Create equity curve data
        equity_data = [{
            'timestamp': datetime.now(),
            'balance': initial_investment
        }]
        
        for trade in simulator.trades:
            equity_data.append({
                'timestamp': trade['exit_time'],
                'balance': trade['balance_after']
            })
        
        equity_df = pd.DataFrame(equity_data)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['timestamp'], equity_df['balance'])
        plt.title(f"{symbol} Trading Performance")
        plt.xlabel("Date")
        plt.ylabel("Account Balance (USD)")
        plt.grid(True)
        
        # Add horizontal line for initial investment
        plt.axhline(y=initial_investment, color='r', linestyle='--', label='Initial Investment')
        
        # Add annotations for trades
        for trade in simulator.trades:
            if trade['profit'] > 0:
                color = 'g'  # Green for profit
            else:
                color = 'r'  # Red for loss
            
            plt.scatter(trade['exit_time'], trade['balance_after'], color=color, s=30)
        
        plt.legend()
        
        # Save chart
        chart_file = os.path.join(results_dir, f"{symbol}_performance.png")
        plt.savefig(chart_file)
        plt.close()
        
        print(f"Performance chart saved to {chart_file}")
        
    except Exception as e:
        print(f"Error generating performance chart: {e}")

def save_summary_stats(simulator, results_dir, symbol, initial_investment):
    """Save summary statistics"""
    try:
        # Calculate statistics
        total_trades = len(simulator.trades)
        winning_trades = sum(1 for trade in simulator.trades if trade['profit'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade['profit'] for trade in simulator.trades if trade['profit'] > 0)
        total_loss = sum(trade['profit'] for trade in simulator.trades if trade['profit'] <= 0)
        
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Create summary dictionary
        summary = {
            'symbol': symbol,
            'initial_investment': initial_investment,
            'final_balance': simulator.current_balance,
            'profit_loss': simulator.current_balance - initial_investment,
            'return_pct': ((simulator.current_balance - initial_investment) / initial_investment) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to JSON
        summary_file = os.path.join(results_dir, f"{symbol}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Summary statistics saved to {summary_file}")
        
    except Exception as e:
        print(f"Error saving summary statistics: {e}")

def display_countdown(next_run_time, interval_seconds, prefix="Next update in"):
    """
    Display a countdown timer until the next interval
    
    Args:
        next_run_time: Datetime of the next scheduled run
        interval_seconds: Interval in seconds
        prefix: Text to display before the countdown
    """
    try:
        now = datetime.now()
        remaining = (next_run_time - now).total_seconds()
        
        if remaining <= 0:
            # If time has passed, calculate the next interval
            elapsed = abs(remaining)
            intervals_passed = int(elapsed // interval_seconds) + 1
            next_run_time = next_run_time + timedelta(seconds=intervals_passed * interval_seconds)
            remaining = (next_run_time - now).total_seconds()
            print(f"Next update in {remaining:.0f} seconds")
            print("Countdown started...")
        
        # Format the remaining time
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        
        # Create a visual progress bar
        total_width = 20
        elapsed_width = int((interval_seconds - remaining) / interval_seconds * total_width)
        bar = "■" * elapsed_width + "□" * (total_width - elapsed_width)
        
        # Calculate progress percentage
        progress_pct = (interval_seconds - remaining) / interval_seconds
        
        # Display the countdown
        if hours > 0:
            countdown = f"{prefix}: {hours:02d}:{minutes:02d}:{seconds:02d} [{bar}] {progress_pct:.0%}"
        else:
            countdown = f"{prefix}: {minutes:02d}:{seconds:02d} [{bar}] {progress_pct:.0%}"
            
        print(countdown, end='\r', flush=True)
        
        # If countdown is complete (less than 1 second remaining)
        if remaining < 1:
            print("\nUpdate time reached!                                      ")  # Clear line with newline
            print(" " * 50, end='\r')  # Clear any remaining characters
            
        return next_run_time
        
    except Exception as e:
        print(f"\nError in countdown display: {e}")
        return next_run_time

def format_time_remaining(seconds):
    """
    Format seconds into a human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"