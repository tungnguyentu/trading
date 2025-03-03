import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from ml_models import MLManager

# Add to imports at the top
import telegram
import asyncio
from dotenv import load_dotenv

# Load environment variables for Telegram
load_dotenv()

class TradingSimulator:
    def __init__(self, historical_data_or_symbol, initial_investment=50.0, daily_profit_target=15.0, leverage=1):
        """
        Initialize the trading simulator
        
        Args:
            historical_data_or_symbol: DataFrame with historical price data or a symbol string
            initial_investment: Starting capital in USD
            daily_profit_target: Target profit per day in USD
            leverage: Margin trading leverage (default: 1 = no leverage)
        """
        # Check if historical_data is a DataFrame or a symbol string
        if isinstance(historical_data_or_symbol, pd.DataFrame):
            self.data = historical_data_or_symbol.copy()
            self.symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'SIMULATION'
        else:
            # It's a symbol string
            self.data = pd.DataFrame()  # Empty DataFrame, will be populated later
            self.symbol = historical_data_or_symbol
            
        self.initial_investment = initial_investment
        self.current_balance = initial_investment
        self.daily_profit_target = daily_profit_target
        self.leverage = max(1, min(20, leverage))  # Constrain leverage between 1-20x
        
        # Trading state
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # Performance tracking
        self.trades = []
        self.daily_profits = {}
        self.equity_curve = []
        
        # Risk management
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.max_position_pct = 0.95  # Max percentage of balance to use in a trade
        
        # Results directory
        self.results_dir = os.path.join(os.getcwd(), "simulation_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Telegram notification setup
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.notifications_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        
        if self.notifications_enabled:
            print("Telegram notifications enabled")
        else:
            print("Telegram notifications disabled - missing bot token or chat ID")
            
        print(f"Trading simulator initialized with {self.leverage}x leverage")
    
    async def send_telegram_notification(self, message):
        """Send notification to Telegram"""
        if not self.notifications_enabled:
            return
            
        try:
            bot = telegram.Bot(token=self.telegram_bot_token)
            await bot.send_message(chat_id=self.telegram_chat_id, text=message)
            print(f"Telegram notification sent: {message}")
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
    
    def send_notification(self, message):
        """Synchronous wrapper for send_telegram_notification"""
        if not self.notifications_enabled:
            return
            
        try:
            # Create a new event loop for each notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the coroutine and close the loop properly
            loop.run_until_complete(self.send_telegram_notification(message))
            loop.close()
        except Exception as e:
            print(f"Error in send_notification: {e}")
    
    def calculate_position_size(self, price):
        """Calculate the position size based on current balance, leverage and max position percentage"""
        max_position_value = self.current_balance * self.max_position_pct * self.leverage
        position_size = max_position_value / price
        
        # Round to 6 decimal places to avoid precision issues
        return round(position_size, 6)

    def execute_trade(self, signal, price, timestamp):
        """Execute a trade based on the signal"""
        # Skip if we already have a position in the same direction
        if (signal == 1 and self.position == 'long') or (signal == -1 and self.position == 'short'):
            return None
            
        # Close existing position if we have one
        if self.position:
            if self.position == 'long':
                profit = self.position_size * (price - self.entry_price)
                self.current_balance += profit
                self.record_trade('long', self.entry_price, price, self.entry_time, timestamp, profit, 'signal_change')
                
                result = f"CLOSED LONG: {self.position_size:.6f} units at ${price:.2f}, Profit: ${profit:.2f}"
            else:  # short
                profit = self.position_size * (self.entry_price - price)
                self.current_balance += profit
                self.record_trade('short', self.entry_price, price, self.entry_time, timestamp, profit, 'signal_change')
                
                result = f"CLOSED SHORT: {self.position_size:.6f} units at ${price:.2f}, Profit: ${profit:.2f}"
                
            self.position = None
            self.entry_price = None
            self.position_size = 0
            
        else:
            result = None
            
        # Open new position
        if signal != 0:
            self.position_size = self.calculate_position_size(price)
            self.entry_price = price
            self.entry_time = timestamp
            
            # Calculate actual investment (position value divided by leverage)
            actual_investment = (self.position_size * price) / self.leverage
            
            if signal == 1:
                self.position = 'long'
                result = f"BUY: {self.position_size:.6f} units at ${price:.2f}"
                
                # Send notification for opening long position
                notification = (
                    f"🟢 BUY Signal for simulation\n"
                    f"Symbol: {self.symbol}\n"
                    f"Entry Price: ${price:.2f}\n"
                    f"Position Size: {self.position_size:.6f} units\n"
                    f"Leverage: {self.leverage}x\n"
                    f"Investment: ${actual_investment:.2f}\n"
                    f"Effective Position: ${(self.position_size * price):.2f}"
                )
                
            else:  # signal == -1
                self.position = 'short'
                result = f"SELL: {self.position_size:.6f} units at ${price:.2f}"
                
                # Send notification for opening short position
                notification = (
                    f"🔴 SELL Signal for simulation\n"
                    f"Symbol: {self.symbol}\n"
                    f"Entry Price: ${price:.2f}\n"
                    f"Position Size: {self.position_size:.6f} units\n"
                    f"Leverage: {self.leverage}x\n"
                    f"Investment: ${actual_investment:.2f}\n"
                    f"Effective Position: ${(self.position_size * price):.2f}"
                )
            
            # Send notification if available
            if hasattr(self, 'notifications_enabled') and self.notifications_enabled:
                try:
                    # Create a new event loop for each notification
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Define the async function
                    async def send_message():
                        bot = telegram.Bot(token=self.telegram_bot_token)
                        await bot.send_message(chat_id=self.telegram_chat_id, text=notification)
                    
                    # Run the coroutine and close the loop properly
                    loop.run_until_complete(send_message())
                    loop.close()
                    print(f"Telegram notification sent: {notification}")
                except Exception as e:
                    print(f"Error sending Telegram notification: {e}")
            
        return result
    
    def check_take_profit_stop_loss(self, price, timestamp):
        """Check if take profit or stop loss is triggered"""
        if not self.position or not self.entry_price:
            return None
            
        # For long positions
        if self.position == 'long':
            # Check daily profit target
            current_profit = self.position_size * (price - self.entry_price)
            day_key = timestamp.strftime('%Y-%m-%d')
            day_profit = self.daily_profits.get(day_key, 0) + current_profit
            
            # Take profit if daily target reached or price hits take profit level
            if day_profit >= self.daily_profit_target or (self.take_profit_price and price >= self.take_profit_price):
                profit = self.position_size * (price - self.entry_price)
                self.current_balance += profit
                reason = 'daily_target' if day_profit >= self.daily_profit_target else 'take_profit'
                self.record_trade('long', self.entry_price, price, 
                                 self.entry_time, timestamp, profit, reason)
                
                # Always send notification for take profit
                notification = (
                    f"💰 TAKE PROFIT (LONG) for simulation\n"
                    f"Symbol: {self.symbol}\n"
                    f"Exit Price: ${price:.2f}\n"
                    f"Profit: ${profit:.2f} ({(price - self.entry_price) / self.entry_price:.2%})\n"
                    f"Reason: {'Daily Target Met' if reason == 'daily_target' else 'Take Profit Level'}\n"
                    f"Balance: ${self.current_balance:.2f}"
                )
                self.send_notification(notification)
                
                self.position = None
                self.entry_price = None
                self.position_size = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return f"TAKE PROFIT (LONG): Closed at ${price:.2f}, Profit: ${profit:.2f}"
                
            # Stop loss
            elif self.stop_loss_price and price <= self.stop_loss_price:
                profit = self.position_size * (price - self.entry_price)
                self.current_balance += profit
                self.record_trade('long', self.entry_price, price, 
                                 self.entry_time, timestamp, profit, 'stop_loss')
                
                # Send notification for stop loss
                notification = (
                    f"🛑 STOP LOSS (LONG) for simulation\n"
                    f"Symbol: {self.symbol}\n"
                    f"Exit Price: ${price:.2f}\n"
                    f"Loss: ${profit:.2f} ({(price - self.entry_price) / self.entry_price:.2%})\n"
                    f"Balance: ${self.current_balance:.2f}"
                )
                self.send_notification(notification)
                
                self.position = None
                self.entry_price = None
                self.position_size = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return f"STOP LOSS (LONG): Closed at ${price:.2f}, Loss: ${profit:.2f}"
        
        # For short positions
        elif self.position == 'short':
            # Check daily profit target
            current_profit = self.position_size * (self.entry_price - price)
            day_key = timestamp.strftime('%Y-%m-%d')
            day_profit = self.daily_profits.get(day_key, 0) + current_profit
            
            # Take profit if daily target reached or price hits take profit level
            if day_profit >= self.daily_profit_target or (self.take_profit_price and price <= self.take_profit_price):
                profit = self.position_size * (self.entry_price - price)
                self.current_balance += profit
                reason = 'daily_target' if day_profit >= self.daily_profit_target else 'take_profit'
                self.record_trade('short', self.entry_price, price, 
                                 self.entry_time, timestamp, profit, reason)
                
                # Send notification for take profit
                notification = (
                    f"💰 TAKE PROFIT (SHORT) for simulation\n"
                    f"Symbol: {self.symbol}\n"
                    f"Exit Price: ${price:.2f}\n"
                    f"Profit: ${profit:.2f} ({(self.entry_price - price) / self.entry_price:.2%})\n"
                    f"Reason: {'Daily Target Met' if reason == 'daily_target' else 'Take Profit Level'}\n"
                    f"Balance: ${self.current_balance:.2f}"
                )
                self.send_notification(notification)
                
                self.position = None
                self.entry_price = None
                self.position_size = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return f"TAKE PROFIT (SHORT): Closed at ${price:.2f}, Profit: ${profit:.2f}"
                
            # Stop loss
            elif self.stop_loss_price and price >= self.stop_loss_price:
                profit = self.position_size * (self.entry_price - price)
                self.current_balance += profit
                self.record_trade('short', self.entry_price, price, 
                                 self.entry_time, timestamp, profit, 'stop_loss')
                
                # Send notification for stop loss
                notification = (
                    f"🛑 STOP LOSS (SHORT) for simulation\n"
                    f"Symbol: {self.symbol}\n"
                    f"Exit Price: ${price:.2f}\n"
                    f"Loss: ${profit:.2f} ({(self.entry_price - price) / self.entry_price:.2%})\n"
                    f"Balance: ${self.current_balance:.2f}"
                )
                self.send_notification(notification)
                
                self.position = None
                self.entry_price = None
                self.position_size = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return f"STOP LOSS (SHORT): Closed at ${price:.2f}, Loss: ${profit:.2f}"
                
        return None
    
    def record_trade(self, trade_type, entry_price, exit_price, entry_time, exit_time, profit, reason):
        """Record a completed trade"""
        # Always use today's date (when script is run)
        current_date = datetime.now()
        
        # Format the entry and exit times with current date but keep original time
        entry_time_str = f"{current_date.strftime('%Y-%m-%d')} {entry_time.strftime('%H:%M:%S')}"
        exit_time_str = f"{current_date.strftime('%Y-%m-%d')} {exit_time.strftime('%H:%M:%S')}"
        
        trade = {
            'type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time_str,
            'exit_time': exit_time_str,
            'profit': profit,
            'reason': reason,
            'profit_usd': profit  # Add this for compatibility with reporting
        }
        
        self.trades.append(trade)
        
        # Update daily profits - use current date only
        day = current_date.strftime('%Y-%m-%d')
        if day not in self.daily_profits:
            self.daily_profits[day] = 0
        self.daily_profits[day] += profit
        
        # Update equity curve with current date
        self.equity_curve.append({
            'timestamp': current_date,
            'balance': self.current_balance
        })
    
    def run_simulation(self, ml_manager=None):
        """Run the simulation on historical data"""
        print(f"Starting simulation with ${self.initial_investment:.2f}")
        
        # Reset state
        self.current_balance = self.initial_investment
        self.position = None
        self.entry_price = None
        self.trades = []
        self.daily_profits = {}
        self.equity_curve = [{
            'timestamp': datetime.now(),  # Use current date instead of historical
            'balance': self.current_balance
        }]
        
        # Temporarily disable notifications for historical simulation
        original_notifications_enabled = self.notifications_enabled
        self.notifications_enabled = False
        
        # Process each candle
        for i in range(1, len(self.data)):
            row = self.data.iloc[i]
            price = row['close']
            timestamp = row['timestamp']
            
            # Check take profit / stop loss first
            tp_sl_result = self.check_take_profit_stop_loss(price, timestamp)
            if tp_sl_result:
                # Use current date for display
                current_time = datetime.now().strftime('%Y-%m-%d')
                candle_time = timestamp.strftime('%H:%M:%S')
                print(f"{current_time} {candle_time}: {tp_sl_result}")
                continue
            
            # Get signal
            if ml_manager:
                # Use ML signal if available
                traditional_signal = row['signal']
                ml_signal, ml_confidence = ml_manager.get_ml_signal(
                    'simulation', self.data.iloc[:i+1])
                
                # Combine signals
                signal = 0
                if traditional_signal == ml_signal:
                    signal = traditional_signal
                elif ml_confidence > 0.75:
                    signal = ml_signal
            else:
                # Use only traditional signal
                signal = row['signal']
            
            # Execute trade if there's a signal
            if signal != 0:
                trade_result = self.execute_trade(signal, price, timestamp)
                if trade_result:
                    # Use current date for display
                    current_time = datetime.now().strftime('%Y-%m-%d')
                    candle_time = timestamp.strftime('%H:%M:%S')
                    print(f"{current_time} {candle_time}: {trade_result}")
        
        # Close any open position at the end
        if self.position:
            last_row = self.data.iloc[-1]
            last_price = last_row['close']
            last_time = last_row['timestamp']
            
            if self.position == 'long':
                profit = self.position_size * (last_price - self.entry_price)
                self.current_balance += profit
                self.record_trade('long', self.entry_price, last_price, 
                                 self.entry_time, last_time, profit, 'simulation_end')
            else:  # short
                profit = self.position_size * (self.entry_price - last_price)
                self.current_balance += profit
                self.record_trade('short', self.entry_price, last_price, 
                                 self.entry_time, last_time, profit, 'simulation_end')
            
            print(f"Closing {self.position} position at end of simulation: ${profit:.2f}")
            self.position = None
            self.entry_price = None
        
        # Generate reports
        self.generate_reports()
        
        return self.current_balance, self.trades, self.daily_profits
    
    def generate_reports(self):
        """Generate performance reports and charts"""
        # Calculate performance metrics
        total_profit = self.current_balance - self.initial_investment
        profit_pct = (total_profit / self.initial_investment) * 100
        num_trades = len(self.trades)
        
        winning_trades = [t for t in self.trades if t['profit_usd'] > 0]
        losing_trades = [t for t in self.trades if t['profit_usd'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        avg_profit = sum(t['profit_usd'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['profit_usd'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Print summary
        print("\n=== SIMULATION RESULTS ===")
        print(f"Initial Investment: ${self.initial_investment:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total Profit/Loss: ${total_profit:.2f} ({profit_pct:.2f}%)")
        print(f"Number of Trades: {num_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Profit: ${avg_profit:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        
        # Daily profits
        print("\n=== DAILY PROFITS ===")
        for day, profit in sorted(self.daily_profits.items()):
            target_met = "✅" if profit >= self.daily_profit_target else "❌"
            print(f"{day}: ${profit:.2f} {target_met}")
        
        days_met_target = sum(1 for p in self.daily_profits.values() if p >= self.daily_profit_target)
        total_days = len(self.daily_profits)
        print(f"Days Met Target: {days_met_target}/{total_days} ({days_met_target/total_days:.2%})")
        
        # Create equity curve chart
        self._plot_equity_curve()
        
        # Create daily profit chart
        self._plot_daily_profits()
        
        # Save trade log
        self._save_trade_log()
    
    def _plot_equity_curve(self):
        """Plot equity curve"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['timestamp'], equity_df['balance'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Account Balance ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/equity_curve.png")
        plt.close()
    
    def _plot_daily_profits(self):
        """Plot daily profits"""
        daily_profit_df = pd.DataFrame([
            {'date': pd.to_datetime(day), 'profit': profit} 
            for day, profit in self.daily_profits.items()
        ])
        daily_profit_df = daily_profit_df.sort_values('date')
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(daily_profit_df['date'], daily_profit_df['profit'])
        
        # Color bars based on profit/loss
        for i, bar in enumerate(bars):
            if daily_profit_df.iloc[i]['profit'] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Add target line
        plt.axhline(y=self.daily_profit_target, color='r', linestyle='--', 
                   label=f'Target (${self.daily_profit_target:.2f})')
        
        plt.title('Daily Profit/Loss')
        plt.xlabel('Date')
        plt.ylabel('Profit/Loss ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/daily_profits.png")
        plt.close()
    
    def _save_trade_log(self):
        """Save detailed trade log to CSV"""
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(f"{self.results_dir}/trade_log.csv", index=False)
        
        # Also save summary stats
        summary = {
            'initial_investment': self.initial_investment,
            'final_balance': self.current_balance,
            'total_profit': self.current_balance - self.initial_investment,
            'profit_pct': ((self.current_balance - self.initial_investment) / self.initial_investment) * 100,
            'num_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t['profit_usd'] > 0]) / len(self.trades) if self.trades else 0,
            'daily_profit_target': self.daily_profit_target,
            'days_met_target': sum(1 for p in self.daily_profits.values() if p >= self.daily_profit_target),
            'total_days': len(self.daily_profits)
        }
        
        # Save summary to CSV
        pd.DataFrame([summary]).to_csv(f"{self.results_dir}/summary.csv", index=False)
        
    def open_long_position(self, price, position_size, stop_loss_price, take_profit_price, timestamp):
        """Open a long position"""
        # Don't open if we already have a position
        if self.position:
            return False
            
        self.position = 'long'
        self.entry_price = price
        self.position_size = position_size
        self.entry_time = timestamp
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        
        return True
    
    def open_short_position(self, price, position_size, stop_loss_price, take_profit_price, timestamp):
        """Open a short position"""
        # Don't open if we already have a position
        if self.position:
            return False
            
        self.position = 'short'
        self.entry_price = price
        self.position_size = position_size
        self.entry_time = timestamp
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        
        return True
    
    def close_position(self, price, timestamp, reason='manual'):
        """Close the current position"""
        if not self.position:
            return None
            
        if self.position == 'long':
            profit = self.position_size * (price - self.entry_price)
        else:  # short
            profit = self.position_size * (self.entry_price - price)
            
        self.current_balance += profit
        self.record_trade(self.position, self.entry_price, price, self.entry_time, timestamp, profit, reason)
        
        result = {
            'position': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'profit': profit,
            'reason': reason
        }
        
        self.position = None
        self.entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_price = None
        
        return result