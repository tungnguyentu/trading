import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException
import telegram
import asyncio
from dotenv import load_dotenv
from ml_models import MLManager
from simulation import TradingSimulator

# Load environment variables
load_dotenv()

class RealtimeSimulator:
    def __init__(self, symbol='BTCUSDT', initial_investment=50.0, daily_profit_target=15.0, leverage=15):
        """
        Initialize the real-time simulator
        
        Args:
            symbol: Trading pair to simulate
            initial_investment: Starting capital in USD
            daily_profit_target: Target profit per day in USD
            leverage: Margin trading leverage (15x-20x)
        """
        self.symbol = symbol
        self.initial_investment = initial_investment
        self.daily_profit_target = daily_profit_target
        
        # Set leverage (constrain between 15x and 20x)
        self.leverage = max(15, min(20, leverage))
        
        # Initialize Binance client
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        
        # Timeframe
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE
        
        # Initialize ML Manager
        self.ml_manager = MLManager()
        
        # Trading parameters
        self.short_window = 20
        self.long_window = 50
        self.atr_period = 14
        
        # Risk management parameters
        self.max_position_size = 0.5  # Maximum 50% of balance for any single trade
        self.max_daily_loss = 0.1  # Maximum 10% daily loss of initial investment
        self.risk_per_trade = 0.02  # Risk 2% of balance per trade
        self.daily_loss = 0
        self.trading_disabled = False
        self.last_reset_day = datetime.now().date()
        
        # Results directory
        self.results_dir = os.path.join(os.getcwd(), "realtime_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Telegram notification setup
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.notifications_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        
        # Initialize simulator
        self.simulator = None
        
        print(f"Real-time simulator initialized for {symbol} with {self.leverage}x leverage")
        
    def get_latest_data(self, lookback_candles=500):
        """Get latest market data"""
        try:
            print(f"Fetching latest data for {self.symbol}")
            
            # Ensure we have enough data for ML training
            min_candles = max(500, lookback_candles)
            
            # Calculate the start time based on number of candles
            # For 15-minute candles, we need to go back lookback_candles * 15 minutes
            start_time = int((datetime.now() - timedelta(minutes=min_candles * 15)).timestamp() * 1000)
            
            klines = self.client.get_historical_klines(
                self.symbol,
                self.timeframe,
                start_str=str(start_time)
            )
            
            print(f"Retrieved {len(klines)} candles from Binance")
            
            if len(klines) < 100:
                print("Warning: Not enough data points retrieved")
                
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume', 'trades',
                'buy_base_volume', 'buy_quote_volume', 'ignore'
            ])
            
            # Convert numeric columns
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['open'] = pd.to_numeric(df['open'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add symbol column
            df['symbol'] = self.symbol
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except BinanceAPIException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for trading signals"""
        # Calculate SMAs
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=self.atr_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1
        
        return df
    
    def send_notification(self, message):
        """Send Telegram notification"""
        if not self.notifications_enabled:
            return
            
        try:
            # Create a new event loop for each notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Define the async function
            async def send_message():
                bot = telegram.Bot(token=self.telegram_bot_token)
                await bot.send_message(chat_id=self.telegram_chat_id, text=message)
            
            # Run the coroutine and close the loop properly
            loop.run_until_complete(send_message())
            loop.close()
            print(f"Telegram notification sent: {message}")
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
    def check_take_profit_stop_loss(self, current_price, timestamp):
        """Check for take profit or stop loss with risk management"""
        import re
        
        result = self.simulator.check_take_profit_stop_loss(current_price, timestamp)
        
        if result and ("STOP LOSS" in result or "TAKE PROFIT" in result):
            # Update daily loss tracking
            loss_match = re.search(r'(Profit|Loss): \$([0-9.-]+)', result)
            if loss_match:
                loss_amount = float(loss_match.group(2))  # Fixed: use group(2) to get the amount
                
                # Update daily loss if it's a loss
                if "Loss" in result:
                    self.daily_loss += loss_amount
                    
                    # Check if maximum daily loss is reached
                    if self.daily_loss >= (self.initial_investment * self.max_daily_loss):
                        self.trading_disabled = True
                        message = (
                            "⚠️ Trading disabled for today\n"
                            f"Reached maximum daily loss: ${self.daily_loss:.2f}\n"
                            f"Current balance: ${self.simulator.current_balance:.2f}"
                        )
                        self.send_notification(message)
            
            # Save results after each trade
            self.save_realtime_results()
        
        return result
        
    def save_realtime_results(self):
        """Save real-time results to files"""
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save trades to CSV
            trades_file = os.path.join(self.results_dir, f"{self.symbol}_trades_{timestamp}.csv")
            trades_df = pd.DataFrame(self.simulator.trades)
            trades_df.to_csv(trades_file, index=False)
            
            # Save equity curve to CSV
            equity_file = os.path.join(self.results_dir, f"{self.symbol}_equity_{timestamp}.csv")
            equity_df = pd.DataFrame(self.simulator.equity_curve)
            equity_df.to_csv(equity_file, index=False)
            
            # Save daily profits to CSV
            daily_file = os.path.join(self.results_dir, f"{self.symbol}_daily_{timestamp}.csv")
            daily_df = pd.DataFrame(list(self.simulator.daily_profits.items()), 
                                   columns=['date', 'profit'])
            daily_df.to_csv(daily_file, index=False)
            
            # Generate performance summary
            summary_file = os.path.join(self.results_dir, f"{self.symbol}_summary_{timestamp}.txt")
            with open(summary_file, 'w') as f:
                f.write(f"=== REAL-TIME SIMULATION SUMMARY ===\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Initial Investment: ${self.initial_investment:.2f}\n")
                f.write(f"Current Balance: ${self.simulator.current_balance:.2f}\n")
                f.write(f"Profit/Loss: ${self.simulator.current_balance - self.initial_investment:.2f}\n")
                f.write(f"Return: {((self.simulator.current_balance - self.initial_investment) / self.initial_investment) * 100:.2f}%\n")
                f.write(f"Total Trades: {len(self.simulator.trades)}\n")
                
                # Calculate win rate
                if len(self.simulator.trades) > 0:
                    winning_trades = sum(1 for trade in self.simulator.trades if trade['profit'] > 0)
                    win_rate = (winning_trades / len(self.simulator.trades)) * 100
                    f.write(f"Win Rate: {win_rate:.2f}%\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"Real-time results saved to {self.results_dir}")
            
        except Exception as e:
            print(f"Error saving real-time results: {e}")
    def execute_trade(self, signal, current_price, timestamp):
        """Execute trade with risk management"""
        # Check if trading should be disabled due to losses
        current_day = datetime.now().date()
        if current_day != self.last_reset_day:
            # Reset daily tracking
            self.daily_loss = 0
            self.trading_disabled = False
            self.last_reset_day = current_day
            
        if self.trading_disabled:
            print("Trading disabled due to reaching maximum daily loss")
            return None
            
        # Don't trade if balance is too low
        if self.simulator.current_balance < (self.initial_investment * 0.5):
            print(f"Balance too low (${self.simulator.current_balance:.2f}), trading paused")
            return None
            
        # Calculate maximum position size based on current balance
        max_position_value = self.simulator.current_balance * self.max_position_size
        
        # Calculate position size based on risk per trade
        risk_amount = self.simulator.current_balance * self.risk_per_trade
        
        # Get ATR for dynamic stop loss and take profit
        latest_df = self.get_latest_data(lookback_candles=20)
        atr = latest_df['ATR'].iloc[-1]
        
        if signal == 1:  # BUY signal
            # Use ATR for stop loss (2x ATR)
            stop_loss_pct = min(0.05, (2 * atr) / current_price)  # Cap at 5%
            # Use ATR for take profit (3x ATR)
            take_profit_pct = min(0.1, (3 * atr) / current_price)  # Cap at 10%
            
            # Calculate position size based on risk
            position_size = (risk_amount / stop_loss_pct) / current_price
            
        elif signal == -1:  # SELL signal
            # Use ATR for stop loss (2x ATR)
            stop_loss_pct = min(0.05, (2 * atr) / current_price)  # Cap at 5%
            # Use ATR for take profit (3x ATR)
            take_profit_pct = min(0.1, (3 * atr) / current_price)  # Cap at 10%
            
            # Calculate position size based on risk
            position_size = (risk_amount / stop_loss_pct) / current_price
        
        # Ensure position size doesn't exceed maximum allowed
        position_value = position_size * current_price
        if position_value > max_position_value:
            position_size = max_position_value / current_price
        
        # Update simulator parameters
        self.simulator.stop_loss_pct = stop_loss_pct
        self.simulator.take_profit_pct = take_profit_pct
        self.simulator.position_size = position_size
        
        # Execute the trade
        return self.simulator.execute_trade(signal, current_price, timestamp)
        
    def run_realtime_simulation(self, duration_hours=24, update_interval_minutes=15):
        """
        Run real-time simulation
        
        Args:
            duration_hours: How long to run the simulation in hours
            update_interval_minutes: How often to update in minutes
        """
        # Get current date and time for investment
        current_datetime = datetime.now()
        investment_date = current_datetime.strftime('%Y-%m-%d')
        
        print(f"Starting real-time simulation for {self.symbol}")
        print(f"Investment date: {investment_date}")
        print(f"Initial investment: ${self.initial_investment:.2f}")
        print(f"Daily profit target: ${self.daily_profit_target:.2f}")
        print(f"Duration: {duration_hours} hours")
        print(f"Update interval: {update_interval_minutes} minutes")
        
        # Get initial data
        df = self.get_latest_data()
        if df is None or len(df) < 100:
            print("Error: Not enough historical data to start simulation")
            return
        
        # Initialize simulator with historical data
        self.simulator = TradingSimulator(
            historical_data=df,
            initial_investment=self.initial_investment,
            daily_profit_target=self.daily_profit_target,
            leverage=self.leverage
        )
        
        # Send notification that simulation is starting
        start_message = (
            f"🚀 REAL-TIME SIMULATION STARTED\n"
            f"Symbol: {self.symbol}\n"
            f"Investment Date: {investment_date}\n"
            f"Initial Investment: ${self.initial_investment:.2f}\n"
            f"Leverage: {self.leverage}x\n"
            f"Daily Profit Target: ${self.daily_profit_target:.2f}\n"
            f"Duration: {duration_hours} hours\n"
            f"Start Time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_notification(start_message)
        
        # Run historical simulation only for training the ML model
        # But don't actually execute trades from historical data
        try:
            print("Running historical simulation for ML training only...")
            # Save original notification setting and disable during training
            original_notifications = self.simulator.notifications_enabled
            self.simulator.notifications_enabled = False
            
            # Check if we have enough data for ML training
            if len(df) > 200:  # Make sure we have enough data points
                # Run simulation and collect trade data for ML training
                historical_balance, historical_trades, _ = self.simulator.run_simulation(self.ml_manager)
                
                # Train ML model with the trade data if we have trades
                if len(historical_trades) > 10:  # Ensure we have enough trades for training
                    print(f"Training ML model with {len(historical_trades)} historical trades")
                    self.ml_manager.train_with_trade_data(historical_trades, df)
                else:
                    print(f"Not enough historical trades ({len(historical_trades)}) for ML training")
                    self.ml_manager = None  # Disable ML manager
            else:
                print("Not enough historical data for ML training, using traditional signals only")
                self.ml_manager = None  # Disable ML manager
            
            # Reset simulator state to start fresh with real-time trading
            self.simulator.current_balance = self.initial_investment
            self.simulator.position = None
            self.simulator.entry_price = None
            self.simulator.position_size = 0
            self.simulator.trades = []
            self.simulator.daily_profits = {}
            self.simulator.equity_curve = [{
                'timestamp': datetime.now(),
                'balance': self.initial_investment
            }]
            
            # Restore notification setting
            self.simulator.notifications_enabled = original_notifications
            
            print(f"ML training complete. Starting fresh with ${self.initial_investment:.2f}")
        except Exception as e:
            print(f"Error during ML training: {e}")
            print("Falling back to traditional signals only")
            self.ml_manager = None  # Disable ML manager if there's an error
            print(f"Starting fresh with ${self.initial_investment:.2f}")
        
        # Calculate end time
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Main real-time loop
        while datetime.now() < end_time:
            try:
                current_time = datetime.now()
                print(f"\n=== Update at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                
                # Get latest data
                latest_df = self.get_latest_data(lookback_candles=2)  # Just get the latest candles
                
                if latest_df is None or len(latest_df) < 1:
                    print("Error fetching latest data, will retry next interval")
                    time.sleep(60)  # Wait a minute before retrying
                    continue
                
                # Get the latest candle
                latest_candle = latest_df.iloc[-1]
                
                # Get current price
                current_price = float(latest_candle['close'])
                print(f"Current {self.symbol} price: ${current_price:.2f}")
                
                # Check for take profit / stop loss
                tp_sl_result = self.simulator.check_take_profit_stop_loss(
                    current_price, latest_candle['timestamp'])
                
                if tp_sl_result:
                    print(tp_sl_result)
                    
                    # Send position closed notification
                    if "TAKE PROFIT" in tp_sl_result or "STOP LOSS" in tp_sl_result:
                        # Extract position type and profit/loss from result
                        position_type = "LONG" if "LONG" in tp_sl_result else "SHORT"
                        reason = "TAKE PROFIT" if "TAKE PROFIT" in tp_sl_result else "STOP LOSS"
                        
                        # Extract profit amount from the result string
                        import re
                        profit_match = re.search(r'(Profit|Loss): \$([0-9.-]+)', tp_sl_result)
                        profit_amount = float(profit_match.group(2)) if profit_match else 0
                        
                        # Create emoji based on profit/loss
                        emoji = "💰" if profit_amount > 0 else "🛑"
                        
                        # Send notification for position closed
                        close_message = (
                            f"{emoji} POSITION CLOSED ({position_type})\n"
                            f"Symbol: {self.symbol}\n"
                            f"Reason: {reason}\n"
                            f"Exit Price: ${current_price:,.2f}\n"
                            f"{'Profit' if profit_amount > 0 else 'Loss'}: ${abs(profit_amount):,.2f}\n"
                            f"Balance: ${self.simulator.current_balance:,.2f}"
                        )
                        self.send_notification(close_message)
                else:
                    # Get trading signals
                    traditional_signal = latest_candle['signal']
                    
                    # Get ML signal with error handling
                    ml_signal = 0
                    ml_confidence = 0
                    try:
                        if self.ml_manager:
                            ml_signal, ml_confidence = self.ml_manager.get_ml_signal(
                                self.symbol, latest_df)
                    except Exception as e:
                        print(f"Error getting ML signal: {e}")
                        ml_signal = 0
                        ml_confidence = 0
                    
                    # Combine signals
                    signal = 0
                    if ml_signal != 0 and traditional_signal == ml_signal:
                        signal = traditional_signal
                    elif ml_signal != 0 and ml_confidence > 0.75:
                        signal = ml_signal
                    else:
                        signal = traditional_signal  # Fallback to traditional signal
                    if traditional_signal == ml_signal:
                        signal = traditional_signal
                    elif ml_confidence > 0.75:
                        signal = ml_signal
                    
                    # Execute trade if there's a signal
                    if signal != 0:
                        # Execute the trade with risk management
                        trade_result = self.execute_trade(signal, current_price, latest_candle['timestamp'])
                        
                        if trade_result:
                            print(trade_result)
                            
                            # Calculate stop loss and take profit levels for notification
                            if signal == 1:  # BUY signal
                                stop_loss_price = current_price * (1 - self.simulator.stop_loss_pct)
                                take_profit_price = current_price * (1 + self.simulator.take_profit_pct)
                            else:  # SELL signal
                                stop_loss_price = current_price * (1 + self.simulator.stop_loss_pct)
                                take_profit_price = current_price * (1 - self.simulator.take_profit_pct)
                            
                            # Send notification for position opened
                            if "BUY" in trade_result or "SELL" in trade_result:
                                position_type = "LONG" if "BUY" in trade_result else "SHORT"
                                emoji = "🟢" if position_type == "LONG" else "🔴"
                                
                                # Extract position size from trade result
                                import re
                                size_match = re.search(r'([0-9.]+) units', trade_result)
                                position_size = float(size_match.group(1)) if size_match else 0
                                
                                open_message = (
                                    f"{emoji} POSITION OPENED ({position_type})\n"
                                    f"Symbol: {self.symbol}\n"
                                    f"Entry Price: ${current_price:,.2f}\n"
                                    f"Position Size: {position_size:,.6f} units\n"
                                    f"Stop Loss: ${stop_loss_price:,.2f}\n"
                                    f"Take Profit: ${take_profit_price:,.2f}\n"
                                    f"Balance: ${self.simulator.current_balance:,.2f}"
                                )
                                self.send_notification(open_message)
                
                # Print current status
                print(f"Current balance: ${self.simulator.current_balance:.2f}")
                if self.simulator.position:
                    position_value = self.simulator.position_size * current_price
                    profit_loss = 0
                    if self.simulator.position == 'long':
                        profit_loss = self.simulator.position_size * (current_price - self.simulator.entry_price)
                    else:  # short
                        profit_loss = self.simulator.position_size * (self.simulator.entry_price - current_price)
                    
                    print(f"Current position: {self.simulator.position.upper()}")
                    print(f"Entry price: ${self.simulator.entry_price:.2f}")
                    print(f"Position size: {self.simulator.position_size:.6f} units")
                    print(f"Position value: ${position_value:.2f}")
                    print(f"Unrealized P/L: ${profit_loss:.2f}")
                    
                    # Send position update notification on every update
                    # This respects the update_interval_minutes parameter
                    profit_pct = (profit_loss / (self.simulator.position_size * self.simulator.entry_price)) * 100
                    emoji = "📈" if profit_loss > 0 else "📉"
                    
                    update_message = (
                        f"{emoji} POSITION UPDATE ({self.simulator.position.upper()})\n"
                        f"Symbol: {self.symbol}\n"
                        f"Current Price: ${current_price:,.2f}\n"
                        f"Entry Price: ${self.simulator.entry_price:,.2f}\n"
                        f"Unrealized P/L: ${profit_loss:,.2f} ({profit_pct:.2f}%)\n"
                        f"Position Duration: {int((current_time - self.simulator.entry_time).total_seconds() / 60)} minutes"
                    )
                    self.send_notification(update_message)
                else:
                    print("No open position")
                
                # Calculate time remaining
                time_remaining = end_time - current_time
                hours, remainder = divmod(time_remaining.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Time remaining: {hours}h {minutes}m {seconds}s")
                
                # Wait for next update
                next_update = current_time + timedelta(minutes=update_interval_minutes)
                sleep_seconds = (next_update - datetime.now()).total_seconds()
                
                if sleep_seconds > 0:
                    print(f"Next update in {sleep_seconds:.0f} seconds")
                    print("Countdown started...")
                    
                    # Real-time countdown with progress bar
                    start_wait = time.time()
                    total_wait = sleep_seconds
                    
                    while time.time() - start_wait < sleep_seconds:
                        elapsed = time.time() - start_wait
                        remaining = sleep_seconds - elapsed
                        mins, secs = divmod(int(remaining), 60)
                        
                        # Calculate progress percentage
                        progress_pct = elapsed / total_wait
                        
                        # Create progress bar (width between 15-20 characters)
                        bar_width = 20
                        filled_width = int(bar_width * progress_pct)
                        bar = '█' * filled_width + '░' * (bar_width - filled_width)
                        
                        # Create countdown display with margin
                        countdown = f"⏱️ Next update in: {mins:02d}:{secs:02d} [{bar}] {progress_pct:.0%}"
                        print(countdown, end='\r', flush=True)
                        time.sleep(1)
                    
                    print("\nUpdate time reached!                                      ")  # Clear line with newline
                    print(" " * 50, end='\r')  # Clear the line with extra space
                    # Removed redundant time.sleep(sleep_seconds) as the while loop already waited
                
            except Exception as e:
                print(f"Error in real-time loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        # Simulation ended
        print("\n=== REAL-TIME SIMULATION COMPLETED ===")
        
        # Close any open positions
        if self.simulator.position:
            # Get final price
            latest_df = self.get_latest_data(lookback_candles=1)
            if latest_df is not None and len(latest_df) > 0:
                final_price = float(latest_df.iloc[-1]['close'])
                
                if self.simulator.position == 'long':
                    profit = self.simulator.position_size * (final_price - self.simulator.entry_price)
                    self.simulator.current_balance += profit
                    self.simulator.record_trade(
                        'long', self.simulator.entry_price, final_price,
                        self.simulator.entry_time, latest_df.iloc[-1]['timestamp'],
                        profit, 'simulation_end'
                    )
                else:  # short
                    profit = self.simulator.position_size * (self.simulator.entry_price - final_price)
                    self.simulator.current_balance += profit
                    self.simulator.record_trade(
                        'short', self.simulator.entry_price, final_price,
                        self.simulator.entry_time, latest_df.iloc[-1]['timestamp'],
                        profit, 'simulation_end'
                    )
                
                print(f"Closed {self.simulator.position} position at end of simulation: ${profit:.2f}")
                
                # Send notification
                end_message = (
                    f"🏁 REAL-TIME SIMULATION COMPLETED\n"
                    f"Symbol: {self.symbol}\n"
                    f"Leverage: {self.leverage}x\n"
                    f"Final Balance: ${self.simulator.current_balance:.2f}\n"
                    f"Total Profit/Loss: ${self.simulator.current_balance - self.initial_investment:.2f}\n"
                    f"Return: {((self.simulator.current_balance - self.initial_investment) / self.initial_investment) * 100:.2f}%\n"
                    f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.send_notification(end_message)
                
                self.simulator.position = None
                self.simulator.entry_price = None
                self.simulator.position_size = 0
        
        # Generate final reports
        self.simulator.generate_reports()
        
        # Save final real-time results
        self.save_realtime_results()
        
        # Return final results
        return self.simulator.current_balance, self.simulator.trades, self.simulator.daily_profits

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Trading Simulator')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair to simulate')
    parser.add_argument('--investment', type=float, default=50.0, help='Initial investment amount')
    parser.add_argument('--target', type=float, default=15.0, help='Daily profit target')
    parser.add_argument('--hours', type=int, default=24, help='Duration in hours')
    parser.add_argument('--interval', type=int, default=15, help='Update interval in minutes')
    parser.add_argument('--leverage', type=int, default=15, help='Margin trading leverage (15-20x)')
    
    args = parser.parse_args()
    
    simulator = RealtimeSimulator(
        symbol=args.symbol,
        initial_investment=args.investment,
        daily_profit_target=args.target,
        leverage=args.leverage
    )
    
    simulator.run_realtime_simulation(
        duration_hours=args.hours,
        update_interval_minutes=args.interval
    )