import time
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from ml_models import MLManager
from .utils.notifications import send_telegram_notification
from .utils.data_fetcher import get_market_data, calculate_indicators
from .utils.trade_manager import (
    execute_trade,
    check_take_profit_stop_loss,
    close_all_positions,
    calculate_position_metrics,
)
from .utils.reporting import save_results

# Load environment variables
load_dotenv()


class RealtimeTrader:
    def __init__(
        self,
        symbol="BTCUSDT",
        initial_investment=50.0,
        daily_profit_target=15.0,
        leverage=15,
        test_mode=False,
    ):
        """
        Initialize the real-time trader

        Args:
            symbol: Trading pair to trade
            initial_investment: Starting capital in USD
            daily_profit_target: Target profit per day in USD
            leverage: Margin trading leverage (15x-20x)
            test_mode: If True, run in test mode with fake balance
        """
        self.symbol = symbol
        self.initial_investment = initial_investment
        self.daily_profit_target = daily_profit_target
        self.test_mode = test_mode

        # Set leverage (constrain between 15x and 20x)
        self.leverage = max(15, min(20, leverage))

        # Initialize Binance client
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = None  # Will be initialized when trading starts

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

        # Position tracking
        self.position = None
        self.entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.entry_time = None
        self.stop_loss_pct = 0.05  # Default 5%
        self.take_profit_pct = 0.1  # Default 10%

        # Test mode variables
        if self.test_mode:
            self.test_balance = initial_investment
            self.test_trades = []
            print(f"Running in TEST MODE with fake balance: ${self.test_balance:.2f}")

        # Results directory
        self.results_dir = os.path.join(os.getcwd(), "realtime_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Telegram notification setup
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.notifications_enabled = bool(
            self.telegram_bot_token and self.telegram_chat_id
        )

        # Trade history
        self.trade_history = []

        print(
            f"Real-time trader initialized for {symbol} with {self.leverage}x leverage"
        )

    def initialize_trading_client(self):
        """Initialize the Binance client for real trading"""
        if not self.api_key or not self.api_secret:
            if not self.test_mode:
                raise ValueError("Binance API key and secret are required for real trading")
            else:
                print("TEST MODE: Using public API access for market data only")
                self.client = Client("", "")  # Public API access for market data only
                return

        self.client = Client(self.api_key, self.api_secret)

        # Set leverage for futures trading (only in real mode)
        if not self.test_mode:
            try:
                self.client.futures_change_leverage(
                    symbol=self.symbol, leverage=self.leverage
                )
                print(f"Leverage set to {self.leverage}x for {self.symbol}")
            except BinanceAPIException as e:
                print(f"Error setting leverage: {e}")

    def get_latest_data(self, lookback_candles=500):
        """Get latest market data"""
        return get_market_data(
            self.client,
            self.symbol,
            self.timeframe,
            lookback_candles,
            self.short_window,
            self.long_window,
            self.atr_period,
        )

    def get_account_balance(self):
        """Get current account balance"""
        if self.test_mode:
            return self.test_balance

        try:
            # For futures trading
            account_info = self.client.futures_account_balance()
            for asset in account_info:
                if asset["asset"] == "USDT":
                    return float(asset["balance"])

            return 0.0
        except BinanceAPIException as e:
            print(f"Error getting account balance: {e}")
            return self.initial_investment  # Fallback

    def get_current_price(self):
        """Get current price of the symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker["price"])
        except BinanceAPIException as e:
            print(f"Error getting current price: {e}")
            # Fallback to getting latest data
            latest_df = self.get_latest_data(lookback_candles=1)
            if latest_df is not None and len(latest_df) > 0:
                return float(latest_df.iloc[-1]["close"])
            return 0.0

    def has_open_position(self):
        """Check if there's an open position"""
        if self.test_mode:
            return self.position is not None

        try:
            # For futures trading
            positions = self.client.futures_position_information(symbol=self.symbol)
            for position in positions:
                if float(position["positionAmt"]) != 0:
                    # Update position tracking
                    self.position = (
                        "long" if float(position["positionAmt"]) > 0 else "short"
                    )
                    self.position_size = abs(float(position["positionAmt"]))
                    self.entry_price = float(position["entryPrice"])
                    return True

            # No open position
            self.position = None
            self.position_size = 0
            self.entry_price = None
            return False
        except BinanceAPIException as e:
            print(f"Error checking open position: {e}")
            return False

    def get_position_info(self):
        """Get information about the current position"""
        if not self.has_open_position():
            return None

        current_price = self.get_current_price()
        position_value = self.position_size * current_price
        profit_loss = 0
        profit_pct = 0

        if self.position == "long":
            profit_loss = self.position_size * (current_price - self.entry_price)
            profit_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # short
            profit_loss = self.position_size * (self.entry_price - current_price)
            profit_pct = (self.entry_price - current_price) / self.entry_price * 100

        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "position_size": self.position_size,
            "position_value": position_value,
            "profit_loss": profit_loss,
            "profit_pct": profit_pct,
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "entry_time": self.entry_time,
        }

    def send_notification(self, message):
        """Send Telegram notification"""
        if self.notifications_enabled:
            send_telegram_notification(
                self.telegram_bot_token, self.telegram_chat_id, message
            )

    def check_take_profit_stop_loss(self, current_price, timestamp):
        """Check for take profit or stop loss with risk management"""
        if not self.has_open_position():
            return None

        result = None

        # Check for take profit
        if self.position == "long" and current_price >= self.take_profit_price:
            result = self.close_position(current_price, timestamp, "TAKE PROFIT")
        elif self.position == "short" and current_price <= self.take_profit_price:
            result = self.close_position(current_price, timestamp, "TAKE PROFIT")

        # Check for stop loss
        elif self.position == "long" and current_price <= self.stop_loss_price:
            result = self.close_position(current_price, timestamp, "STOP LOSS")
        elif self.position == "short" and current_price >= self.stop_loss_price:
            result = self.close_position(current_price, timestamp, "STOP LOSS")

        if result:
            # Update daily loss tracking
            if result["profit"] < 0:
                self.daily_loss += abs(result["profit"])

                # Check if maximum daily loss is reached
                if self.daily_loss >= (self.initial_investment * self.max_daily_loss):
                    self.trading_disabled = True
                    message = (
                        "⚠️ Trading disabled for today\n"
                        f"Reached maximum daily loss: ${self.daily_loss:.2f}\n"
                        f"Current balance: ${self.get_account_balance():.2f}"
                    )
                    self.send_notification(message)

            # Format result message
            position_type = self.position.upper()
            reason = result["reason"]
            profit = result["profit"]

            result_message = (
                f"{position_type} position closed - {reason}\n"
                f"Exit price: ${current_price:.2f}\n"
                f"{'Profit' if profit >= 0 else 'Loss'}: ${abs(profit):.2f}"
            )

            # Save results
            self.save_trading_results()

            return result_message

        return None

    def execute_trade(self, signal, current_price, timestamp):
        """Execute trade with risk management"""
        # Don't trade if trading is disabled due to losses
        current_day = datetime.now().date()
        if current_day != self.last_reset_day:
            # Reset daily tracking
            self.daily_loss = 0
            self.trading_disabled = False
            self.last_reset_day = current_day

        if self.trading_disabled:
            print("Trading disabled due to reaching maximum daily loss")
            return None

        # Don't trade if there's already an open position
        if self.has_open_position():
            print("Already have an open position, skipping trade")
            return None

        # Don't trade if balance is too low
        account_balance = self.get_account_balance()
        if account_balance < (self.initial_investment * 0.5):
            print(f"Balance too low (${account_balance:.2f}), trading paused")
            return None

        # Calculate maximum position size based on current balance
        max_position_value = account_balance * self.max_position_size

        # Calculate position size based on risk per trade
        risk_amount = account_balance * self.risk_per_trade

        # Get ATR for dynamic stop loss and take profit
        latest_df = self.get_latest_data(lookback_candles=20)
        atr = latest_df["ATR"].iloc[-1]

        if signal == 1:  # BUY signal
            # Use ATR for stop loss (2x ATR)
            self.stop_loss_pct = min(0.05, (2 * atr) / current_price)  # Cap at 5%
            # Use ATR for take profit (3x ATR)
            self.take_profit_pct = min(0.1, (3 * atr) / current_price)  # Cap at 10%

            # Calculate position size based on risk
            position_size = (risk_amount / self.stop_loss_pct) / current_price

            # Calculate stop loss and take profit prices
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
            self.take_profit_price = current_price * (1 + self.take_profit_pct)

        elif signal == -1:  # SELL signal
            # Use ATR for stop loss (2x ATR)
            self.stop_loss_pct = min(0.05, (2 * atr) / current_price)  # Cap at 5%
            # Use ATR for take profit (3x ATR)
            self.take_profit_pct = min(0.1, (3 * atr) / current_price)  # Cap at 10%

            # Calculate position size based on risk
            position_size = (risk_amount / self.stop_loss_pct) / current_price

            # Calculate stop loss and take profit prices
            self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
            self.take_profit_price = current_price * (1 - self.take_profit_pct)

        # Ensure position size doesn't exceed maximum allowed
        position_value = position_size * current_price
        if position_value > max_position_value:
            position_size = max_position_value / current_price

        # Execute the trade
        try:
            if signal == 1:  # BUY signal
                if not self.test_mode:
                    # For futures trading - open long position
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="BUY",
                        type="MARKET",
                        quantity=position_size,
                    )
                else:
                    # Test mode - simulate order
                    print(f"TEST MODE: Simulating BUY order for {position_size:.6f} {self.symbol} at ${current_price:.2f}")

                # Update position tracking
                self.position = "long"
                self.entry_price = current_price
                self.position_size = position_size
                self.entry_time = timestamp

                # Add to trade history
                trade_record = {
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": current_price,
                    "size": position_size,
                    "value": position_size * current_price,
                    "stop_loss": self.stop_loss_price,
                    "take_profit": self.take_profit_price,
                }

                self.trade_history.append(trade_record)
                if self.test_mode:
                    self.test_trades.append(trade_record)

                return f"BUY: Opened long position at ${current_price:.2f} with {position_size:.6f} units"

            elif signal == -1:  # SELL signal
                if not self.test_mode:
                    # For futures trading - open short position
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=position_size,
                    )
                else:
                    # Test mode - simulate order
                    print(f"TEST MODE: Simulating SELL order for {position_size:.6f} {self.symbol} at ${current_price:.2f}")

                # Update position tracking
                self.position = "short"
                self.entry_price = current_price
                self.position_size = position_size
                self.entry_time = timestamp

                # Add to trade history
                trade_record = {
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": current_price,
                    "size": position_size,
                    "value": position_size * current_price,
                    "stop_loss": self.stop_loss_price,
                    "take_profit": self.take_profit_price,
                }

                self.trade_history.append(trade_record)
                if self.test_mode:
                    self.test_trades.append(trade_record)

                return f"SELL: Opened short position at ${current_price:.2f} with {position_size:.6f} units"

        except BinanceAPIException as e:
            print(f"Error executing trade: {e}")
            return None

    def close_position(self, current_price, timestamp, reason="manual"):
        """Close the current position"""
        if not self.has_open_position():
            return None

        try:
            if not self.test_mode:
                # For futures trading - close position
                side = "SELL" if self.position == "long" else "BUY"
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type="MARKET",
                    quantity=self.position_size,
                )
            else:
                # Test mode - simulate order
                side = "SELL" if self.position == "long" else "BUY"
                print(f"TEST MODE: Simulating {side} order to close position for {self.position_size:.6f} {self.symbol} at ${current_price:.2f}")

            # Calculate profit/loss
            if self.position == "long":
                profit = self.position_size * (current_price - self.entry_price)
            else:  # short
                profit = self.position_size * (self.entry_price - current_price)

            # Update test balance in test mode
            if self.test_mode:
                self.test_balance += profit
                print(f"TEST MODE: Balance updated to ${self.test_balance:.2f} (Profit/Loss: ${profit:.2f})")

            # Add to trade history
            close_record = {
                "timestamp": timestamp,
                "action": "CLOSE",
                "price": current_price,
                "size": self.position_size,
                "value": self.position_size * current_price,
                "profit": profit,
                "reason": reason,
            }

            self.trade_history.append(close_record)
            if self.test_mode:
                self.test_trades.append(close_record)

            # Reset position tracking
            position_type = self.position
            self.position = None
            self.entry_price = None
            self.position_size = 0
            self.stop_loss_price = None
            self.take_profit_price = None

            return {
                "position": position_type,
                "exit_price": current_price,
                "profit": profit,
                "reason": reason,
            }

        except BinanceAPIException as e:
            print(f"Error closing position: {e}")
            return None

    def save_trading_results(self):
        """Save real-time trading results to files"""
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Save trade history
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            mode_prefix = "test_" if self.test_mode else ""
            df.to_csv(
                os.path.join(self.results_dir, f"{mode_prefix}{self.symbol}_trade_history.csv"),
                index=False,
            )

        # Save current status
        status = {
            "symbol": self.symbol,
            "balance": self.get_account_balance(),
            "initial_investment": self.initial_investment,
            "profit_loss": self.get_account_balance() - self.initial_investment,
            "return_pct": (
                (self.get_account_balance() - self.initial_investment)
                / self.initial_investment
            )
            * 100,
            "position": self.position,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_mode": self.test_mode
        }

        status_df = pd.DataFrame([status])
        mode_prefix = "test_" if self.test_mode else ""
        status_df.to_csv(
            os.path.join(self.results_dir, f"{mode_prefix}{self.symbol}_status.csv"), index=False
        )

    def run_real_trading(self, duration_hours=24, update_interval_minutes=15):
        """
        Run real-time trading

        Args:
            duration_hours: How long to run the trading in hours
            update_interval_minutes: How often to update in minutes
        """
        return run_real_trading(self, duration_hours, update_interval_minutes)


def run_real_trading(realtime_trader, duration_hours=24, update_interval_minutes=15):
    """
    Run real-time trading

    Args:
        realtime_trader: The RealtimeTrader instance
        duration_hours: How long to run the trading in hours
        update_interval_minutes: How often to update in minutes
    """
    print(
        f"Starting real-time trading for {realtime_trader.symbol} for {duration_hours} hours"
    )
    print(f"Update interval: {update_interval_minutes} minutes")

    # Initialize Binance client for real trading
    realtime_trader.initialize_trading_client()

    # Calculate start and end times
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    # Send notification
    realtime_trader.send_notification(
        f"🚀 REAL TRADING STARTED\n"
        f"Symbol: {realtime_trader.symbol}\n"
        f"Initial Investment: ${realtime_trader.initial_investment:.2f}\n"
        f"Leverage: {realtime_trader.leverage}x\n"
        f"Duration: {duration_hours} hours\n"
        f"Update Interval: {update_interval_minutes} minutes\n"
        f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Main trading loop
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            print(f"\n=== Update: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            # Get latest data
            latest_df = realtime_trader.get_latest_data(
                lookback_candles=2
            )  # Just get the latest candles

            if latest_df is None or len(latest_df) < 1:
                print("Error fetching latest data, will retry next interval")
                time.sleep(60)  # Wait a minute before retrying
                continue

            # Get the latest candle
            latest_candle = latest_df.iloc[-1]

            # Get current price
            current_price = float(latest_candle["close"])
            print(f"Current {realtime_trader.symbol} price: ${current_price:.2f}")

            # Check for take profit / stop loss for open positions
            if realtime_trader.has_open_position():
                tp_sl_result = realtime_trader.check_take_profit_stop_loss(
                    current_price, latest_candle["timestamp"]
                )

                if tp_sl_result:
                    print(tp_sl_result)

                    # Send position closed notification
                    if "TAKE PROFIT" in tp_sl_result or "STOP LOSS" in tp_sl_result:
                        # Extract position type and profit/loss from result
                        position_type = "LONG" if "LONG" in tp_sl_result else "SHORT"
                        reason = (
                            "TAKE PROFIT"
                            if "TAKE PROFIT" in tp_sl_result
                            else "STOP LOSS"
                        )

                        # Extract profit amount from the result string
                        profit_match = re.search(
                            r"(Profit|Loss): \$([0-9.-]+)", tp_sl_result
                        )
                        profit_amount = (
                            float(profit_match.group(2)) if profit_match else 0
                        )

                        # Create emoji based on profit/loss
                        emoji = "💰" if profit_amount > 0 else "🛑"

                        # Send notification for position closed
                        close_message = (
                            f"{emoji} POSITION CLOSED ({position_type})\n"
                            f"Symbol: {realtime_trader.symbol}\n"
                            f"Reason: {reason}\n"
                            f"Exit Price: ${current_price:,.2f}\n"
                            f"{'Profit' if profit_amount > 0 else 'Loss'}: ${abs(profit_amount):,.2f}\n"
                            f"Balance: ${realtime_trader.get_account_balance():,.2f}"
                        )
                        realtime_trader.send_notification(close_message)
            else:
                # Get trading signals
                traditional_signal = latest_candle["signal"]

                # Get ML signal with error handling
                ml_signal = 0
                ml_confidence = 0
                try:
                    if realtime_trader.ml_manager:
                        ml_signal, ml_confidence = (
                            realtime_trader.ml_manager.get_ml_signal(
                                realtime_trader.symbol, latest_df
                            )
                        )
                except Exception as e:
                    print(f"Error getting ML signal: {e}")
                    ml_signal = 0
                    ml_confidence = 0

                # Combine signals
                signal = 0
                if traditional_signal == ml_signal:
                    signal = traditional_signal
                elif ml_confidence > 0.75:
                    signal = ml_signal
                else:
                    signal = traditional_signal  # Fallback to traditional signal

                # Execute trade if there's a signal
                if signal != 0 and not realtime_trader.trading_disabled:
                    # Execute the trade with risk management
                    trade_result = realtime_trader.execute_trade(
                        signal, current_price, latest_candle["timestamp"]
                    )

                    if trade_result:
                        print(trade_result)

                        # Calculate stop loss and take profit levels for notification
                        if signal == 1:  # BUY signal
                            stop_loss_price = current_price * (
                                1 - realtime_trader.stop_loss_pct
                            )
                            take_profit_price = current_price * (
                                1 + realtime_trader.take_profit_pct
                            )
                        else:  # SELL signal
                            stop_loss_price = current_price * (
                                1 + realtime_trader.stop_loss_pct
                            )
                            take_profit_price = current_price * (
                                1 - realtime_trader.take_profit_pct
                            )

                        # Send notification for position opened
                        if "BUY" in trade_result or "SELL" in trade_result:
                            position_type = "LONG" if "BUY" in trade_result else "SHORT"
                            emoji = "🟢" if position_type == "LONG" else "🔴"

                            # Extract position size from trade result
                            size_match = re.search(r"([0-9.]+) units", trade_result)
                            position_size = (
                                float(size_match.group(1)) if size_match else 0
                            )

                            open_message = (
                                f"{emoji} POSITION OPENED ({position_type})\n"
                                f"Symbol: {realtime_trader.symbol}\n"
                                f"Entry Price: ${current_price:,.2f}\n"
                                f"Position Size: {position_size:,.6f} units\n"
                                f"Stop Loss: ${stop_loss_price:,.2f}\n"
                                f"Take Profit: ${take_profit_price:,.2f}\n"
                                f"Balance: ${realtime_trader.get_account_balance():,.2f}"
                            )
                            realtime_trader.send_notification(open_message)

            # Print current status
            account_balance = realtime_trader.get_account_balance()
            print(f"Current balance: ${account_balance:.2f}")

            if realtime_trader.has_open_position():
                position_info = realtime_trader.get_position_info()

                if position_info:
                    print(f"Current position: {position_info['position'].upper()}")
                    print(f"Entry price: ${position_info['entry_price']:.2f}")
                    print(f"Position size: {position_info['position_size']:.6f} units")
                    print(f"Position value: ${position_info['position_value']:.2f}")
                    print(f"Unrealized P/L: ${position_info['profit_loss']:.2f}")

                    # Send position update notification on every update
                    # Calculate profit percentage
                    profit_pct = position_info["profit_pct"]
                    profit_loss = position_info["profit_loss"]

                    # Only send update if significant change (>1% profit change)
                    if abs(profit_pct) > 1:
                        emoji = "📈" if profit_pct > 0 else "📉"
                        update_message = (
                            f"{emoji} POSITION UPDATE ({position_info['position'].upper()})\n"
                            f"Symbol: {realtime_trader.symbol}\n"
                            f"Current Price: ${current_price:,.2f}\n"
                            f"Entry Price: ${position_info['entry_price']:,.2f}\n"
                            f"Unrealized P/L: ${profit_loss:,.2f} ({profit_pct:.2f}%)\n"
                            f"Stop Loss: ${position_info['stop_loss']:,.2f}\n"
                            f"Take Profit: ${position_info['take_profit']:,.2f}"
                        )
                        realtime_trader.send_notification(update_message)

            # Save results
            realtime_trader.save_trading_results()

            # Wait for next update with countdown
            sleep_seconds = update_interval_minutes * 60
            print(f"Next update in {update_interval_minutes} minutes...")
            print(f"Next update in {sleep_seconds} seconds")
            print("Countdown started...")

            # Display countdown timer
            start_wait = time.time()
            total_wait = sleep_seconds

            while time.time() - start_wait < sleep_seconds:
                try:
                    # Calculate elapsed and remaining time
                    elapsed = time.time() - start_wait
                    remaining = sleep_seconds - elapsed
                    mins, secs = divmod(int(remaining), 60)

                    # Calculate progress percentage
                    progress_pct = elapsed / total_wait

                    # Create progress bar (width between 15-20 characters)
                    bar_width = 20
                    filled_width = int(bar_width * progress_pct)
                    bar = "█" * filled_width + "░" * (bar_width - filled_width)

                    # Create countdown display with margin
                    countdown = f"⏱️ Next update in: {mins:02d}:{secs:02d} [{bar}] {progress_pct:.0%}"
                    print(countdown, end="\r", flush=True)
                    time.sleep(1)

                except KeyboardInterrupt:
                    print("\nTrading interrupted by user")
                    break

            print(
                "\nUpdate time reached!                                      "
            )  # Clear line with newline
            print(" " * 50, end="\r")  # Clear the line with extra space

        except KeyboardInterrupt:
            print("\nTrading interrupted by user")
            break

        except Exception as e:
            print(f"Error in trading loop: {e}")
            realtime_trader.send_notification(f"⚠️ ERROR: {e}")
            # Wait a bit before retrying
            time.sleep(60)

    # Trading completed
    print("\n=== Trading Completed ===")

    # Get final account balance
    final_balance = realtime_trader.get_account_balance()
    profit_loss = final_balance - realtime_trader.initial_investment
    return_pct = (profit_loss / realtime_trader.initial_investment) * 100

    print(f"Final balance: ${final_balance:.2f}")
    print(f"Profit/Loss: ${profit_loss:.2f}")
    print(f"Return: {return_pct:.2f}%")

    # Close any open positions
    if realtime_trader.has_open_position():
        print("Closing open position...")
        current_price = realtime_trader.get_current_price()
        close_result = realtime_trader.close_position(
            current_price, datetime.now(), "end_of_session"
        )

        if close_result:
            print(f"Position closed: {close_result}")

    # Send final notification
    realtime_trader.send_notification(
        f"🏁 TRADING SESSION COMPLETED\n"
        f"Symbol: {realtime_trader.symbol}\n"
        f"Duration: {duration_hours} hours\n"
        f"Final Balance: ${final_balance:.2f}\n"
        f"Profit/Loss: ${profit_loss:.2f} ({return_pct:.2f}%)"
    )

    return {
        "final_balance": final_balance,
        "profit_loss": profit_loss,
        "return_pct": return_pct,
    }
