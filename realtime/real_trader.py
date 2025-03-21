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
from .utils.data_fetcher import get_market_data
import math

# Load environment variables
load_dotenv()


class RealtimeTrader:
    def __init__(
        self,
        symbol="BTCUSDT",
        initial_investment=50.0,
        daily_profit_target=15.0,
        leverage=5,
        test_mode=False,
        use_full_investment=False,
        use_full_margin=False,
        compound_interest=False,
        enable_pyramiding=False,
        max_pyramid_entries=2,
        pyramid_threshold_pct=0.01,
        use_dynamic_take_profit=False,
        trend_following_mode=False,
        use_enhanced_signals=False,
        signal_confirmation_threshold=2,
        signal_cooldown_minutes=15,
        use_scalping_mode=False,
        scalping_tp_factor=0.5,
        scalping_sl_factor=0.8,
        use_ml_signals=False,
        ml_confidence=0.6,
        train_ml=False,
        retrain_interval=0,
        reassess_positions=False,
        fixed_tp=0,
        fixed_sl=0,
    ):
        """
        Initialize the RealtimeTrader with configuration parameters

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            initial_investment: Initial investment amount in USD
            daily_profit_target: Daily profit target in USD
            leverage: Leverage for margin trading (1-50x)
            test_mode: If True, run in test mode with fake balance
            use_full_investment: If True, use full investment amount for each trade
            use_full_margin: If True, use full investment as margin (high risk)
            compound_interest: If True, reinvest profits to increase position sizes
            enable_pyramiding: If True, allow adding to winning positions
            max_pyramid_entries: Maximum number of pyramid entries allowed
            pyramid_threshold_pct: Profit percentage required before pyramiding
            use_dynamic_take_profit: If True, adjust take profit based on volatility
            trend_following_mode: If True, only trade in direction of market trend
            use_enhanced_signals: If True, use multiple indicators for signal confirmation
            signal_confirmation_threshold: Number of indicators required to confirm a signal
            signal_cooldown_minutes: Minimum time between signals in minutes
            use_scalping_mode: If True, use scalping mode for trading
            scalping_tp_factor: Factor to adjust take profit for scalping mode
            scalping_sl_factor: Factor to adjust stop loss for scalping mode
            use_ml_signals: If True, use machine learning signals
            ml_confidence: Minimum confidence threshold for ML signals
            train_ml: If True, train the machine learning model
            retrain_interval: Interval in days for retraining the model
            reassess_positions: If True, re-evaluate positions based on latest signals
            fixed_tp: Fixed take profit percentage
            fixed_sl: Fixed stop loss percentage
        """
        # Trading parameters
        self.symbol = symbol
        self.initial_investment = initial_investment
        self.daily_profit_target = daily_profit_target
        self.leverage = leverage
        self.test_mode = test_mode
        self.use_full_investment = use_full_investment
        self.use_full_margin = use_full_margin
        self.compound_interest = compound_interest

        # Pyramiding settings
        self.enable_pyramiding = enable_pyramiding
        self.max_pyramid_entries = max_pyramid_entries
        self.pyramid_threshold_pct = pyramid_threshold_pct
        self.pyramid_entries = 0  # Track number of pyramid entries made
        self.pyramid_entry_prices = []  # Track entry prices for pyramid entries
        self.pyramid_position_sizes = []  # Track position sizes for pyramid entries

        # Advanced strategy settings
        self.use_dynamic_take_profit = use_dynamic_take_profit
        self.trend_following_mode = trend_following_mode
        self.trend_strength = 0  # Track trend strength (0-100)
        self.trend_direction = 0  # Track trend direction (1=up, -1=down, 0=neutral)

        # Enhanced signal settings
        self.use_enhanced_signals = use_enhanced_signals
        self.signal_confirmation_threshold = signal_confirmation_threshold
        self.signal_cooldown_minutes = (
            signal_cooldown_minutes  # Minimum time between signals
        )
        self.last_signal_time = None

        # Add per-symbol balance tracking for compound interest
        self.symbol_balance = initial_investment
        self.total_symbol_profit = 0.0

        # Scalping mode settings
        self.use_scalping_mode = use_scalping_mode
        self.scalping_tp_factor = scalping_tp_factor
        self.scalping_sl_factor = scalping_sl_factor

        # ML signal settings
        self.use_ml_signals = use_ml_signals
        self.ml_confidence = ml_confidence
        self.train_ml = train_ml
        self.retrain_interval = retrain_interval
        self.last_train_time = None

        # Position reassessment settings
        self.reassess_positions = reassess_positions

        # Fixed take profit and stop loss settings (in USDT)
        self.fixed_tp = fixed_tp  # Absolute amount in USDT
        self.fixed_sl = fixed_sl  # Absolute amount in USDT

        # Initialize trading client
        self.client = None

        # Set leverage (constrain between 1x and 50x)
        self.leverage = max(1, min(50, leverage))

        # Initialize Binance client
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")

        # Timeframe
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE

        # Initialize ML Manager
        self.ml_manager = MLManager()

        # Trading parameters
        self.short_window = 20
        self.long_window = 50
        self.atr_period = 14
        self.atr_multiplier = 1.5  # Multiplier for ATR-based stop loss

        # Risk management parameters
        self.max_position_size = (
            0.95 if use_full_investment else 0.2
        )  # Use 95% of balance if full investment mode
        self.max_daily_loss = 0.1  # Maximum 10% daily loss of initial investment
        self.risk_pct = 0.02  # Default risk per trade (2% of account balance)
        self.risk_per_trade = (
            0.02  # Risk 2% of balance per trade (used in execute_trade)
        )
        self.use_atr_for_risk = True  # Use ATR for stop loss calculation
        self.daily_loss = 0
        self.trading_disabled = False
        self.last_reset_day = datetime.now().date()

        # Daily profit tracking
        self.daily_profits = {}

        # Position tracking
        self.position = None
        self.entry_price = None
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.entry_time = None
        self.stop_loss_pct = 0.02  # Default 2%
        self.take_profit_pct = 0.04  # Default 4%

        # Partial take profit settings
        self.partial_tp_enabled = True  # Enable partial take profit by default
        self.partial_tp_pct = 0.02  # Take partial profit at 2% gain
        self.partial_tp_size = 0.5  # Close 50% of position at first take profit
        self.partial_tp_executed = (
            False  # Track if partial take profit has been executed
        )
        self.original_position_size = (
            0  # Track original position size before partial TP
        )

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

        # Add order tracking variables
        self.stop_loss_order_id = None
        self.take_profit_order_id = None

        print(
            f"Real-time trader initialized for {symbol} with {self.leverage}x leverage"
        )

    def initialize_trading_client(self):
        """Initialize the Binance client for real trading"""
        if not self.api_key or not self.api_secret:
            if not self.test_mode:
                raise ValueError(
                    "Binance API key and secret are required for real trading"
                )
            else:
                print("TEST MODE: Using public API access for market data only")
                self.client = Client("", "")  # Public API access for market data only
                return

        self.client = Client(self.api_key, self.api_secret, testnet=True)

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

                    # Ensure stop_loss_price and take_profit_price are initialized if they're None
                    if self.stop_loss_price is None:
                        # Calculate default stop loss based on position type and a conservative 1% risk
                        if self.position == "long":
                            self.stop_loss_price = (
                                self.entry_price * 0.99
                            )  # 1% below entry for long
                        else:
                            self.stop_loss_price = (
                                self.entry_price * 1.01
                            )  # 1% above entry for short
                        print(
                            f"WARNING: stop_loss_price was None, initialized to ${self.stop_loss_price:.2f}"
                        )

                    if self.take_profit_price is None:
                        # Calculate default take profit based on position type and a 2% target
                        if self.position == "long":
                            self.take_profit_price = (
                                self.entry_price * 1.02
                            )  # 2% above entry for long
                        else:
                            self.take_profit_price = (
                                self.entry_price * 0.98
                            )  # 2% below entry for short
                        print(
                            f"WARNING: take_profit_price was None, initialized to ${self.take_profit_price:.2f}"
                        )

                    return True

            # No open position
            self.position = None
            self.position_size = 0  # Set to 0 instead of None
            self.entry_price = None
            self.stop_loss_price = None  # Ensure stop_loss_price is reset
            self.take_profit_price = None  # Ensure take_profit_price is reset
            return False
        except BinanceAPIException as e:
            print(f"Error checking open position: {e}")
            return False

    def get_position_info(self):
        """Get information about the current position"""
        if not self.has_open_position():
            return None

        current_price = self.get_current_price()
        if current_price == 0.0 or self.entry_price is None or self.position_size == 0:
            return None

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
            "partial_tp_enabled": self.partial_tp_enabled,
            "partial_tp_executed": self.partial_tp_executed,
            "partial_tp_pct": self.partial_tp_pct,
            "partial_tp_size": self.partial_tp_size,
            "original_position_size": (
                self.original_position_size
                if self.original_position_size > 0
                else self.position_size
            ),
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

        # Debug logging for stop loss and take profit values
        print(
            f"DEBUG - Position: {self.position}, Stop Loss: {self.stop_loss_price}, Take Profit: {self.take_profit_price}"
        )

        # Always verify that stop loss and take profit orders are active first
        if not self.test_mode and (
            self.stop_loss_order_id or self.take_profit_order_id
        ):
            print(f"Checking orders for {self.symbol} at price ${current_price:.2f}")
            self.verify_stop_loss_take_profit_orders()

        # Calculate current profit/loss for daily target checking
        if self.position == "long":
            current_profit = self.position_size * (current_price - self.entry_price)
            current_profit_pct = (
                (current_price - self.entry_price) / self.entry_price * 100
            )
            print(
                f"Current profit for LONG position: ${current_profit:.2f} ({current_profit_pct:.2f}%) (Entry: ${self.entry_price:.2f}, Current: ${current_price:.2f})"
            )

            # Check for pyramiding opportunity
            if (
                self.enable_pyramiding
                and self.pyramid_entries < self.max_pyramid_entries
            ):
                # Only pyramid if we're in profit above threshold
                if current_profit_pct >= self.pyramid_threshold_pct * 100:
                    self.execute_pyramid_entry(current_price, timestamp)

            # Check for partial take profit if enabled and not already executed
            if (
                self.partial_tp_enabled
                and not self.partial_tp_executed
                and current_price >= self.entry_price * (1 + self.partial_tp_pct)
            ):
                print(
                    f"🎯 Partial take profit triggered at ${current_price:.2f} ({self.partial_tp_pct*100:.1f}% gain)"
                )
                self.execute_partial_take_profit(current_price, timestamp)

            # Check if we should update the trailing stop loss
            if current_price > self.entry_price and self.stop_loss_price is not None:
                # Calculate potential new stop loss based on current price
                # Use a percentage of the current profit as the trailing distance
                trailing_distance = min(
                    0.005, (current_price - self.entry_price) * 0.3
                )  # 30% of current profit, max 0.5%
                potential_stop_loss = current_price - (
                    current_price * trailing_distance
                )

                # Only update if the new stop loss would be higher than the current one
                if potential_stop_loss > self.stop_loss_price:
                    old_stop_loss = self.stop_loss_price
                    self.stop_loss_price = potential_stop_loss
                    print(
                        f"🔄 Updated trailing stop loss: ${old_stop_loss:.2f} -> ${self.stop_loss_price:.2f}"
                    )

                    # Update the stop loss order in Binance if not in test mode
                    if not self.test_mode:
                        # If we have a stop loss order ID, update it, otherwise recreate it
                        if self.stop_loss_order_id:
                            self.update_stop_loss_order()
                        else:
                            self.recreate_stop_loss_order()

            # Check for daily profit target or take profit price
            day_key = timestamp.strftime("%Y-%m-%d")
            if not hasattr(self, "daily_profits"):
                self.daily_profits = {}

            day_profit = self.daily_profits.get(day_key, 0) + current_profit

            # Check if daily profit target is reached OR take profit price is hit
            if day_profit >= self.daily_profit_target or (
                self.take_profit_price is not None
                and current_price >= self.take_profit_price
            ):
                reason = (
                    "DAILY TARGET"
                    if day_profit >= self.daily_profit_target
                    else "TAKE PROFIT"
                )
                print(
                    f"🎯 {reason} reached for LONG position! Current price: ${current_price:.2f}"
                )
                # Close position
                result = self.close_position(current_price, timestamp, reason)
            # Check for stop loss
            elif (
                self.stop_loss_price is not None
                and current_price <= self.stop_loss_price
            ):
                print(
                    f"🛑 Stop loss triggered for LONG position! Current price: ${current_price:.2f}, Stop loss: ${self.stop_loss_price:.2f}"
                )
                # Close position
                result = self.close_position(current_price, timestamp, "STOP LOSS")

        else:  # short
            current_profit = self.position_size * (self.entry_price - current_price)
            current_profit_pct = (
                (self.entry_price - current_price) / self.entry_price * 100
            )
            print(
                f"Current profit for SHORT position: ${current_profit:.2f} ({current_profit_pct:.2f}%) (Entry: ${self.entry_price:.2f}, Current: ${current_price:.2f})"
            )

            # Check for pyramiding opportunity
            if (
                self.enable_pyramiding
                and self.pyramid_entries < self.max_pyramid_entries
            ):
                # Only pyramid if we're in profit above threshold
                if current_profit_pct >= self.pyramid_threshold_pct * 100:
                    self.execute_pyramid_entry(current_price, timestamp)

            # Check for partial take profit if enabled and not already executed
            if (
                self.partial_tp_enabled
                and not self.partial_tp_executed
                and current_price <= self.entry_price * (1 - self.partial_tp_pct)
            ):
                print(
                    f"🎯 Partial take profit triggered at ${current_price:.2f} ({self.partial_tp_pct*100:.1f}% gain)"
                )
                self.execute_partial_take_profit(current_price, timestamp)

            # Check if we should update the trailing stop loss
            if current_price < self.entry_price and self.stop_loss_price is not None:
                # Calculate potential new stop loss based on current price
                # Use a percentage of the current profit as the trailing distance
                trailing_distance = min(
                    0.005, (self.entry_price - current_price) * 0.3
                )  # 30% of current profit, max 0.5%
                potential_stop_loss = current_price + (
                    current_price * trailing_distance
                )

                # Only update if the new stop loss would be lower than the current one
                if potential_stop_loss < self.stop_loss_price:
                    old_stop_loss = self.stop_loss_price
                    self.stop_loss_price = potential_stop_loss
                    print(
                        f"🔄 Updated trailing stop loss: ${old_stop_loss:.2f} -> ${self.stop_loss_price:.2f}"
                    )

                    # Update the stop loss order in Binance if not in test mode
                    if not self.test_mode:
                        # If we have a stop loss order ID, update it, otherwise recreate it
                        if self.stop_loss_order_id:
                            self.update_stop_loss_order()
                        else:
                            self.recreate_stop_loss_order()

            # Check for daily profit target or take profit price
            day_key = timestamp.strftime("%Y-%m-%d")
            if not hasattr(self, "daily_profits"):
                self.daily_profits = {}

            day_profit = self.daily_profits.get(day_key, 0) + current_profit

            # Check if daily profit target is reached OR take profit price is hit
            if day_profit >= self.daily_profit_target or (
                self.take_profit_price is not None
                and current_price <= self.take_profit_price
            ):
                reason = (
                    "DAILY TARGET"
                    if day_profit >= self.daily_profit_target
                    else "TAKE PROFIT"
                )
                print(
                    f"🎯 {reason} reached for SHORT position! Current price: ${current_price:.2f}"
                )
                # Close position
                result = self.close_position(current_price, timestamp, reason)
            # Check for stop loss
            elif (
                self.stop_loss_price is not None
                and current_price >= self.stop_loss_price
            ):
                print(
                    f"🛑 Stop loss triggered for SHORT position! Current price: ${current_price:.2f}, Stop loss: ${self.stop_loss_price:.2f}"
                )
                # Close position
                result = self.close_position(current_price, timestamp, "STOP LOSS")

        if result:
            # Format result message
            position_type = self.position.upper() if self.position else "POSITION"
            reason = result["reason"]
            profit = result["profit"]

            result_message = (
                f"{position_type} closed - {reason}\n"
                f"Exit price: ${current_price:.2f}\n"
                f"Profit: ${abs(profit):.2f}"
            )

            # Save results
            self.save_trading_results()

            return result_message
        else:
            # Only print daily profit info if we have a position
            if self.has_open_position():
                day_key = timestamp.strftime("%Y-%m-%d")
                if not hasattr(self, "daily_profits"):
                    self.daily_profits = {}
                day_profit = self.daily_profits.get(day_key, 0) + current_profit
                print(
                    f"Daily profit: ${day_profit:.2f} / Target: ${self.daily_profit_target:.2f}"
                )

        return None

    def execute_partial_take_profit(self, current_price, timestamp):
        """Execute a partial take profit by closing part of the position"""
        if not self.has_open_position() or self.partial_tp_executed:
            return

        try:
            # Store original position size if not already stored
            if self.original_position_size == 0:
                self.original_position_size = self.position_size

            # Calculate the portion of the position to close
            close_size = self.position_size * self.partial_tp_size

            # Get the correct precision for the quantity
            quantity_precision = self.get_quantity_precision()

            # Format the close size with the correct precision
            close_size = float("{:0.0{}f}".format(close_size, quantity_precision))

            # Ensure close size is greater than zero
            if close_size <= 0:
                print("Error: Close size for partial take profit is zero or negative.")
                return

            if not self.test_mode:
                # Cancel existing take profit and stop loss orders
                try:
                    # Cancel all open orders for the symbol
                    self.client.futures_cancel_all_open_orders(symbol=self.symbol)
                    print(
                        f"Cancelled all open orders for {self.symbol} before partial take profit"
                    )

                    # Reset order IDs since we've canceled them
                    self.stop_loss_order_id = None
                    self.take_profit_order_id = None
                except BinanceAPIException as e:
                    print(f"Error cancelling orders before partial take profit: {e}")

                # Execute the partial close
                side = "SELL" if self.position == "long" else "BUY"
                print(
                    f"Executing partial take profit: {side} {close_size} {self.symbol} at MARKET price (${current_price:.2f})"
                )

                order = self.client.futures_create_order(
                    symbol=self.symbol, side=side, type="MARKET", quantity=close_size
                )
                print(f"Partial take profit order executed: {order}")
            else:
                # Test mode - simulate order
                side = "SELL" if self.position == "long" else "BUY"
                print(
                    f"TEST MODE: Simulating partial take profit: {side} {close_size} {self.symbol} at MARKET price (${current_price:.2f})"
                )

            # Calculate profit for the closed portion
            if self.position == "long":
                profit = (current_price - self.entry_price) * close_size
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
            else:  # short
                profit = (self.entry_price - current_price) * close_size
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100

            # Update test balance in test mode
            if self.test_mode:
                self.test_balance += profit
                print(
                    f"TEST MODE: Updated balance to ${self.test_balance:.2f} (partial profit: ${profit:.2f})"
                )

            # Update per-symbol balance for compound interest
            self.symbol_balance += profit
            self.total_symbol_profit += profit

            # Update position size
            remaining_size = self.position_size - close_size
            self.position_size = float(
                "{:0.0{}f}".format(remaining_size, quantity_precision)
            )

            # Move stop loss to break-even (entry price)
            self.stop_loss_price = self.entry_price
            print(f"🔄 Moved stop loss to break-even: ${self.stop_loss_price:.2f}")

            # Add to trade history
            partial_close_record = {
                "timestamp": timestamp,
                "action": "PARTIAL_CLOSE",
                "price": current_price,
                "size": close_size,
                "value": close_size * current_price,
                "profit": profit,
                "profit_pct": profit_pct,
                "reason": "PARTIAL_TAKE_PROFIT",
                "order_type": "MARKET",
            }

            self.trade_history.append(partial_close_record)
            if self.test_mode:
                self.test_trades.append(partial_close_record)

            # Update daily profits
            day_key = timestamp.strftime("%Y-%m-%d")
            if day_key not in self.daily_profits:
                self.daily_profits[day_key] = 0
            self.daily_profits[day_key] += profit

            # Send notification
            message = (
                f"🔔 Partial take profit executed\n"
                f"Symbol: {self.symbol}\n"
                f"Type: {self.position.upper()}\n"
                f"Entry: ${self.entry_price:.2f}\n"
                f"Exit: ${current_price:.2f}\n"
                f"Closed: {close_size} units ({self.partial_tp_size*100:.0f}% of position)\n"
                f"Profit: ${profit:.2f} ({profit_pct:.2f}%)\n"
                f"Remaining: {self.position_size} units\n"
                f"Stop loss moved to break-even: ${self.stop_loss_price:.2f}\n"
                f"Account Balance: ${self.get_account_balance():.2f}\n"
                f"Symbol Balance: ${self.symbol_balance:.2f}"
            )
            self.send_notification(message)

            # Mark partial take profit as executed
            self.partial_tp_executed = True

            # Place new stop loss order at break-even
            if not self.test_mode:
                # Use recreate_stop_loss_order instead of update_stop_loss_order
                # since we've already canceled the previous order
                self.recreate_stop_loss_order()

            print(
                f"✅ Partial take profit executed successfully. Remaining position: {self.position_size} units with stop loss at break-even."
            )

        except BinanceAPIException as e:
            print(f"❌ Error executing partial take profit: {e}")

    def verify_stop_loss_take_profit_orders(self):
        """Verify that stop loss and take profit orders are active in Binance"""
        if self.test_mode:
            return

        try:
            # Get all open orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)

            # Check if our stop loss and take profit orders are in the list
            stop_loss_found = False
            take_profit_found = False

            print(
                f"Verifying orders for {self.symbol}. Found {len(open_orders)} open orders."
            )

            # Debug: Print all order details
            self.debug_print_orders(open_orders)

            for order in open_orders:
                order_id = str(order["orderId"])
                order_type = order.get("type", "UNKNOWN")
                order_status = order.get("status", "UNKNOWN")

                if order_id == str(self.stop_loss_order_id):
                    stop_loss_found = True
                    print(
                        f"✅ Stop loss order confirmed active: ID={order_id}, Type={order_type}, Status={order_status}, Price={order.get('stopPrice', 'N/A')}"
                    )
                elif order_id == str(self.take_profit_order_id):
                    take_profit_found = True
                    print(
                        f"✅ Take profit order confirmed active: ID={order_id}, Type={order_type}, Status={order_status}, Price={order.get('stopPrice', 'N/A')}"
                    )
                else:
                    print(
                        f"Found other order: ID={order_id}, Type={order_type}, Status={order_status}"
                    )

            # If orders are not found, recreate them
            if not stop_loss_found and self.stop_loss_order_id:
                print(
                    f"⚠️ Warning: Stop loss order {self.stop_loss_order_id} not found in open orders. Recreating..."
                )
                self.recreate_stop_loss_order()

            if not take_profit_found and self.take_profit_order_id:
                print(
                    f"⚠️ Warning: Take profit order {self.take_profit_order_id} not found in open orders. Recreating..."
                )
                self.recreate_take_profit_order()

        except BinanceAPIException as e:
            print(f"Error verifying stop loss and take profit orders: {e}")

    def debug_print_orders(self, orders):
        """Print detailed information about orders for debugging"""
        print("\n===== DEBUG: ORDER DETAILS =====")
        for i, order in enumerate(orders):
            print(f"Order {i+1}:")
            for key, value in order.items():
                print(f"  {key}: {value}")
            print("------------------------------")
        print("================================\n")

    def recreate_stop_loss_order(self):
        """Recreate stop loss order if it's missing"""
        if self.test_mode or not self.has_open_position():
            print("Cannot recreate stop loss order: test mode or no open position")
            return

        if self.stop_loss_price is None:
            print("Cannot recreate stop loss order: stop_loss_price is None")
            return

        print(
            f"DEBUG - Recreating stop loss order for {self.position.upper()} position with size {self.position_size}"
        )

        try:
            # Format price with correct precision
            price_precision = self.get_price_precision()
            stop_loss_price = float(
                "{:0.0{}f}".format(self.stop_loss_price, price_precision)
            )

            # For LONG positions: SELL to close
            # For SHORT positions: BUY to close
            close_side = "SELL" if self.position == "long" else "BUY"

            print(
                f"Recreating stop loss order: {self.symbol} {close_side} at {stop_loss_price}"
            )

            # Place stop loss order with exact parameters from Binance documentation
            stop_loss_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=stop_loss_price,
                reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                quantity=self.position_size,
                workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                timeInForce="GTC",  # Good Till Cancelled
            )
            self.stop_loss_order_id = stop_loss_order["orderId"]
            print(f"✅ Stop loss order recreated: {stop_loss_order}")

        except BinanceAPIException as e:
            print(f"❌ Error recreating stop loss order: {e}")
            # Log detailed error information
            print(
                f"Error details: Symbol={self.symbol}, Position={self.position}, Size={self.position_size}, Stop Price={self.stop_loss_price}"
            )

    def recreate_take_profit_order(self):
        """Recreate take profit order if it's missing"""
        if self.test_mode or not self.has_open_position():
            return

        try:
            # Format price with correct precision
            price_precision = self.get_price_precision()
            take_profit_price = float(
                "{:0.0{}f}".format(self.take_profit_price, price_precision)
            )

            # For LONG positions: SELL to close
            # For SHORT positions: BUY to close
            close_side = "SELL" if self.position == "long" else "BUY"

            print(
                f"Recreating take profit order: {self.symbol} {close_side} at {take_profit_price}"
            )

            # Place take profit order with exact parameters from Binance documentation
            take_profit_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=take_profit_price,
                reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                quantity=self.position_size,
                workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                timeInForce="GTC",  # Good Till Cancelled
            )
            self.take_profit_order_id = take_profit_order["orderId"]
            print(f"✅ Take profit order recreated: {take_profit_order}")

        except BinanceAPIException as e:
            print(f"❌ Error recreating take profit order: {e}")

    def get_symbol_info(self):
        """Get symbol information including precision requirements"""
        try:
            # For spot trading
            exchange_info = self.client.get_exchange_info()

            # Find the symbol info
            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == self.symbol:
                    return symbol_info

            return None
        except BinanceAPIException as e:
            print(f"Error getting symbol info: {e}")
            return None

    def get_futures_symbol_info(self):
        """Get futures symbol information including precision requirements"""
        try:
            # For futures trading
            futures_exchange_info = self.client.futures_exchange_info()

            # Find the symbol info
            for symbol_info in futures_exchange_info["symbols"]:
                if symbol_info["symbol"] == self.symbol:
                    return symbol_info

            return None
        except BinanceAPIException as e:
            print(f"Error getting futures symbol info: {e}")
            return None

    def get_quantity_precision(self):
        """Get the quantity precision for the symbol"""
        # Try futures first
        futures_info = self.get_futures_symbol_info()
        if futures_info and "quantityPrecision" in futures_info:
            return futures_info["quantityPrecision"]

        # Fall back to spot trading calculation
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 5 decimal places if we can't get the info
            return 5

        # Get the lot size filter
        for filter in symbol_info["filters"]:
            if filter["filterType"] == "LOT_SIZE":
                step_size = float(filter["stepSize"])
                # Calculate precision based on step size
                if step_size == 1.0:
                    return 0
                precision = 0
                step_size_str = "{:0.8f}".format(step_size)
                while step_size_str[len(step_size_str) - 1 - precision] == "0":
                    precision += 1
                return 8 - precision

        # Default to 5 decimal places if we can't find the LOT_SIZE filter
        return 5

    def get_price_precision(self):
        """Get the price precision for the symbol"""
        # Try futures first
        futures_info = self.get_futures_symbol_info()
        if futures_info and "pricePrecision" in futures_info:
            return futures_info["pricePrecision"]

        # Fall back to spot trading calculation
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 2 decimal places if we can't get the info
            return 2

        # Get the price filter
        for filter in symbol_info["filters"]:
            if filter["filterType"] == "PRICE_FILTER":
                tick_size = float(filter["tickSize"])
                # Calculate precision based on tick size
                if tick_size == 1.0:
                    return 0
                precision = 0
                tick_size_str = "{:0.8f}".format(tick_size)
                while tick_size_str[len(tick_size_str) - 1 - precision] == "0":
                    precision += 1
                return 8 - precision

        # Default to 2 decimal places if we can't find the PRICE_FILTER filter
        return 2

    def get_min_notional(self):
        """Get the minimum notional value (order value) for the symbol"""
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 10 if we can't get the info
            return 10.0

        # Get the min notional filter
        for filter in symbol_info["filters"]:
            if filter["filterType"] == "MIN_NOTIONAL":
                return float(filter["minNotional"])

        # Default to 10 if we can't find the MIN_NOTIONAL filter
        return 10.0

    def get_min_quantity(self):
        """Get the minimum quantity for the symbol"""
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 0.001 if we can't get the info
            return 0.001

        # Get the lot size filter
        for filter in symbol_info["filters"]:
            if filter["filterType"] == "LOT_SIZE":
                return float(filter["minQty"])

        # Default to 0.001 if we can't find the LOT_SIZE filter
        return 0.001

    def calculate_dynamic_position_size(self, current_price, atr_value=None):
        """Calculate position size dynamically based on volatility and market conditions"""
        # Get base position size using current method
        base_position_size = self.calculate_position_size(current_price)
        
        # Get latest data for analysis
        latest_df = self.get_latest_data(lookback_candles=20)
        if latest_df is None or len(latest_df) < 20:
            return base_position_size
            
        # Calculate ATR if not provided
        if atr_value is None:
            atr_value = latest_df['ATR'].iloc[-1] if 'ATR' in latest_df.columns else (current_price * 0.02)
            
        # Calculate volatility ratio (ATR as percentage of price)
        volatility_ratio = atr_value / current_price
        
        # Get market trend info
        trend_direction, trend_strength = self.analyze_market_trend()
        
        # Calculate trend alignment factor (0.5 to 1.5)
        if self.position == "long":
            trend_factor = 1.0 + (trend_strength/100 * 0.5) if trend_direction == 1 else 1.0 - (trend_strength/100 * 0.5)
        else:  # short
            trend_factor = 1.0 + (trend_strength/100 * 0.5) if trend_direction == -1 else 1.0 - (trend_strength/100 * 0.5)
            
        # Adjust position size based on volatility
        volatility_adjustment = 1.0
        if volatility_ratio > 0.03:  # High volatility
            volatility_adjustment = 0.6  # Reduce position size by 40%
        elif volatility_ratio > 0.02:  # Medium-high volatility
            volatility_adjustment = 0.8  # Reduce position size by 20%
        elif volatility_ratio < 0.01:  # Low volatility
            volatility_adjustment = 1.2  # Increase position size by 20%
            
        # Performance adjustment based on recent trades
        performance_adjustment = 1.0
        if len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-5:]
            profitable_trades = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
            
            if profitable_trades >= 4:  # 80%+ win rate
                performance_adjustment = 1.2
            elif profitable_trades <= 1:  # 20% or less win rate
                performance_adjustment = 0.7
                
        # Market condition adjustments
        market_condition_adjustment = 1.0
        if 'rsi' in latest_df.columns:
            rsi = latest_df['rsi'].iloc[-1]
            if (self.position == "long" and rsi > 70) or (self.position == "short" and rsi < 30):
                market_condition_adjustment = 0.8  # Reduce size in overbought/oversold conditions
                
        # Calculate final position size with all adjustments
        adjusted_position_size = base_position_size * volatility_adjustment * trend_factor * performance_adjustment * market_condition_adjustment
        
        # Ensure position size doesn't exceed maximum risk
        max_position_size = self.calculate_position_size(current_price, risk_pct=0.02)  # Max 2% risk
        adjusted_position_size = min(adjusted_position_size, max_position_size)
        
        # Format with correct precision
        quantity_precision = self.get_quantity_precision()
        adjusted_position_size = float("{:0.0{}f}".format(adjusted_position_size, quantity_precision))
        
        # Log adjustments
        print(f"\nPosition Size Adjustments:")
        print(f"Base size: {base_position_size:.4f}")
        print(f"Volatility adjustment: {volatility_adjustment:.2f}x")
        print(f"Trend factor: {trend_factor:.2f}x")
        print(f"Performance adjustment: {performance_adjustment:.2f}x")
        print(f"Market condition adjustment: {market_condition_adjustment:.2f}x")
        print(f"Final size: {adjusted_position_size:.4f}")
        
        return adjusted_position_size

    def calculate_position_size(self, current_price, risk_pct=None):
        """Calculate position size based on risk management"""
        if risk_pct is None:
            risk_pct = self.risk_pct

        # Get account balance (either real or test)
        account_balance = self.get_account_balance()

        # For compound interest, use symbol balance instead of account balance
        if self.compound_interest:
            account_balance = self.symbol_balance

        # Calculate the dollar amount to risk
        risk_amount = account_balance * risk_pct

        # Calculate stop loss distance as percentage of price
        if self.use_atr_for_risk:
            # Get latest data
            latest_df = self.get_latest_data(lookback_candles=14)

            if latest_df is not None and len(latest_df) > 0:
                # Use ATR for stop loss distance
                atr = (
                    latest_df["atr"].iloc[-1]
                    if "atr" in latest_df.columns
                    else (current_price * 0.01)
                )
                stop_loss_distance = atr * self.atr_multiplier
            else:
                # Fallback to percentage-based stop loss
                stop_loss_distance = current_price * self.stop_loss_pct
        else:
            # Use percentage-based stop loss
            stop_loss_distance = current_price * self.stop_loss_pct

        # Calculate position size based on risk amount and stop loss distance
        position_size = (risk_amount / stop_loss_distance) * self.leverage

        # If using full investment, calculate position size based on full account balance
        if self.use_full_investment:
            full_position_size = (account_balance * self.leverage) / current_price
            position_size = min(position_size, full_position_size)

        # If using full margin, use the maximum position size allowed by the leverage
        if self.use_full_margin:
            position_size = (account_balance * self.leverage) / current_price

        # Get the correct precision for the quantity
        quantity_precision = self.get_quantity_precision()

        # Format the position size with the correct precision
        position_size = float("{:0.0{}f}".format(position_size, quantity_precision))

        return position_size

    def calculate_dynamic_take_profit(self, current_price):
        """Calculate dynamic take profit levels based on market conditions"""
        # Get latest data
        latest_df = self.get_latest_data(lookback_candles=20)
        if latest_df is None or len(latest_df) < 20:
            return current_price * (1 + self.take_profit_pct if self.position == "long" else 1 - self.take_profit_pct)
            
        # Get ATR for volatility-based adjustments
        atr = latest_df['ATR'].iloc[-1] if 'ATR' in latest_df.columns else (current_price * 0.02)
        volatility_ratio = atr / current_price
        
        # Get trend information
        trend_direction, trend_strength = self.analyze_market_trend()
        
        # Base take profit percentage
        base_tp_pct = self.take_profit_pct
        
        # Adjust based on trend alignment
        if (self.position == "long" and trend_direction == 1) or (self.position == "short" and trend_direction == -1):
            # Aligned with trend - extend take profit
            trend_factor = 1.0 + (trend_strength / 100)  # 1.0 to 2.0
        else:
            # Counter-trend - reduce take profit
            trend_factor = 0.8
            
        # Adjust based on volatility
        volatility_factor = 1.0 + (volatility_ratio * 5)  # Increase TP in high volatility
        
        # Market condition adjustments
        market_factor = 1.0
        if 'rsi' in latest_df.columns:
            rsi = latest_df['rsi'].iloc[-1]
            if (self.position == "long" and rsi > 70) or (self.position == "short" and rsi < 30):
                market_factor = 0.8  # Reduce TP in overbought/oversold conditions
                
        # Calculate final take profit percentage
        adjusted_tp_pct = base_tp_pct * trend_factor * volatility_factor * market_factor
        
        # Cap the maximum take profit percentage
        max_tp_pct = 0.1  # 10%
        adjusted_tp_pct = min(adjusted_tp_pct, max_tp_pct)
        
        # Calculate take profit price
        if self.position == "long":
            take_profit_price = current_price * (1 + adjusted_tp_pct)
        else:  # short
            take_profit_price = current_price * (1 - adjusted_tp_pct)
            
        # Format with correct precision
        price_precision = self.get_price_precision()
        take_profit_price = float("{:0.0{}f}".format(take_profit_price, price_precision))
        
        # Log adjustments
        print(f"\nTake Profit Adjustments:")
        print(f"Base TP%: {base_tp_pct*100:.2f}%")
        print(f"Trend factor: {trend_factor:.2f}x")
        print(f"Volatility factor: {volatility_factor:.2f}x")
        print(f"Market factor: {market_factor:.2f}x")
        print(f"Final TP%: {adjusted_tp_pct*100:.2f}%")
        print(f"TP Price: {take_profit_price:.2f}")
        
        return take_profit_price

    def calculate_dynamic_stop_loss(self, current_price):
        """Calculate dynamic stop loss based on market conditions"""
        # Get latest data
        latest_df = self.get_latest_data(lookback_candles=20)
        if latest_df is None or len(latest_df) < 20:
            return current_price * (1 - self.stop_loss_pct if self.position == "long" else 1 + self.stop_loss_pct)
            
        # Get ATR for volatility-based adjustments
        atr = latest_df['ATR'].iloc[-1] if 'ATR' in latest_df.columns else (current_price * 0.02)
        
        # Base stop loss percentage (use ATR)
        base_sl_pct = (atr / current_price) * self.atr_multiplier
        
        # Get trend information
        trend_direction, trend_strength = self.analyze_market_trend()
        
        # Adjust based on trend alignment
        if (self.position == "long" and trend_direction == 1) or (self.position == "short" and trend_direction == -1):
            # Aligned with trend - can use tighter stop
            trend_factor = 0.8
        else:
            # Counter-trend - need wider stop
            trend_factor = 1.2
            
        # Market condition adjustments
        market_factor = 1.0
        if 'rsi' in latest_df.columns:
            rsi = latest_df['rsi'].iloc[-1]
            if (self.position == "long" and rsi < 30) or (self.position == "short" and rsi > 70):
                market_factor = 0.8  # Tighter stop in oversold/overbought conditions
                
        # Calculate final stop loss percentage
        adjusted_sl_pct = base_sl_pct * trend_factor * market_factor
        
        # Cap the maximum stop loss percentage
        max_sl_pct = 0.03  # 3%
        adjusted_sl_pct = min(adjusted_sl_pct, max_sl_pct)
        
        # Calculate stop loss price
        if self.position == "long":
            stop_loss_price = current_price * (1 - adjusted_sl_pct)
        else:  # short
            stop_loss_price = current_price * (1 + adjusted_sl_pct)
            
        # Format with correct precision
        price_precision = self.get_price_precision()
        stop_loss_price = float("{:0.0{}f}".format(stop_loss_price, price_precision))
        
        # Log adjustments
        print(f"\nStop Loss Adjustments:")
        print(f"Base SL%: {base_sl_pct*100:.2f}%")
        print(f"Trend factor: {trend_factor:.2f}x")
        print(f"Market factor: {market_factor:.2f}x")
        print(f"Final SL%: {adjusted_sl_pct*100:.2f}%")
        print(f"SL Price: {stop_loss_price:.2f}")
        
        return stop_loss_price

    def analyze_market_trend(self):
        """
        Analyze the current market trend to determine trend direction and strength

        Returns:
            Tuple of (trend_direction, trend_strength)
            trend_direction: 1 for uptrend, -1 for downtrend, 0 for neutral
            trend_strength: 0-100 scale where 100 is strongest trend
        """
        # Get latest data
        latest_df = self.get_latest_data(lookback_candles=50)

        if latest_df is None or len(latest_df) < 50:
            return (0, 0)  # Neutral, no strength if not enough data

        # Use multiple indicators to determine trend

        # 1. Moving Average Analysis
        if (
            "ema_8" in latest_df.columns
            and "ema_21" in latest_df.columns
            and "ema_50" in latest_df.columns
        ):
            ema_8 = latest_df["ema_8"].iloc[-1]
            ema_21 = latest_df["ema_21"].iloc[-1]
            ema_50 = latest_df["ema_50"].iloc[-1]

            # Check alignment of EMAs
            if ema_8 > ema_21 > ema_50:
                ma_direction = 1  # Strong uptrend
                ma_strength = 80
            elif ema_8 < ema_21 < ema_50:
                ma_direction = -1  # Strong downtrend
                ma_strength = 80
            elif ema_8 > ema_21 and ema_21 < ema_50:
                ma_direction = 1  # Potential uptrend starting
                ma_strength = 40
            elif ema_8 < ema_21 and ema_21 > ema_50:
                ma_direction = -1  # Potential downtrend starting
                ma_strength = 40
            else:
                ma_direction = 0  # Mixed signals
                ma_strength = 20
        else:
            # Simple moving average calculation if EMAs not available
            prices = latest_df["close"].values
            ma_8 = np.mean(prices[-8:])
            ma_21 = np.mean(prices[-21:])
            ma_50 = np.mean(prices[-50:])

            if ma_8 > ma_21 > ma_50:
                ma_direction = 1
                ma_strength = 70
            elif ma_8 < ma_21 < ma_50:
                ma_direction = -1
                ma_strength = 70
            else:
                ma_direction = 0
                ma_strength = 30

        # 2. Price Action Analysis
        current_price = latest_df["close"].iloc[-1]
        price_10_ago = latest_df["close"].iloc[-10]
        price_30_ago = latest_df["close"].iloc[-30]

        short_term_change = (current_price - price_10_ago) / price_10_ago
        medium_term_change = (current_price - price_30_ago) / price_30_ago

        if short_term_change > 0.02 and medium_term_change > 0.05:
            pa_direction = 1  # Strong uptrend
            pa_strength = 90
        elif short_term_change < -0.02 and medium_term_change < -0.05:
            pa_direction = -1  # Strong downtrend
            pa_strength = 90
        elif short_term_change > 0.01:
            pa_direction = 1  # Moderate uptrend
            pa_strength = 60
        elif short_term_change < -0.01:
            pa_direction = -1  # Moderate downtrend
            pa_strength = 60
        else:
            pa_direction = 0  # Sideways
            pa_strength = 20

        # 3. Volume Analysis
        if "volume" in latest_df.columns:
            recent_volume = np.mean(latest_df["volume"].iloc[-5:])
            avg_volume = np.mean(latest_df["volume"].iloc[-20:])
            volume_ratio = recent_volume / avg_volume

            # Higher volume in trend direction strengthens the signal
            volume_strength = min(100, volume_ratio * 50)
        else:
            volume_strength = 50  # Neutral if no volume data

        # Combine signals with weightings
        ma_weight = 0.4
        pa_weight = 0.4
        vol_weight = 0.2

        # Calculate final direction (weighted sum of directions)
        direction_score = (ma_direction * ma_weight) + (pa_direction * pa_weight)

        if direction_score > 0.2:
            final_direction = 1
        elif direction_score < -0.2:
            final_direction = -1
        else:
            final_direction = 0

        # Calculate final strength (weighted average of strengths)
        final_strength = (
            (ma_strength * ma_weight)
            + (pa_strength * pa_weight)
            + (volume_strength * vol_weight)
        )

        # Adjust strength based on agreement between indicators
        if ma_direction == pa_direction and ma_direction != 0:
            final_strength *= 1.2  # Boost strength when indicators agree

        # Cap at 100
        final_strength = min(100, final_strength)

        print(
            f"Market trend analysis: Direction={final_direction}, Strength={final_strength:.1f}%"
        )
        print(f"  MA Analysis: Direction={ma_direction}, Strength={ma_strength}%")
        print(f"  Price Action: Direction={pa_direction}, Strength={pa_strength}%")
        print(f"  Volume: Ratio={volume_ratio:.2f}, Strength={volume_strength:.1f}%")

        return (final_direction, final_strength)

    def generate_enhanced_signal(self, latest_df):
        """
        Generate enhanced trading signals using multiple indicators for confirmation

        Args:
            latest_df: DataFrame with latest price data and indicators

        Returns:
            Tuple of (signal, confidence)
            signal: 1 for long, -1 for short, 0 for no signal
            confidence: 0-100 scale where 100 is highest confidence
        """
        print("\n========== ENHANCED SIGNAL ANALYSIS START ==========")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Symbol: {self.symbol}")
        print(f"Current Price: ${latest_df.iloc[-1]['close']:.4f}")
        print(
            f"Signal Threshold: {self.signal_confirmation_threshold} indicators required"
        )

        if latest_df is None or len(latest_df) < 50:
            print(
                "DEBUG: Not enough data for signal analysis (need at least 50 candles)"
            )
            print("========== ENHANCED SIGNAL ANALYSIS END ==========\n")
            return (0, 0)  # No signal if not enough data

        # Get the latest candle
        latest = latest_df.iloc[-1]

        # Initialize signal counters
        long_signals = 0
        short_signals = 0
        total_indicators = 0

        # Debug: Print available columns for troubleshooting
        print(f"DEBUG: Available indicators: {', '.join(latest_df.columns)}")

        # Calculate missing indicators if needed
        print("\n--- Calculating Missing Indicators ---")
        df = latest_df.copy()

        # Calculate EMAs if missing
        if "ema_8" not in df.columns or "ema_21" not in df.columns:
            print("DEBUG: Calculating EMAs on the fly")
            if "ema_8" not in df.columns:
                df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
            if "ema_21" not in df.columns:
                df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

        # Calculate RSI if missing
        if "rsi" not in df.columns:
            print("DEBUG: Calculating RSI on the fly")
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        # Calculate MACD if missing
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            print("DEBUG: Calculating MACD on the fly")
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Calculate Bollinger Bands if missing
        if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
            print("DEBUG: Calculating Bollinger Bands on the fly")
            window = 20
            std_dev = 2
            rolling_mean = df["close"].rolling(window=window).mean()
            rolling_std = df["close"].rolling(window=window).std()
            df["bb_upper"] = rolling_mean + (rolling_std * std_dev)
            df["bb_lower"] = rolling_mean - (rolling_std * std_dev)
            df["bb_middle"] = rolling_mean

        # Calculate Stochastic Oscillator if missing
        if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
            print("DEBUG: Calculating Stochastic Oscillator on the fly")
            window = 14
            k_window = 3
            d_window = 3
            # Calculate %K
            low_min = df["low"].rolling(window=window).min()
            high_max = df["high"].rolling(window=window).max()
            df["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min))
            # Calculate %D
            df["stoch_d"] = df["stoch_k"].rolling(window=d_window).mean()

        # Calculate ADX if missing
        if "adx" not in df.columns:
            print("DEBUG: Calculating ADX on the fly")
            window = 14
            # True Range
            df["tr1"] = abs(df["high"] - df["low"])
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            df["atr"] = df["tr"].rolling(window=window).mean()

            # Plus Directional Movement (+DM)
            df["plus_dm"] = 0.0
            df.loc[
                (df["high"] - df["high"].shift() > df["low"].shift() - df["low"])
                & (df["high"] - df["high"].shift() > 0),
                "plus_dm",
            ] = (
                df["high"] - df["high"].shift()
            )

            # Minus Directional Movement (-DM)
            df["minus_dm"] = 0.0
            df.loc[
                (df["low"].shift() - df["low"] > df["high"] - df["high"].shift())
                & (df["low"].shift() - df["low"] > 0),
                "minus_dm",
            ] = (
                df["low"].shift() - df["low"]
            )

            # Smooth +DM and -DM
            df["plus_di"] = 100 * (
                df["plus_dm"].rolling(window=window).mean() / df["atr"]
            )
            df["minus_di"] = 100 * (
                df["minus_dm"].rolling(window=window).mean() / df["atr"]
            )

            # Directional Movement Index (DX)
            df["dx"] = 100 * (
                abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
            )

            # Average Directional Index (ADX)
            df["adx"] = df["dx"].rolling(window=window).mean()

        # Update latest with calculated indicators
        latest = df.iloc[-1]

        # 1. Moving Average Crossover
        print("\n--- EMA Crossover Analysis ---")
        if "ema_8" in df.columns and "ema_21" in df.columns:
            total_indicators += 1

            # Current values
            ema_8_current = latest["ema_8"]
            ema_21_current = latest["ema_21"]

            # Previous values
            ema_8_prev = df.iloc[-2]["ema_8"]
            ema_21_prev = df.iloc[-2]["ema_21"]

            print(
                f"DEBUG: EMA 8 current: {ema_8_current:.4f}, previous: {ema_8_prev:.4f}"
            )
            print(
                f"DEBUG: EMA 21 current: {ema_21_current:.4f}, previous: {ema_21_prev:.4f}"
            )

            # Check for crossover
            if ema_8_prev <= ema_21_prev and ema_8_current > ema_21_current:
                # Bullish crossover
                long_signals += 1
                print("EMA Crossover: BULLISH (8 EMA crossed above 21 EMA) ✅")
            elif ema_8_prev >= ema_21_prev and ema_8_current < ema_21_current:
                # Bearish crossover
                short_signals += 1
                print("EMA Crossover: BEARISH (8 EMA crossed below 21 EMA) ✅")
            else:
                # Check if EMA 8 is above EMA 21 (bullish trend)
                if ema_8_current > ema_21_current:
                    print(
                        "EMA Crossover: NEUTRAL but trending BULLISH (EMA 8 above EMA 21) ⚠️"
                    )
                    # Add a smaller weight for existing trend
                    long_signals += 0.5
                # Check if EMA 8 is below EMA 21 (bearish trend)
                elif ema_8_current < ema_21_current:
                    print(
                        "EMA Crossover: NEUTRAL but trending BEARISH (EMA 8 below EMA 21) ⚠️"
                    )
                    # Add a smaller weight for existing trend
                    short_signals += 0.5
                else:
                    print("EMA Crossover: NEUTRAL (no recent crossover) ❌")
        else:
            print("DEBUG: EMA indicators not available in dataframe")

        # 2. RSI
        print("\n--- RSI Analysis ---")
        if "rsi" in df.columns:
            total_indicators += 1
            rsi = latest["rsi"]
            print(f"DEBUG: RSI value: {rsi:.2f}")

            if rsi < 30:
                # Oversold - bullish
                long_signals += 1
                print(f"RSI: BULLISH (Oversold at {rsi:.1f}) ✅")
            elif rsi > 70:
                # Overbought - bearish
                short_signals += 1
                print(f"RSI: BEARISH (Overbought at {rsi:.1f}) ✅")
            else:
                # Add smaller weight for trending conditions
                if rsi > 50 and rsi < 70:
                    print(
                        f"RSI: NEUTRAL but trending BULLISH ({rsi:.1f} is between 50-70) ⚠️"
                    )
                    short_signals += 0.3
                elif rsi > 30 and rsi < 50:
                    print(
                        f"RSI: NEUTRAL but trending BEARISH ({rsi:.1f} is between 30-50) ⚠️"
                    )
                    long_signals += 0.3
                else:
                    print(f"RSI: NEUTRAL ({rsi:.1f} is between 30-70) ❌")
        else:
            print("DEBUG: RSI indicator not available in dataframe")

        # 3. MACD
        print("\n--- MACD Analysis ---")
        if "macd" in df.columns and "macd_signal" in df.columns:
            total_indicators += 1

            # Current values
            macd_current = latest["macd"]
            signal_current = latest["macd_signal"]

            # Previous values
            macd_prev = df.iloc[-2]["macd"]
            signal_prev = df.iloc[-2]["macd_signal"]

            print(f"DEBUG: MACD current: {macd_current:.6f}, previous: {macd_prev:.6f}")
            print(
                f"DEBUG: Signal current: {signal_current:.6f}, previous: {signal_prev:.6f}"
            )

            # Check for crossover
            if macd_prev <= signal_prev and macd_current > signal_current:
                # Bullish crossover
                long_signals += 1
                print(f"MACD: BULLISH (MACD crossed above signal line) ✅")
            elif macd_prev >= signal_prev and macd_current < signal_current:
                # Bearish crossover
                short_signals += 1
                print(f"MACD: BEARISH (MACD crossed below signal line) ✅")
            else:
                # Check MACD position relative to zero line for trend bias
                if macd_current > 0:
                    print(f"MACD: NEUTRAL but trending BULLISH (MACD above zero) ⚠️")
                    long_signals += 0.3
                elif macd_current < 0:
                    print(f"MACD: NEUTRAL but trending BEARISH (MACD below zero) ⚠️")
                    short_signals += 0.3
                else:
                    print(f"MACD: NEUTRAL (no recent crossover) ❌")
        else:
            print("DEBUG: MACD indicators not available in dataframe")

        # 4. Bollinger Bands
        print("\n--- Bollinger Bands Analysis ---")
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            total_indicators += 1

            close = latest["close"]
            bb_upper = latest["bb_upper"]
            bb_lower = latest["bb_lower"]
            bb_middle = (
                latest["bb_middle"]
                if "bb_middle" in latest.index
                else (bb_upper + bb_lower) / 2
            )

            # Previous close
            prev_close = df.iloc[-2]["close"]

            print(f"DEBUG: Close current: {close:.4f}, previous: {prev_close:.4f}")
            print(
                f"DEBUG: BB Upper: {bb_upper:.4f}, Middle: {bb_middle:.4f}, Lower: {bb_lower:.4f}"
            )

            # Calculate position within the bands (0-100%)
            band_width = bb_upper - bb_lower
            if band_width > 0:  # Avoid division by zero
                position_pct = ((close - bb_lower) / band_width) * 100
                print(f"DEBUG: Price position: {position_pct:.1f}% of band width")

                # Check for bounces off bands
                if prev_close <= bb_lower and close > bb_lower:
                    # Price bouncing off lower band - bullish
                    long_signals += 1
                    print(f"Bollinger Bands: BULLISH (Price bounced off lower band) ✅")
                elif prev_close >= bb_upper and close < bb_upper:
                    # Price bouncing off upper band - bearish
                    short_signals += 1
                    print(f"Bollinger Bands: BEARISH (Price bounced off upper band) ✅")
                else:
                    # Add smaller weight for position within bands
                    if position_pct < 20:
                        print(
                            f"Bollinger Bands: NEUTRAL but near lower band ({position_pct:.1f}%) ⚠️"
                        )
                        long_signals += 0.3
                    elif position_pct > 80:
                        print(
                            f"Bollinger Bands: NEUTRAL but near upper band ({position_pct:.1f}%) ⚠️"
                        )
                        short_signals += 0.3
                    else:
                        print(f"Bollinger Bands: NEUTRAL (no band interaction) ❌")
            else:
                print(f"Bollinger Bands: NEUTRAL (bands too narrow) ❌")
        else:
            print("DEBUG: Bollinger Bands indicators not available in dataframe")

        # 5. Stochastic Oscillator
        print("\n--- Stochastic Oscillator Analysis ---")
        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            total_indicators += 1

            # Current values
            k_current = latest["stoch_k"]
            d_current = latest["stoch_d"]

            # Previous values
            k_prev = df.iloc[-2]["stoch_k"]
            d_prev = df.iloc[-2]["stoch_d"]

            print(f"DEBUG: K current: {k_current:.2f}, previous: {k_prev:.2f}")
            print(f"DEBUG: D current: {d_current:.2f}, previous: {d_prev:.2f}")

            if k_prev <= d_prev and k_current > d_current and k_current < 20:
                # Bullish crossover in oversold territory
                long_signals += 1
                print(
                    f"Stochastic: BULLISH (K crossed above D in oversold territory) ✅"
                )
            elif k_prev >= d_prev and k_current < d_current and k_current > 80:
                # Bearish crossover in overbought territory
                short_signals += 1
                print(
                    f"Stochastic: BEARISH (K crossed below D in overbought territory) ✅"
                )
            else:
                # Add smaller weight for overbought/oversold conditions
                if k_current < 20:
                    print(f"Stochastic: NEUTRAL but OVERSOLD (K at {k_current:.1f}) ⚠️")
                    long_signals += 0.3
                elif k_current > 80:
                    print(
                        f"Stochastic: NEUTRAL but OVERBOUGHT (K at {k_current:.1f}) ⚠️"
                    )
                    short_signals += 0.3
                else:
                    print(f"Stochastic: NEUTRAL (no significant signal) ❌")
        else:
            print("DEBUG: Stochastic indicators not available in dataframe")

        # 6. ADX for trend strength
        print("\n--- ADX Trend Strength Analysis ---")
        if "adx" in df.columns:
            adx = latest["adx"]
            print(f"DEBUG: ADX value: {adx:.2f}")
            print(f"ADX: {adx:.1f} - {'Strong' if adx > 25 else 'Weak'} trend")

            # ADX doesn't give direction, just confirms strength of other signals
            if adx > 25:
                # Strong trend - boost existing signals
                old_long = long_signals
                old_short = short_signals
                long_signals = (
                    long_signals * 1.2 if long_signals > short_signals else long_signals
                )
                short_signals = (
                    short_signals * 1.2
                    if short_signals > long_signals
                    else short_signals
                )
                print(
                    f"DEBUG: Strong trend detected - boosting signals (Long: {old_long:.1f} → {long_signals:.1f}, Short: {old_short:.1f} → {short_signals:.1f})"
                )
        else:
            print("DEBUG: ADX indicator not available in dataframe")

        # 7. Volume analysis
        print("\n--- Volume Analysis ---")
        if "volume" in df.columns:
            total_indicators += 1

            # Current volume
            current_volume = latest["volume"]

            # Average volume (10 periods)
            avg_volume = df["volume"].iloc[-10:].mean()

            # Volume ratio
            volume_ratio = current_volume / avg_volume

            # Price change
            price_change = latest["close"] - df.iloc[-2]["close"]

            print(
                f"DEBUG: Current volume: {current_volume:.2f}, Avg volume: {avg_volume:.2f}"
            )
            print(f"DEBUG: Volume ratio: {volume_ratio:.2f}x average")
            print(
                f"DEBUG: Price change: {price_change:.6f} ({(price_change/df.iloc[-2]['close']*100):.2f}%)"
            )

            # High volume with price increase - bullish
            if volume_ratio > 1.5 and price_change > 0:
                long_signals += 1
                print(
                    f"Volume: BULLISH (High volume with price increase, ratio: {volume_ratio:.2f}) ✅"
                )
            # High volume with price decrease - bearish
            elif volume_ratio > 1.5 and price_change < 0:
                short_signals += 1
                print(
                    f"Volume: BEARISH (High volume with price decrease, ratio: {volume_ratio:.2f}) ✅"
                )
            else:
                # Add smaller weight for volume patterns
                if volume_ratio > 1.2 and price_change > 0:
                    print(
                        f"Volume: NEUTRAL but increased volume with price rise (ratio: {volume_ratio:.2f}) ⚠️"
                    )
                    long_signals += 0.2
                elif volume_ratio > 1.2 and price_change < 0:
                    print(
                        f"Volume: NEUTRAL but increased volume with price drop (ratio: {volume_ratio:.2f}) ⚠️"
                    )
                    short_signals += 0.2
                else:
                    print(f"Volume: NEUTRAL (no significant volume pattern) ❌")
        else:
            print("DEBUG: Volume data not available in dataframe")

        # 8. Support/Resistance breakouts
        print("\n--- Support/Resistance Analysis ---")
        # This is a simplified version - in a real system you'd have more sophisticated S/R detection
        if len(df) >= 20:
            total_indicators += 1

            # Find recent highs and lows
            recent_high = df["high"].iloc[-20:].max()
            recent_low = df["low"].iloc[-20:].min()

            current_close = latest["close"]
            prev_close = df.iloc[-2]["close"]

            print(
                f"DEBUG: Recent high: {recent_high:.4f}, Recent low: {recent_low:.4f}"
            )
            print(
                f"DEBUG: Current close: {current_close:.4f}, Previous close: {prev_close:.4f}"
            )

            # Calculate distance to high and low as percentage
            distance_to_high_pct = ((recent_high - current_close) / current_close) * 100
            distance_to_low_pct = ((current_close - recent_low) / current_close) * 100

            print(f"DEBUG: Distance to high: {distance_to_high_pct:.2f}%")
            print(f"DEBUG: Distance to low: {distance_to_low_pct:.2f}%")

            # Breakout above resistance
            if prev_close < recent_high and current_close > recent_high:
                long_signals += 1
                print(
                    f"Support/Resistance: BULLISH (Breakout above resistance at {recent_high:.4f}) ✅"
                )
            # Breakdown below support
            elif prev_close > recent_low and current_close < recent_low:
                short_signals += 1
                print(
                    f"Support/Resistance: BEARISH (Breakdown below support at {recent_low:.4f}) ✅"
                )
            else:
                # Add smaller weight for proximity to support/resistance
                if distance_to_high_pct < 1.0:
                    print(
                        f"Support/Resistance: NEUTRAL but near resistance ({distance_to_high_pct:.2f}% from high) ⚠️"
                    )
                    short_signals += 0.2
                elif distance_to_low_pct < 1.0:
                    print(
                        f"Support/Resistance: NEUTRAL but near support ({distance_to_low_pct:.2f}% from low) ⚠️"
                    )
                    long_signals += 0.2
                else:
                    print(f"Support/Resistance: NEUTRAL (no breakout/breakdown) ❌")
        else:
            print("DEBUG: Not enough data for Support/Resistance analysis")

        # Calculate final signal
        print("\n--- Final Signal Calculation ---")
        print(
            f"DEBUG: Long signals: {long_signals:.1f}, Short signals: {short_signals:.1f}, Total indicators: {total_indicators}"
        )
        print(
            f"DEBUG: Signal threshold: {self.signal_confirmation_threshold} indicators required"
        )

        # Require at least the threshold number of confirming indicators
        if (
            long_signals >= self.signal_confirmation_threshold
            and long_signals > short_signals
        ):
            signal = 1  # Long
            confidence = min(100, (long_signals / total_indicators) * 100)
            print(
                f"DEBUG: Long signals ({long_signals:.1f}) meet threshold and exceed short signals ({short_signals:.1f})"
            )
        elif (
            short_signals >= self.signal_confirmation_threshold
            and short_signals > long_signals
        ):
            signal = -1  # Short
            confidence = min(100, (short_signals / total_indicators) * 100)
            print(
                f"DEBUG: Short signals ({short_signals:.1f}) meet threshold and exceed long signals ({long_signals:.1f})"
            )
        else:
            signal = 0  # No clear signal
            confidence = 0
            print(f"DEBUG: Neither long nor short signals meet threshold requirements")

        # Check for signal cooldown period
        if self.last_signal_time is not None:
            time_since_last_signal = (
                datetime.now() - self.last_signal_time
            ).total_seconds() / 60
            cooldown_minutes = self.signal_cooldown_minutes

            # Reduce cooldown time for scalping mode
            if self.use_scalping_mode:
                cooldown_minutes = max(
                    5, self.signal_cooldown_minutes // 2
                )  # Minimum 5 minutes, otherwise half the normal cooldown

            if time_since_last_signal < cooldown_minutes:
                print(
                    f"Signal ignored: Cooldown period active ({time_since_last_signal:.1f} minutes since last signal, need {cooldown_minutes} minutes)"
                )
                return (0, 0)  # Return neutral signal with zero confidence

        # Update last signal time if we have a valid signal
        if signal != 0:
            self.last_signal_time = datetime.now()
            print(f"DEBUG: Updated last signal time to {self.last_signal_time}")

        # Print summary
        print("\n--- Signal Summary ---")
        print(
            f"Enhanced Signal Analysis: Long indicators: {long_signals:.1f}, Short indicators: {short_signals:.1f}, Total: {total_indicators}"
        )
        print(
            f"Final Signal: {'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'} with {confidence:.1f}% confidence"
        )
        print("========== ENHANCED SIGNAL ANALYSIS END ==========\n")

        return (signal, confidence)

    def execute_trade(self, signal, current_price, timestamp):
        """Execute a trade with risk management"""
        # Check if we already have an open position
        if self.has_open_position():
            return None

        # Check if trading is disabled due to daily loss limit
        if self.trading_disabled:
            print(f"Trading is disabled due to reaching daily loss limit")
            return None

        # Check if we have enough balance
        account_balance = self.get_account_balance()

        # For compound interest, check symbol balance instead
        if self.compound_interest:
            if self.symbol_balance < 10:  # Minimum balance check
                print(f"Symbol balance too low for trading: ${self.symbol_balance:.2f}")
                return None
        elif account_balance < 10:  # Minimum balance check
            print(f"Account balance too low for trading: ${account_balance:.2f}")
            return None

        # Print debug info for compound interest
        if self.compound_interest:
            print(
                f"Using compound interest. Symbol balance: ${self.symbol_balance:.2f}"
            )
            print(f"Initial investment: ${self.initial_investment:.2f}")
            print(f"Total symbol profit: ${self.total_symbol_profit:.2f}")
            print(
                f"Compound interest: {self.total_symbol_profit/self.initial_investment*100:.2f}%"
            )

        # If enhanced signals are enabled, verify the signal with multiple indicators
        if self.use_enhanced_signals:
            # Get latest data for signal analysis
            latest_df = self.get_latest_data(lookback_candles=50)

            if latest_df is not None and len(latest_df) >= 50:
                enhanced_signal, confidence = self.generate_enhanced_signal(latest_df)

                # Only proceed if enhanced signal confirms the original signal
                if enhanced_signal != signal:
                    print(
                        f"Signal rejected: Enhanced signal ({enhanced_signal}) does not match original signal ({signal})"
                    )
                    return None
                elif confidence < 60:  # Require at least 60% confidence
                    print(f"Signal rejected: Confidence too low ({confidence:.1f}%)")
                    return None
                else:
                    print(f"Signal confirmed with {confidence:.1f}% confidence")
            else:
                print("Warning: Not enough data for enhanced signal analysis")

        # If trend following mode is enabled, analyze market trend
        if self.trend_following_mode:
            trend_direction, trend_strength = self.analyze_market_trend()

            # Only trade in the direction of the trend if it's strong enough
            if trend_strength > 50:
                if (signal == 1 and trend_direction != 1) or (
                    signal == -1 and trend_direction != -1
                ):
                    print(
                        f"Signal {signal} ignored: against current market trend (direction: {trend_direction}, strength: {trend_strength:.1f}%)"
                    )
                    return None
                else:
                    print(
                        f"Signal {signal} aligned with market trend (direction: {trend_direction}, strength: {trend_strength:.1f}%)"
                    )
            else:
                print(
                    f"Weak market trend detected (strength: {trend_strength:.1f}%), proceeding with signal {signal}"
                )

        # Determine position type based on signal
        position = "long" if signal == 1 else "short"

        # Calculate position size using dynamic sizing for better risk management
        position_size = self.calculate_dynamic_position_size(current_price)

        # Calculate stop loss and take profit prices
        if position == "long":
            # For long positions: stop loss below entry, take profit above entry
            stop_loss_price = current_price * (1 - self.stop_loss_pct)

            # Use dynamic take profit if enabled
            if self.use_dynamic_take_profit:
                take_profit_price = self.calculate_dynamic_take_profit(current_price)
            else:
                take_profit_price = current_price * (1 + self.take_profit_pct)
        else:
            # For short positions: stop loss above entry, take profit below entry
            stop_loss_price = current_price * (1 + self.stop_loss_pct)

            # Use dynamic take profit if enabled
            if self.use_dynamic_take_profit:
                take_profit_price = self.calculate_dynamic_take_profit(current_price)
            else:
                take_profit_price = current_price * (1 - self.take_profit_pct)

        # Don't trade if trading is disabled due to losses
        current_day = datetime.now().date()
        if current_day != self.last_reset_day:
            # Reset daily tracking
            self.daily_loss = 0
            self.trading_disabled = False
            self.last_reset_day = current_day

        # Don't trade if trading is disabled
        if self.trading_disabled:
            print("Trading disabled due to reaching maximum daily loss")
            return None

        # Don't trade if there's already an open position
        if self.has_open_position():
            print("Already have an open position, skipping trade")
            return None

        # Reset partial take profit tracking for new position
        self.partial_tp_executed = False
        self.original_position_size = 0

        # Don't trade if balance is too low
        account_balance = self.get_account_balance()
        print(f"Current account balance: ${account_balance:.2f}")
        print(f"Initial investment: ${self.initial_investment:.2f}")
        print(f"Compound interest enabled: {self.compound_interest}")
        print(f"Symbol balance for {self.symbol}: ${self.symbol_balance:.2f}")

        if self.symbol_balance < (self.initial_investment * 0.5):
            print(
                f"Symbol balance too low (${self.symbol_balance:.2f}), trading paused"
            )
            return None

        # Get the correct precision for the quantity and price
        quantity_precision = self.get_quantity_precision()
        price_precision = self.get_price_precision()
        min_quantity = self.get_min_quantity()
        min_notional = self.get_min_notional()

        # Calculate position size based on full investment, full margin, or risk management
        if self.use_full_margin:
            # Always use initial investment for full margin mode, regardless of compound interest setting
            investment_amount = self.initial_investment
            # Calculate position size to use full investment as margin
            # Formula: position_size = (investment * leverage) / current_price
            position_size = (investment_amount * self.leverage) / current_price
            print(f"Using FULL MARGIN mode: {investment_amount:.2f} USD as margin")
            print(
                f"Controlling position worth: {investment_amount * self.leverage:.2f} USD"
            )
        elif self.use_full_investment:
            # Use almost full balance for position size (95% to leave room for fees)
            # If compound interest is enabled, use symbol balance, otherwise use initial investment
            if self.compound_interest:
                investment_amount = self.symbol_balance * 0.95
                print(
                    f"Using full investment mode with compound interest: {investment_amount:.2f} USD for position (based on symbol balance)"
                )
            else:
                investment_amount = self.initial_investment * 0.95
                print(
                    f"Using full investment mode: {investment_amount:.2f} USD for position"
                )
            position_size = investment_amount / current_price
        else:
            # Calculate maximum position size based on current balance
            max_position_value = (
                self.symbol_balance * self.max_position_size
                if self.compound_interest
                else account_balance * self.max_position_size
            )

            # Calculate position size based on risk per trade
            # If compound interest is enabled, calculate risk amount based on symbol balance
            # Otherwise, use initial investment
            if self.compound_interest:
                risk_amount = self.symbol_balance * self.risk_per_trade
                print(
                    f"Using compound interest: Risk amount ${risk_amount:.2f} based on symbol balance ${self.symbol_balance:.2f}"
                )
            else:
                risk_amount = self.initial_investment * self.risk_per_trade
                print(
                    f"Using fixed investment: Risk amount ${risk_amount:.2f} based on initial investment ${self.initial_investment:.2f}"
                )

            # Calculate initial position size based on risk
            position_size = (risk_amount / self.stop_loss_pct) / current_price

            # Ensure position size doesn't exceed maximum allowed
            position_value = position_size * current_price
            if position_value > max_position_value:
                position_size = max_position_value / current_price

        # Get ATR for dynamic stop loss and take profit
        latest_df = self.get_latest_data(lookback_candles=20)
        atr = latest_df["ATR"].iloc[-1]
        print(f"ATR: {atr:.4f}")

        if signal == 1:  # BUY signal
            # Use ATR for stop loss (1.0x ATR instead of 1.5x)
            self.stop_loss_pct = min(
                0.01, (1.0 * atr) / current_price
            )  # Cap at 1% instead of 3%
            # Use ATR for take profit (2x ATR instead of 3x)
            self.take_profit_pct = min(
                0.06, (2 * atr) / current_price
            )  # Cap at 6% instead of 10%

            # Apply scalping mode adjustments if enabled
            if self.use_scalping_mode:
                # Reduce take profit for faster exits
                self.take_profit_pct = min(
                    0.02, (self.scalping_tp_factor * atr) / current_price
                )  # Cap at 2%
                # Adjust stop loss for scalping
                self.stop_loss_pct = min(
                    0.01, (self.scalping_sl_factor * atr) / current_price
                )  # Cap at 1%
                print(
                    f"Scalping mode enabled: Using smaller take profit ({self.take_profit_pct*100:.4f}%)"
                )

            # Recalculate position size if using risk-based sizing
            if not self.use_full_margin and not self.use_full_investment:
                position_size = (risk_amount / self.stop_loss_pct) / current_price

            # Override with fixed values if provided (in USDT)
            if self.fixed_tp > 0:
                # Calculate take profit price directly from fixed dollar amount
                fixed_tp_price = current_price + (self.fixed_tp / position_size)
                # Calculate percentage equivalent
                self.take_profit_pct = (fixed_tp_price - current_price) / current_price
                print(
                    f"Using fixed take profit: ${self.fixed_tp:.2f} ({self.take_profit_pct*100:.4f}%)"
                )

            if self.fixed_sl > 0:
                # Calculate stop loss price directly from fixed dollar amount
                fixed_sl_price = current_price - (self.fixed_sl / position_size)
                # Calculate percentage equivalent
                self.stop_loss_pct = (current_price - fixed_sl_price) / current_price
                print(
                    f"Using fixed stop loss: ${self.fixed_sl:.2f} ({self.stop_loss_pct*100:.4f}%)"
                )

            # Calculate stop loss and take profit prices
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
            self.take_profit_price = current_price * (1 + self.take_profit_pct)

            # Debug logging for stop loss and take profit calculation
            print(
                f"DEBUG - Calculated stop loss: ${self.stop_loss_price:.2f} ({self.stop_loss_pct*100:.2f}% above entry)"
            )
            print(
                f"DEBUG - Calculated take profit: ${self.take_profit_price:.2f} ({self.take_profit_pct*100:.2f}% below entry)"
            )
        elif signal == -1:  # SELL signal
            # Use ATR for stop loss (1.0x ATR instead of 1.5x)
            self.stop_loss_pct = min(
                0.01, (1.0 * atr) / current_price
            )  # Cap at 1% instead of 3%
            # Use ATR for take profit (2x ATR instead of 3x)
            self.take_profit_pct = min(
                0.06, (2 * atr) / current_price
            )  # Cap at 6% instead of 10%

            # Apply scalping mode adjustments if enabled
            if self.use_scalping_mode:
                # Reduce take profit for faster exits
                self.take_profit_pct = min(
                    0.02, (self.scalping_tp_factor * atr) / current_price
                )  # Cap at 2%
                # Adjust stop loss for scalping
                self.stop_loss_pct = min(
                    0.01, (self.scalping_sl_factor * atr) / current_price
                )  # Cap at 1%
                print(
                    f"Scalping mode enabled: Using smaller take profit ({self.take_profit_pct*100:.2f}%)"
                )

            # Recalculate position size if using risk-based sizing
            if not self.use_full_margin and not self.use_full_investment:
                position_size = (risk_amount / self.stop_loss_pct) / current_price

            # Override with fixed values if provided (in USDT)
            if self.fixed_tp > 0:
                # Calculate take profit price directly from fixed dollar amount
                fixed_tp_price = current_price + (self.fixed_tp / position_size)
                # Calculate percentage equivalent
                self.take_profit_pct = (fixed_tp_price - current_price) / current_price
                print(
                    f"Using fixed take profit: ${self.fixed_tp:.2f} ({self.take_profit_pct*100:.4f}%)"
                )

            if self.fixed_sl > 0:
                # Calculate stop loss price directly from fixed dollar amount
                fixed_sl_price = current_price - (self.fixed_sl / position_size)
                # Calculate percentage equivalent
                self.stop_loss_pct = (current_price - fixed_sl_price) / current_price
                print(
                    f"Using fixed stop loss: ${self.fixed_sl:.2f} ({self.stop_loss_pct*100:.4f}%)"
                )

            # Calculate stop loss and take profit prices
            self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
            self.take_profit_price = current_price * (1 - self.take_profit_pct)

            # Debug logging for stop loss and take profit calculation
            print(
                f"DEBUG - Calculated stop loss: ${self.stop_loss_price:.2f} ({self.stop_loss_pct*100:.2f}% above entry)"
            )
            print(
                f"DEBUG - Calculated take profit: ${self.take_profit_price:.2f} ({self.take_profit_pct*100:.2f}% below entry)"
            )
        # Format the position size with the correct precision using the direct method
        position_size = float("{:0.0{}f}".format(position_size, quantity_precision))

        # Ensure position size meets minimum quantity requirement
        if position_size < min_quantity:
            print(
                f"Warning: Position size {position_size} is below minimum quantity {min_quantity}"
            )
            position_size = min_quantity

        # Ensure order value meets minimum notional requirement
        order_value = position_size * current_price
        if order_value < min_notional:
            print(
                f"Warning: Order value ${order_value:.2f} is below minimum notional ${min_notional}"
            )
            # Calculate the minimum position size needed to meet the minimum notional value
            min_position_size = min_notional / current_price
            # Format to the correct precision
            min_position_size = float(
                "{:0.0{}f}".format(min_position_size, quantity_precision)
            )
            # Ensure it's at least the minimum quantity
            min_position_size = max(min_position_size, min_quantity)

            print(
                f"Adjusting position size from {position_size} to {min_position_size} to meet minimum requirements"
            )
            position_size = min_position_size

        # Final check to ensure position size is greater than zero
        if position_size <= 0:
            print("Error: Position size is zero or negative. Cannot execute trade.")
            return None

        # Execute the trade
        try:
            if signal == 1:  # BUY signal
                if not self.test_mode:
                    # For futures trading - open long position at MARKET price (executes immediately at current market price)
                    print(
                        f"Opening LONG position at MARKET price for {position_size} {self.symbol}"
                    )
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="BUY",
                        type="MARKET",  # MARKET order type ensures execution at current market price
                        quantity=position_size,
                    )
                    print(f"Order executed: {order}")
                else:
                    # Test mode - simulate order
                    print(
                        f"TEST MODE: Simulating BUY order for {position_size} {self.symbol} at MARKET price (${current_price:.{price_precision}f})"
                    )

                # Update position tracking
                self.position = "long"
                self.entry_price = current_price
                self.position_size = position_size
                self.entry_time = timestamp

                # Set stop loss and take profit prices
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.take_profit_price = current_price * (1 + self.take_profit_pct)

                # Place actual stop loss and take profit orders in Binance
                if not self.test_mode:
                    try:
                        # Format prices with correct precision
                        price_precision = self.get_price_precision()
                        stop_loss_price = float(
                            "{:0.0{}f}".format(self.stop_loss_price, price_precision)
                        )
                        take_profit_price = float(
                            "{:0.0{}f}".format(self.take_profit_price, price_precision)
                        )

                        # For LONG positions: SELL to close
                        # For SHORT positions: BUY to close
                        close_side = "SELL" if signal == 1 else "BUY"

                        # Place stop loss order
                        stop_loss_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type="STOP_MARKET",
                            stopPrice=stop_loss_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC",  # Good Till Cancelled
                        )
                        self.stop_loss_order_id = stop_loss_order["orderId"]
                        print(f"Stop loss order placed: {stop_loss_order}")

                        # Place take profit order
                        take_profit_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type="TAKE_PROFIT_MARKET",
                            stopPrice=take_profit_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC",  # Good Till Cancelled
                        )
                        self.take_profit_order_id = take_profit_order["orderId"]
                        print(f"Take profit order placed: {take_profit_order}")
                    except BinanceAPIException as order_error:
                        print(
                            f"Error placing stop loss or take profit orders: {order_error}"
                        )
                        # Continue with the trade even if setting the orders fails

                # Add to trade history
                trade_record = {
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": current_price,
                    "size": position_size,
                    "value": position_size * current_price,
                    "stop_loss": self.stop_loss_price,
                    "take_profit": self.take_profit_price,
                    "order_type": "MARKET",  # Record that this was a market order
                }

                self.trade_history.append(trade_record)
                if self.test_mode:
                    self.test_trades.append(trade_record)

                return f"BUY: Opened long position at MARKET price (${current_price:.{price_precision}f}) with {position_size} units"

            elif signal == -1:  # SELL signal
                if not self.test_mode:
                    # For futures trading - open short position at MARKET price (executes immediately at current market price)
                    print(
                        f"Opening SHORT position at MARKET price for {position_size} {self.symbol}"
                    )
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="SELL",
                        type="MARKET",  # MARKET order type ensures execution at current market price
                        quantity=position_size,
                    )
                    print(f"Order executed: {order}")
                else:
                    # Test mode - simulate order
                    print(
                        f"TEST MODE: Simulating SELL order for {position_size} {self.symbol} at MARKET price (${current_price:.{price_precision}f})"
                    )

                # Update position tracking
                self.position = "short"
                self.entry_price = current_price
                self.position_size = position_size
                self.entry_time = timestamp

                # Set stop loss and take profit prices
                self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
                self.take_profit_price = current_price * (1 - self.take_profit_pct)

                # Place actual stop loss and take profit orders in Binance
                if not self.test_mode:
                    try:
                        # Format prices with correct precision
                        price_precision = self.get_price_precision()
                        stop_loss_price = float(
                            "{:0.0{}f}".format(self.stop_loss_price, price_precision)
                        )
                        take_profit_price = float(
                            "{:0.0{}f}".format(self.take_profit_price, price_precision)
                        )

                        # For LONG positions: SELL to close
                        # For SHORT positions: BUY to close
                        close_side = "SELL" if signal == 1 else "BUY"

                        # Place stop loss order
                        stop_loss_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type="STOP_MARKET",
                            stopPrice=stop_loss_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC",  # Good Till Cancelled
                        )
                        self.stop_loss_order_id = stop_loss_order["orderId"]
                        print(f"Stop loss order placed: {stop_loss_order}")

                        # Place take profit order
                        take_profit_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type="TAKE_PROFIT_MARKET",
                            stopPrice=take_profit_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC",  # Good Till Cancelled
                        )
                        self.take_profit_order_id = take_profit_order["orderId"]
                        print(f"Take profit order placed: {take_profit_order}")
                    except BinanceAPIException as order_error:
                        print(
                            f"Error placing stop loss or take profit orders: {order_error}"
                        )
                        # Continue with the trade even if setting the orders fails

                # Add to trade history
                trade_record = {
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": current_price,
                    "size": position_size,
                    "value": position_size * current_price,
                    "stop_loss": self.stop_loss_price,
                    "take_profit": self.take_profit_price,
                    "order_type": "MARKET",  # Record that this was a market order
                }

                self.trade_history.append(trade_record)
                if self.test_mode:
                    self.test_trades.append(trade_record)

                return f"SELL: Opened short position at MARKET price (${current_price:.{price_precision}f}) with {position_size} units"

        except BinanceAPIException as e:
            error_message = str(e)
            print(f"Error executing trade: {error_message}")

            # Handle precision errors specifically
            if "Precision is over the maximum defined for this asset" in error_message:
                print("Precision error detected. Trying to adjust position size...")

                # Try to get futures precision directly
                futures_info = self.get_futures_symbol_info()
                if futures_info and "quantityPrecision" in futures_info:
                    adjusted_precision = futures_info["quantityPrecision"]
                    adjusted_position_size = float(
                        "{:0.0{}f}".format(position_size, adjusted_precision)
                    )
                    print(
                        f"Adjusted position size from {position_size} to {adjusted_position_size} using futures precision"
                    )

                    # Ensure adjusted position size is greater than zero
                    if adjusted_position_size <= 0:
                        adjusted_position_size = float(
                            "{:0.0{}f}".format(min_quantity, adjusted_precision)
                        )
                        print(
                            f"Adjusted position size was zero, setting to minimum quantity: {adjusted_position_size}"
                        )

                    # Try again with the adjusted position size
                    if not self.test_mode:
                        try:
                            side = "BUY" if signal == 1 else "SELL"
                            print(
                                f"Retrying with adjusted position size: {adjusted_position_size}"
                            )
                            order = self.client.futures_create_order(
                                symbol=self.symbol,
                                side=side,
                                type="MARKET",
                                quantity=adjusted_position_size,
                            )
                            print(f"Order executed: {order}")

                            # Update position tracking
                            self.position = "long" if signal == 1 else "short"
                            self.entry_price = current_price
                            self.position_size = adjusted_position_size
                            self.entry_time = timestamp

                            # Set stop loss and take profit prices
                            self.stop_loss_price = (
                                current_price * (1 - self.stop_loss_pct)
                                if signal == 1
                                else current_price * (1 + self.stop_loss_pct)
                            )
                            self.take_profit_price = (
                                current_price * (1 + self.take_profit_pct)
                                if signal == 1
                                else current_price * (1 - self.take_profit_pct)
                            )

                            # Place actual stop loss and take profit orders in Binance
                            if not self.test_mode:
                                try:
                                    # Format prices with correct precision
                                    price_precision = self.get_price_precision()
                                    stop_loss_price = float(
                                        "{:0.0{}f}".format(
                                            self.stop_loss_price, price_precision
                                        )
                                    )
                                    take_profit_price = float(
                                        "{:0.0{}f}".format(
                                            self.take_profit_price, price_precision
                                        )
                                    )

                                    # For LONG positions: SELL to close
                                    # For SHORT positions: BUY to close
                                    close_side = "SELL" if signal == 1 else "BUY"

                                    # Place stop loss order
                                    stop_loss_order = self.client.futures_create_order(
                                        symbol=self.symbol,
                                        side=close_side,
                                        type="STOP_MARKET",
                                        stopPrice=stop_loss_price,
                                        reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                        quantity=adjusted_position_size,
                                        workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                        timeInForce="GTC",  # Good Till Cancelled
                                    )
                                    self.stop_loss_order_id = stop_loss_order["orderId"]
                                    print(f"Stop loss order placed: {stop_loss_order}")

                                    # Place take profit order
                                    take_profit_order = self.client.futures_create_order(
                                        symbol=self.symbol,
                                        side=close_side,
                                        type="TAKE_PROFIT_MARKET",
                                        stopPrice=take_profit_price,
                                        reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                        quantity=adjusted_position_size,
                                        workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                        timeInForce="GTC",  # Good Till Cancelled
                                    )
                                    self.take_profit_order_id = take_profit_order[
                                        "orderId"
                                    ]
                                    print(
                                        f"Take profit order placed: {take_profit_order}"
                                    )
                                except BinanceAPIException as order_error:
                                    print(
                                        f"Error placing stop loss or take profit orders: {order_error}"
                                    )
                                    # Continue with the trade even if setting the orders fails

                            # Add to trade history
                            trade_record = {
                                "timestamp": timestamp,
                                "action": "BUY" if signal == 1 else "SELL",
                                "price": current_price,
                                "size": adjusted_position_size,
                                "value": adjusted_position_size * current_price,
                                "stop_loss": self.stop_loss_price,
                                "take_profit": self.take_profit_price,
                                "order_type": "MARKET",
                            }

                            self.trade_history.append(trade_record)

                            action = "BUY" if signal == 1 else "SELL"
                            position_type = "long" if signal == 1 else "short"
                            return f"{action}: Opened {position_type} position at MARKET price (${current_price:.{price_precision}f}) with {adjusted_position_size} units"
                        except BinanceAPIException as retry_error:
                            print(f"Error on retry: {retry_error}")
                            return None

                # If in test mode, provide guidance
                if self.test_mode:
                    return f"TEST MODE: Precision error. Try increasing your investment amount or using a different trading pair."

            # Handle quantity errors specifically
            elif "Quantity less than or equal to zero" in error_message:
                print(
                    "Error: Position size is zero or negative. Trying with minimum quantity..."
                )

                # Try with minimum quantity
                min_valid_quantity = min_quantity
                # Ensure it meets minimum notional requirement
                if min_valid_quantity * current_price < min_notional:
                    min_valid_quantity = min_notional / current_price

                print(f"Retrying with minimum valid quantity: {min_valid_quantity}")

                if not self.test_mode:
                    try:
                        side = "BUY" if signal == 1 else "SELL"
                        order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=side,
                            type="MARKET",
                            quantity=min_valid_quantity,
                        )
                        print(f"Order executed: {order}")

                        # Update position tracking
                        self.position = "long" if signal == 1 else "short"
                        self.entry_price = current_price
                        self.position_size = min_valid_quantity
                        self.entry_time = timestamp

                        # Set stop loss and take profit prices
                        self.stop_loss_price = (
                            current_price * (1 - self.stop_loss_pct)
                            if signal == 1
                            else current_price * (1 + self.stop_loss_pct)
                        )
                        self.take_profit_price = (
                            current_price * (1 + self.take_profit_pct)
                            if signal == 1
                            else current_price * (1 - self.take_profit_pct)
                        )

                        # Place actual stop loss and take profit orders in Binance
                        if not self.test_mode:
                            try:
                                # Format prices with correct precision
                                price_precision = self.get_price_precision()
                                stop_loss_price = float(
                                    "{:0.0{}f}".format(
                                        self.stop_loss_price, price_precision
                                    )
                                )
                                take_profit_price = float(
                                    "{:0.0{}f}".format(
                                        self.take_profit_price, price_precision
                                    )
                                )

                                # For LONG positions: SELL to close
                                # For SHORT positions: BUY to close
                                close_side = "SELL" if signal == 1 else "BUY"

                                # Place stop loss order
                                stop_loss_order = self.client.futures_create_order(
                                    symbol=self.symbol,
                                    side=close_side,
                                    type="STOP_MARKET",
                                    stopPrice=stop_loss_price,
                                    reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                    quantity=min_valid_quantity,
                                    workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                    timeInForce="GTC",  # Good Till Cancelled
                                )
                                self.stop_loss_order_id = stop_loss_order["orderId"]
                                print(f"Stop loss order placed: {stop_loss_order}")

                                # Place take profit order
                                take_profit_order = self.client.futures_create_order(
                                    symbol=self.symbol,
                                    side=close_side,
                                    type="TAKE_PROFIT_MARKET",
                                    stopPrice=take_profit_price,
                                    reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                    quantity=min_valid_quantity,
                                    workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                    timeInForce="GTC",  # Good Till Cancelled
                                )
                                self.take_profit_order_id = take_profit_order["orderId"]
                                print(f"Take profit order placed: {take_profit_order}")
                            except BinanceAPIException as order_error:
                                print(
                                    f"Error placing stop loss or take profit orders: {order_error}"
                                )
                                # Continue with the trade even if setting the orders fails

                        # Add to trade history
                        trade_record = {
                            "timestamp": timestamp,
                            "action": "BUY" if signal == 1 else "SELL",
                            "price": current_price,
                            "size": min_valid_quantity,
                            "value": min_valid_quantity * current_price,
                            "stop_loss": self.stop_loss_price,
                            "take_profit": self.take_profit_price,
                            "order_type": "MARKET",
                        }

                        self.trade_history.append(trade_record)

                        action = "BUY" if signal == 1 else "SELL"
                        position_type = "long" if signal == 1 else "short"
                        return f"{action}: Opened {position_type} position at MARKET price (${current_price:.{price_precision}f}) with {min_valid_quantity} units"
                    except BinanceAPIException as retry_error:
                        print(f"Error on retry: {retry_error}")
                        return None

            return None

    def execute_pyramid_entry(self, current_price, timestamp):
        """Add to a winning position using pyramiding strategy"""
        if not self.has_open_position() or not self.enable_pyramiding:
            return

        try:
            # Calculate position size for pyramid entry (smaller than initial entry)
            # Use 50% of the size that would be calculated for a new position
            pyramid_size_factor = 0.5
            base_position_size = self.calculate_position_size(current_price)
            pyramid_position_size = base_position_size * pyramid_size_factor

            # Get the correct precision for the quantity
            quantity_precision = self.get_quantity_precision()

            # Format the position size with the correct precision
            pyramid_position_size = float(
                "{:0.0{}f}".format(pyramid_position_size, quantity_precision)
            )

            # Ensure pyramid size is greater than zero
            if pyramid_position_size <= 0:
                print("Error: Pyramid position size is zero or negative.")
                return

            # Execute the pyramid entry
            if not self.test_mode:
                # For LONG positions: BUY to add
                # For SHORT positions: SELL to add
                side = "BUY" if self.position == "long" else "SELL"
                print(
                    f"Executing pyramid entry: {side} {pyramid_position_size} {self.symbol} at MARKET price (${current_price:.2f})"
                )

                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type="MARKET",
                    quantity=pyramid_position_size,
                )
                print(f"Pyramid entry order executed: {order}")
            else:
                # Test mode - simulate order
                side = "BUY" if self.position == "long" else "SELL"
                print(
                    f"TEST MODE: Simulating pyramid entry: {side} {pyramid_position_size} {self.symbol} at MARKET price (${current_price:.2f})"
                )

            # Store pyramid entry details
            self.pyramid_entries += 1
            self.pyramid_entry_prices.append(current_price)
            self.pyramid_position_sizes.append(pyramid_position_size)

            # Update average entry price and total position size
            total_position_value = (self.entry_price * self.position_size) + (
                current_price * pyramid_position_size
            )
            new_position_size = self.position_size + pyramid_position_size
            new_entry_price = total_position_value / new_position_size

            # Update position tracking
            old_entry_price = self.entry_price
            old_position_size = self.position_size
            self.entry_price = new_entry_price
            self.position_size = new_position_size

            # Update stop loss order with new position size
            if not self.test_mode and self.stop_loss_order_id:
                self.update_stop_loss_order()

            # Add to trade history
            pyramid_entry_record = {
                "timestamp": timestamp,
                "action": "PYRAMID_ENTRY",
                "price": current_price,
                "size": pyramid_position_size,
                "value": pyramid_position_size * current_price,
                "old_entry_price": old_entry_price,
                "new_entry_price": new_entry_price,
                "old_position_size": old_position_size,
                "new_position_size": new_position_size,
                "pyramid_count": self.pyramid_entries,
            }

            self.trade_history.append(pyramid_entry_record)
            if self.test_mode:
                self.test_trades.append(pyramid_entry_record)

            # Send notification
            message = (
                f"🔺 Pyramid Entry #{self.pyramid_entries}\n"
                f"Symbol: {self.symbol}\n"
                f"Type: {self.position.upper()}\n"
                f"Entry Price: ${current_price:.4f}\n"
                f"Added Size: {pyramid_position_size} units\n"
                f"New Position Size: {new_position_size} units\n"
                f"New Average Entry: ${new_entry_price:.4f}\n"
                f"Account Balance: ${self.get_account_balance():.2f}\n"
                f"Symbol Balance: ${self.symbol_balance:.2f}"
            )
            self.send_notification(message)

            print(
                f"✅ Pyramid entry #{self.pyramid_entries} executed successfully. New position size: {new_position_size} units with average entry: ${new_entry_price:.2f}"
            )

        except BinanceAPIException as e:
            print(f"❌ Error executing pyramid entry: {e}")

    def close_position(self, current_price, timestamp, reason="manual"):
        """Close the current position"""
        if not self.has_open_position():
            return None

        # Calculate profit/loss
        if self.position == "long":
            profit = self.position_size * (current_price - self.entry_price)
            profit_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # short
            profit = self.position_size * (self.entry_price - current_price)
            profit_pct = (self.entry_price - current_price) / self.entry_price * 100

        # Update test balance in test mode
        if self.test_mode:
            self.test_balance += profit
            print(
                f"TEST MODE: Updated balance to ${self.test_balance:.2f} (profit: ${profit:.2f})"
            )

        # Update per-symbol balance for compound interest
        self.symbol_balance += profit
        self.total_symbol_profit += profit

        # Execute the close in Binance if not in test mode
        if not self.test_mode:
            try:
                # Cancel any existing orders first
                self.client.futures_cancel_all_open_orders(symbol=self.symbol)
                print(f"Cancelled all open orders for {self.symbol}")

                # For LONG positions: SELL to close
                # For SHORT positions: BUY to close
                close_side = "SELL" if self.position == "long" else "BUY"

                # Execute market order to close position
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=close_side,
                    type="MARKET",
                    quantity=self.position_size,
                    reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                )
                print(f"Position closed with order: {order}")
            except BinanceAPIException as e:
                print(f"Error closing position: {e}")

        # Add to trade history
        close_record = {
            "timestamp": timestamp,
            "action": "CLOSE",
            "price": current_price,
            "size": self.position_size,
            "value": self.position_size * current_price,
            "profit": profit,
            "profit_pct": profit_pct,
            "reason": reason,
            "order_type": "MARKET",
            "pyramid_entries": self.pyramid_entries,
        }

        self.trade_history.append(close_record)
        if self.test_mode:
            self.test_trades.append(close_record)

        # Update daily profits
        day_key = timestamp.strftime("%Y-%m-%d")
        if day_key not in self.daily_profits:
            self.daily_profits[day_key] = 0
        self.daily_profits[day_key] += profit

        # Calculate total profit including any partial take profit
        total_profit = profit
        partial_profit = 0
        partial_profit_message = ""

        # If we executed a partial take profit earlier, include that profit in the notification
        if self.partial_tp_executed and self.original_position_size > 0:
            # Calculate the size of the partial position that was closed
            partial_size = self.original_position_size * self.partial_tp_size

            # Find the partial take profit record in trade history
            for trade in reversed(self.trade_history):
                if trade.get("reason") == "PARTIAL_TAKE_PROFIT":
                    partial_profit = trade.get("profit", 0)
                    partial_price = trade.get("price", 0)
                    break

            if partial_profit > 0:
                total_profit += partial_profit
                partial_profit_message = f"\nPartial TP Profit: ${partial_profit:.2f} ({self.partial_tp_size*100:.0f}% of position)"

        # Send notification
        message = (
            f"{'💰' if profit > 0 else '🛑'} POSITION CLOSED ({self.position.upper()})\n"
            f"Symbol: {self.symbol}\n"
            f"Reason: {reason}\n"
            f"Entry: ${self.entry_price:.2f}\n"
            f"Exit: ${current_price:.2f}\n"
            f"Position Size: {self.position_size:.6f} units"
        )

        # Add pyramid entries info if applicable
        if self.pyramid_entries > 0:
            message += f"\nPyramid Entries: {self.pyramid_entries}"

        # Add partial take profit info if applicable
        if self.partial_tp_executed and self.original_position_size > 0:
            message += f"\nOriginal Size: {self.original_position_size:.6f} units"
            message += partial_profit_message
            message += f"\nRemaining Profit: ${profit:.2f}"
            message += f"\nTotal Profit: ${total_profit:.2f} ({total_profit/self.entry_price/self.original_position_size*100:.2f}%)"
        else:
            message += f"\nProfit: ${profit:.2f} ({profit_pct:.2f}%)"

        message += f"\nAccount Balance: ${self.get_account_balance():.2f}\nSymbol Balance: ${self.symbol_balance:.2f}"

        self.send_notification(message)

        # Reset position tracking
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.entry_time = None
        self.stop_loss_order_id = None
        self.take_profit_order_id = None
        self.partial_tp_executed = False
        self.original_position_size = 0
        self.pyramid_entries = 0
        self.pyramid_entry_prices = []
        self.pyramid_position_sizes = []

        # Save results
        self.save_trading_results()

        return {
            "symbol": self.symbol,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": current_price,
            "profit": profit,
            "profit_pct": profit_pct,
            "reason": reason,
            "timestamp": timestamp,
            "partial_tp_executed": self.partial_tp_executed,
            "total_profit": total_profit if self.partial_tp_executed else profit,
            "pyramid_entries": self.pyramid_entries,
        }

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
                os.path.join(
                    self.results_dir, f"{mode_prefix}{self.symbol}_trade_history.csv"
                ),
                index=False,
            )

        # Save daily profits
        if self.daily_profits:
            daily_profits_list = [
                {"date": date, "profit": profit}
                for date, profit in self.daily_profits.items()
            ]
            daily_profits_df = pd.DataFrame(daily_profits_list)
            mode_prefix = "test_" if self.test_mode else ""
            daily_profits_df.to_csv(
                os.path.join(
                    self.results_dir, f"{mode_prefix}{self.symbol}_daily_profits.csv"
                ),
                index=False,
            )

        # Calculate days that met target
        days_met_target = sum(
            1
            for profit in self.daily_profits.values()
            if profit >= self.daily_profit_target
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
            "daily_profit_target": self.daily_profit_target,
            "days_met_target": days_met_target,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_mode": self.test_mode,
        }

        status_df = pd.DataFrame([status])
        mode_prefix = "test_" if self.test_mode else ""
        status_df.to_csv(
            os.path.join(self.results_dir, f"{mode_prefix}{self.symbol}_status.csv"),
            index=False,
        )

    def run_real_trading(
        self, duration_hours=24, update_interval_minutes=15, update_interval_seconds=0
    ):
        """
        Run real-time trading

        Args:
            duration_hours: How long to run the trading in hours
            update_interval_minutes: How often to update in minutes
            update_interval_seconds: How often to update in seconds (overrides minutes if set)
        """
        return run_real_trading(
            self, duration_hours, update_interval_minutes, update_interval_seconds
        )

    def update_stop_loss_order(self):
        """Update the stop loss order in Binance with the new stop loss price"""
        if (
            self.test_mode
            or not self.has_open_position()
            or not self.stop_loss_order_id
            or self.stop_loss_price is None
        ):
            if self.stop_loss_price is None:
                print("Cannot update stop loss order: stop_loss_price is None")
            elif not self.stop_loss_order_id:
                print(
                    "Cannot update stop loss order: stop_loss_order_id is None, will recreate instead"
                )
                self.recreate_stop_loss_order()
            elif not self.has_open_position():
                print("Cannot update stop loss order: no open position")
            elif self.test_mode:
                print("Cannot update stop loss order: in test mode")
            return

        try:
            print(
                f"DEBUG - Updating stop loss order: ID={self.stop_loss_order_id}, Current price={self.stop_loss_price}"
            )

            # Cancel the existing stop loss order
            try:
                cancel_result = self.client.futures_cancel_order(
                    symbol=self.symbol, orderId=self.stop_loss_order_id
                )
                print(
                    f"Cancelled existing stop loss order: {self.stop_loss_order_id}, Result: {cancel_result}"
                )
            except BinanceAPIException as cancel_error:
                print(f"❌ Error cancelling stop loss order: {cancel_error}")
                if "Unknown order" in str(cancel_error):
                    print(
                        f"Order {self.stop_loss_order_id} already cancelled or does not exist. Will create new order."
                    )
                    self.stop_loss_order_id = None
                    self.recreate_stop_loss_order()
                    return
                else:
                    # Re-raise for other errors
                    raise

            # Format price with correct precision
            price_precision = self.get_price_precision()
            stop_loss_price = float(
                "{:0.0{}f}".format(self.stop_loss_price, price_precision)
            )

            # For LONG positions: SELL to close
            # For SHORT positions: BUY to close
            close_side = "SELL" if self.position == "long" else "BUY"

            print(
                f"Creating new stop loss order: {self.symbol} {close_side} at {stop_loss_price}, Size: {self.position_size}"
            )

            # Place new stop loss order
            stop_loss_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=stop_loss_price,
                reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                quantity=self.position_size,
                workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                timeInForce="GTC",  # Good Till Cancelled
            )
            self.stop_loss_order_id = stop_loss_order["orderId"]
            print(f"✅ New trailing stop loss order placed: {stop_loss_order}")

        except BinanceAPIException as e:
            print(f"❌ Error updating stop loss order: {e}")
            print(f"Error code: {e.code}, Error message: {e.message}")
            print(
                f"Error details: Symbol={self.symbol}, Position={self.position}, Size={self.position_size}, Stop Price={self.stop_loss_price}"
            )

            # If we get an "Unknown order" error, reset the order ID and try to recreate
            if "Unknown order" in str(e):
                print("Resetting stop_loss_order_id and recreating order")
                self.stop_loss_order_id = None
                self.recreate_stop_loss_order()
        except Exception as e:
            print(f"❌ Unexpected error updating stop loss order: {e}")
            print(f"Error type: {type(e).__name__}")
            print(
                f"Error details: Symbol={self.symbol}, Position={self.position}, Size={self.position_size}, Stop Price={self.stop_loss_price}"
            )

    def reassess_position(self, current_price, timestamp):
        """
        Reassess current position based on latest signals and potentially exit or adjust

        Args:
            current_price: Current price of the asset
            timestamp: Current timestamp

        Returns:
            dict or None: Result of position adjustment if any action was taken
        """
        if not self.has_open_position():
            return None

        print("\n=== POSITION REASSESSMENT ===")
        print(f"Current position: {self.position.upper()} at ${self.entry_price:.4f}")
        print(f"Current price: ${current_price:.4f}")

        # Get latest data for signal analysis
        latest_df = self.get_latest_data(lookback_candles=20)
        if latest_df is None or len(latest_df) < 20:
            print("Not enough data for position reassessment")
            return None

        # Get traditional signal
        traditional_signal = latest_df.iloc[-1]["signal"]

        # Get ML signal if enabled
        ml_signal, ml_confidence = 0, 0
        if self.ml_manager and self.use_ml_signals:
            try:
                ml_signal, ml_confidence = self.ml_manager.get_ml_signal(
                    self.symbol, latest_df
                )
                print(
                    f"ML Signal: {'BUY' if ml_signal == 1 else 'SELL' if ml_signal == -1 else 'NEUTRAL'} with {ml_confidence:.2f} confidence"
                )
            except Exception as e:
                print(f"Error getting ML signal: {e}")

        # Get enhanced signal if enabled
        enhanced_signal, enhanced_confidence = 0, 0
        if self.use_enhanced_signals:
            enhanced_signal, enhanced_confidence = self.generate_enhanced_signal(
                latest_df
            )
            print(
                f"Enhanced Signal: {'BUY' if enhanced_signal == 1 else 'SELL' if enhanced_signal == -1 else 'NEUTRAL'} with {enhanced_confidence:.2f} confidence"
            )

        # Determine current position direction (1 for long, -1 for short)
        position_direction = 1 if self.position == "long" else -1

        # Combine signals to determine if position should be adjusted
        signal_strength = 0
        signal_count = 0

        # Count signals that agree with current position
        if traditional_signal == position_direction:
            signal_strength += 1
            signal_count += 1
            print(f"Traditional signal AGREES with current position")
        elif traditional_signal != 0:
            signal_strength -= 1
            signal_count += 1
            print(f"Traditional signal CONTRADICTS current position")

        if ml_signal == position_direction and self.use_ml_signals:
            signal_strength += 1 if ml_confidence >= self.ml_confidence else 0.5
            signal_count += 1
            print(
                f"ML signal AGREES with current position (confidence: {ml_confidence:.2f})"
            )
        elif ml_signal != 0 and self.use_ml_signals:
            signal_strength -= 1 if ml_confidence >= self.ml_confidence else 0.5
            signal_count += 1
            print(
                f"ML signal CONTRADICTS current position (confidence: {ml_confidence:.2f})"
            )

        if enhanced_signal == position_direction and self.use_enhanced_signals:
            signal_strength += 1
            signal_count += 1
            print(
                f"Enhanced signal AGREES with current position (confidence: {enhanced_confidence:.2f})"
            )
        elif enhanced_signal != 0 and self.use_enhanced_signals:
            signal_strength -= 1
            signal_count += 1
            print(
                f"Enhanced signal CONTRADICTS current position (confidence: {enhanced_confidence:.2f})"
            )

        # Calculate signal agreement ratio
        if signal_count > 0:
            agreement_ratio = signal_strength / signal_count
        else:
            agreement_ratio = 0

        print(
            f"Signal agreement ratio: {agreement_ratio:.2f} ({signal_strength}/{signal_count})"
        )

        # Determine action based on signal agreement
        if (
            agreement_ratio <= -0.5
        ):  # Strong contradiction (more than half of signals disagree)
            print("DECISION: Strong signal contradiction - Closing position")
            return self.close_position(
                current_price, timestamp, reason="SIGNAL_REVERSAL"
            )

        elif agreement_ratio < 0:  # Mild contradiction
            # Check if we have already executed a partial take profit
            if not self.partial_tp_executed and self.original_position_size > 0:
                print(
                    "DECISION: Mild signal contradiction - Executing partial exit (50%)"
                )
                return self.execute_partial_take_profit(current_price, timestamp)
            else:
                print(
                    "DECISION: Mild signal contradiction - Partial exit already executed, holding remaining position"
                )

        elif agreement_ratio > 0.5 and self.enable_pyramiding:  # Strong agreement
            # Check if we can add to position (pyramiding)
            if self.pyramid_entries < self.max_pyramid_entries:
                # Calculate profit so far
                if self.position == "long":
                    current_profit_pct = (
                        (current_price - self.entry_price) / self.entry_price * 100
                    )
                else:  # short
                    current_profit_pct = (
                        (self.entry_price - current_price) / self.entry_price * 100
                    )

                # Only pyramid if we're in profit above threshold
                if current_profit_pct >= self.pyramid_threshold_pct * 100:
                    print(
                        f"DECISION: Strong signal agreement and position in profit ({current_profit_pct:.2f}%) - Adding to position"
                    )
                    return self.execute_pyramid_entry(current_price, timestamp)
                else:
                    print(
                        f"DECISION: Strong signal agreement but profit ({current_profit_pct:.2f}%) below threshold ({self.pyramid_threshold_pct * 100:.2f}%) - Holding position"
                    )
            else:
                print(
                    f"DECISION: Strong signal agreement but maximum pyramid entries reached - Holding position"
                )
        else:
            print("DECISION: Neutral signal assessment - Holding position")

        # If we reach here, no action was taken
        return None

    def train_ml_model(self):
        """Train the ML model with the latest data"""
        if not self.ml_manager:
            print("ML manager not initialized, cannot train model")
            return False

        try:
            print(f"\n=== Training ML Model for {self.symbol} ===")
            # Get enough data for training
            training_df = self.get_latest_data(lookback_candles=500)

            if training_df is None or len(training_df) < 100:
                print(
                    f"Not enough data for ML training. Need at least 100 candles, got {len(training_df) if training_df is not None else 0}"
                )
                return False

            # Train the model
            model, scaler = self.ml_manager.train_ml_model(
                self.symbol, training_df, force_retrain=True
            )

            if model is None or scaler is None:
                print("Failed to train ML model")
                return False

            # Update last train time
            self.last_train_time = datetime.now()

            print(
                f"ML model for {self.symbol} trained successfully at {self.last_train_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Send notification
            self.send_notification(
                f"🧠 ML MODEL TRAINED\n"
                f"Symbol: {self.symbol}\n"
                f"Time: {self.last_train_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Data Points: {len(training_df)}\n"
                f"Next Training: {(self.last_train_time + timedelta(hours=self.retrain_interval)).strftime('%Y-%m-%d %H:%M:%S') if self.retrain_interval > 0 else 'Not scheduled'}"
            )

            return True

        except Exception as e:
            print(f"Error training ML model: {e}")
            return False

    def generate_trading_signal(self, latest_df):
        """
        Generate trading signal based on latest data and configured strategies

        Args:
            latest_df: DataFrame with latest price data and indicators

        Returns:
            int: 1 for buy signal, -1 for sell signal, 0 for no signal
        """
        if latest_df is None or len(latest_df) < 2:
            print("Not enough data to generate signal")
            return 0

        print("\n=== SIGNAL GENERATION ===")
        
        # Get current position info
        has_position = self.has_open_position()
        if has_position:
            print(f"Current position: {self.position.upper()}")
            # Don't generate new signals if we already have a position
            print("No new signals generated while position is open")
            return 0

        # Initialize signal components
        traditional_signal = 0
        ml_signal = 0
        ml_confidence = 0
        enhanced_signal = 0
        enhanced_confidence = 0

        # 1. Get traditional technical signals
        try:
            # Get the latest and previous candle
            current = latest_df.iloc[-1]
            previous = latest_df.iloc[-2]

            # Check if we have the necessary indicators
            required_indicators = ['ema_8', 'ema_21', 'rsi', 'macd', 'macd_signal']
            has_indicators = all(indicator in current.index for indicator in required_indicators)

            if has_indicators:
                # EMA Crossover
                ema_8_current = current['ema_8']
                ema_21_current = current['ema_21']
                ema_8_prev = previous['ema_8']
                ema_21_prev = previous['ema_21']

                # RSI conditions
                rsi = current['rsi']

                # MACD conditions
                macd_current = current['macd']
                signal_current = current['macd_signal']
                macd_prev = previous['macd']
                signal_prev = previous['macd_signal']

                # Generate traditional signal
                if ema_8_prev <= ema_21_prev and ema_8_current > ema_21_current:
                    # Bullish EMA crossover
                    if rsi < 70:  # Not overbought
                        traditional_signal = 1
                        print("Traditional Signal: BULLISH (EMA crossover with RSI confirmation)")
                elif ema_8_prev >= ema_21_prev and ema_8_current < ema_21_current:
                    # Bearish EMA crossover
                    if rsi > 30:  # Not oversold
                        traditional_signal = -1
                        print("Traditional Signal: BEARISH (EMA crossover with RSI confirmation)")

                # MACD confirmation
                if macd_prev <= signal_prev and macd_current > signal_current:
                    if traditional_signal != -1:  # Don't contradict bearish signal
                        traditional_signal = 1
                        print("Traditional Signal: BULLISH (MACD crossover)")
                elif macd_prev >= signal_prev and macd_current < signal_current:
                    if traditional_signal != 1:  # Don't contradict bullish signal
                        traditional_signal = -1
                        print("Traditional Signal: BEARISH (MACD crossover)")
            else:
                print("Warning: Some required indicators are missing")

        except Exception as e:
            print(f"Error generating traditional signals: {e}")

        # 2. Get ML signals if enabled
        if self.use_ml_signals and self.ml_manager:
            try:
                ml_signal, ml_confidence = self.ml_manager.get_ml_signal(self.symbol, latest_df)
                print(f"ML Signal: {'BUY' if ml_signal == 1 else 'SELL' if ml_signal == -1 else 'NEUTRAL'} with {ml_confidence:.2f} confidence")

                # Check if ML model needs retraining
                if self.train_ml and self.retrain_interval > 0:
                    if self.last_train_time is None or \
                       (datetime.now() - self.last_train_time).total_seconds() > (self.retrain_interval * 3600):
                        print("ML model requires retraining...")
                        self.train_ml_model()

            except Exception as e:
                print(f"Error getting ML signal: {e}")
                ml_signal = 0
                ml_confidence = 0

        # 3. Get enhanced signals if enabled
        if self.use_enhanced_signals:
            enhanced_signal, enhanced_confidence = self.generate_enhanced_signal(latest_df)
            print(f"Enhanced Signal: {'BUY' if enhanced_signal == 1 else 'SELL' if enhanced_signal == -1 else 'NEUTRAL'} with {enhanced_confidence:.2f} confidence")

        # 4. Analyze market trend if trend following is enabled
        trend_direction = 0
        trend_strength = 0
        if self.trend_following_mode:
            trend_direction, trend_strength = self.analyze_market_trend()
            print(f"Market Trend: {'BULLISH' if trend_direction == 1 else 'BEARISH' if trend_direction == -1 else 'NEUTRAL'} with {trend_strength:.1f}% strength")

        # 5. Combine signals to make final decision
        final_signal = 0
        signal_weight = 0
        total_weight = 0

        # Traditional signal weight (base weight: 1.0)
        if traditional_signal != 0:
            signal_weight += traditional_signal * 1.0
            total_weight += 1.0

        # ML signal weight (base weight: 1.5 if confidence is high)
        if self.use_ml_signals and ml_signal != 0:
            if ml_confidence >= self.ml_confidence:
                signal_weight += ml_signal * 1.5
                total_weight += 1.5
            else:
                signal_weight += ml_signal * 0.5
                total_weight += 0.5

        # Enhanced signal weight (base weight: 2.0 if confidence is high)
        if self.use_enhanced_signals and enhanced_signal != 0:
            if enhanced_confidence >= 70:
                signal_weight += enhanced_signal * 2.0
                total_weight += 2.0
            else:
                signal_weight += enhanced_signal * 1.0
                total_weight += 1.0

        # Calculate weighted average signal
        if total_weight > 0:
            weighted_signal = signal_weight / total_weight
            
            # Strong signal threshold
            if abs(weighted_signal) >= 0.6:
                final_signal = 1 if weighted_signal > 0 else -1
            
            print(f"Weighted signal: {weighted_signal:.2f}")

        # Apply trend following filter if enabled
        if self.trend_following_mode and final_signal != 0:
            if trend_strength >= 50:  # Only apply trend filter if trend is strong enough
                if (final_signal == 1 and trend_direction == -1) or \
                   (final_signal == -1 and trend_direction == 1):
                    print(f"Signal rejected: Against market trend (strength: {trend_strength:.1f}%)")
                    final_signal = 0
                else:
                    print(f"Signal aligned with market trend (strength: {trend_strength:.1f}%)")
            else:
                print(f"Weak market trend ({trend_strength:.1f}%) - proceeding with signal")

        # Apply scalping mode adjustments if enabled
        if self.use_scalping_mode and final_signal != 0:
            # In scalping mode, we want to be more aggressive with entries
            # but also more careful with market conditions
            
            # Check recent volatility
            if 'ATR' in latest_df.columns:
                atr = latest_df['ATR'].iloc[-1]
                avg_price = latest_df['close'].mean()
                volatility_ratio = atr / avg_price
                
                # For scalping, we prefer moderate volatility
                if volatility_ratio < 0.001:  # Too low volatility
                    print("Signal rejected: Volatility too low for scalping")
                    final_signal = 0
                elif volatility_ratio > 0.005:  # Too high volatility
                    print("Signal rejected: Volatility too high for scalping")
                    final_signal = 0
                else:
                    print(f"Volatility suitable for scalping ({volatility_ratio*100:.3f}%)")

        # Final signal summary
        if final_signal == 1:
            print("FINAL SIGNAL: BUY 🟢")
        elif final_signal == -1:
            print("FINAL SIGNAL: SELL 🔴")
        else:
            print("FINAL SIGNAL: NEUTRAL ⚪")

        return final_signal


def run_real_trading(
    realtime_trader,
    duration_hours=24,
    update_interval_minutes=15,
    update_interval_seconds=0,
):
    """
    Run real-time trading

    Args:
        realtime_trader: The RealtimeTrader instance
        duration_hours: How long to run the trading in hours
        update_interval_minutes: How often to update in minutes
        update_interval_seconds: How often to update in seconds (overrides minutes if set)
    """
    print(
        f"Starting real-time trading for {realtime_trader.symbol} for {duration_hours} hours"
    )

    # Determine the update interval
    if update_interval_seconds > 0:
        update_interval = update_interval_seconds
        interval_unit = "seconds"
    else:
        update_interval = update_interval_minutes * 60  # Convert minutes to seconds
        interval_unit = "minutes"

    print(
        f"Update interval: {update_interval_seconds if update_interval_seconds > 0 else update_interval_minutes} {interval_unit}"
    )
    print(f"Daily profit target: ${realtime_trader.daily_profit_target:.2f}")

    # Initialize Binance client for real trading
    realtime_trader.initialize_trading_client()

    # Train ML model at startup if enabled
    if realtime_trader.use_ml_signals and realtime_trader.train_ml:
        print("Training ML model at startup...")
        realtime_trader.train_ml_model()

    # Calculate start and end times
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    # Track the current day for daily resets
    current_day = datetime.now().date()

    # Track last signal time for cooldown
    last_signal_time = None

    # Main trading loop
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            print(f"\n=== Update: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            # Check signal cooldown
            if last_signal_time is not None:
                cooldown_remaining = realtime_trader.signal_cooldown_minutes * 60 - (current_time - last_signal_time).total_seconds()
                if cooldown_remaining > 0:
                    mins, secs = divmod(int(cooldown_remaining), 60)
                    print(f"⏳ Signal cooldown: {mins:02d}:{secs:02d} remaining")
                else:
                    print("✅ Signal cooldown complete")

            # Get latest data
            latest_df = realtime_trader.get_latest_data(lookback_candles=50)  # Increased lookback for better analysis

            if latest_df is None or len(latest_df) < 50:
                print("Error fetching latest data, will retry next interval")
                time.sleep(60)
                continue

            # Get the latest candle
            latest_candle = latest_df.iloc[-1]
            current_price = float(latest_candle["close"])
            print(f"Current {realtime_trader.symbol} price: ${current_price:.4f}")

            # Check existing position first
            if realtime_trader.has_open_position():
                print("Checking existing position...")
                # Check take profit and stop loss
                result = realtime_trader.check_take_profit_stop_loss(current_price, current_time)
                if result:
                    print(f"Position closed: {result}")
                    last_signal_time = current_time  # Reset cooldown after position close
                
                # Reassess position if enabled
                if realtime_trader.reassess_positions:
                    reassess_result = realtime_trader.reassess_position(current_price, current_time)
                    if reassess_result:
                        print(f"Position adjustment: {reassess_result}")
            else:
                # Generate trading signal
                signal = realtime_trader.generate_trading_signal(latest_df)
                
                # Execute trade if signal is generated and cooldown is complete
                if signal != 0 and (last_signal_time is None or 
                    (current_time - last_signal_time).total_seconds() >= realtime_trader.signal_cooldown_minutes * 60):
                    
                    print(f"Signal generated: {'LONG' if signal == 1 else 'SHORT'}")
                    
                    # Execute the trade
                    trade_result = realtime_trader.execute_trade(signal, current_price, current_time)
                    if trade_result:
                        print(f"Trade executed: {trade_result}")
                        last_signal_time = current_time  # Update last signal time
                        
                        # Calculate position value
                        position_value = realtime_trader.position_size * current_price
                        
                        # Send notification if enabled
                        realtime_trader.send_notification(
                            f"🔔 New Trade Opened\n"
                            f"Symbol: {realtime_trader.symbol}\n"
                            f"Type: {'LONG' if signal == 1 else 'SHORT'}\n"
                            f"Entry Price: ${current_price:.4f}\n"
                            f"Position Size: {realtime_trader.position_size:.4f}\n"
                            f"Position Value: ${position_value:.2f}\n"
                            f"Take Profit: ${realtime_trader.take_profit_price:.4f} ({((realtime_trader.take_profit_price - current_price) / current_price * 100 * (1 if signal == 1 else -1)):.2f}%)\n"
                            f"Stop Loss: ${realtime_trader.stop_loss_price:.4f} ({((realtime_trader.stop_loss_price - current_price) / current_price * 100 * (1 if signal == 1 else -1)):.2f}%)\n"
                            f"Leverage: {realtime_trader.leverage}x\n"
                            f"Risk: ${(position_value * abs((realtime_trader.stop_loss_price - current_price) / current_price)):.2f}\n"
                            f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        
                        # Also send notification for position management settings
                        if realtime_trader.partial_tp_enabled:
                            partial_tp_price = current_price * (1 + realtime_trader.partial_tp_pct if signal == 1 else 1 - realtime_trader.partial_tp_pct)
                            realtime_trader.send_notification(
                                f"📊 Position Management\n"
                                f"Partial TP Enabled: {realtime_trader.partial_tp_size * 100:.0f}% at ${partial_tp_price:.4f} ({realtime_trader.partial_tp_pct * 100:.1f}%)\n"
                                f"Trailing Stop: Active\n"
                                f"Break-even Stop: Will move SL to entry after partial TP"
                            )
            
            # Save results
            realtime_trader.save_trading_results()

            # Wait for the next update interval
            next_update = current_time + timedelta(seconds=update_interval)
            sleep_time = (next_update - datetime.now()).total_seconds()

            if sleep_time > 0:
                print(f"\nNext update at {next_update.strftime('%H:%M:%S')}")
                
                # Start countdown timer
                start_time = time.time()
                end_wait_time = start_time + sleep_time

                while time.time() < end_wait_time:
                    # Calculate remaining time
                    remaining = end_wait_time - time.time()
                    mins, secs = divmod(int(remaining), 60)
                    
                    # Calculate signal cooldown
                    if last_signal_time is not None:
                        cooldown_remaining = realtime_trader.signal_cooldown_minutes * 60 - (datetime.now() - last_signal_time).total_seconds()
                        cooldown_status = ""
                        if cooldown_remaining > 0:
                            c_mins, c_secs = divmod(int(cooldown_remaining), 60)
                            cooldown_status = f" | 🔒 Signal cooldown: {c_mins:02d}:{c_secs:02d}"
                        else:
                            cooldown_status = " | 🔓 Ready for new signals"
                    else:
                        cooldown_status = " | 🔓 Ready for new signals"

                    # Create progress bar
                    bar_length = 20
                    progress = 1 - (remaining / sleep_time)
                    filled_len = int(bar_length * progress)
                    bar = "█" * filled_len + "░" * (bar_length - filled_len)

                    # Display countdown with signal cooldown status
                    countdown = f"⏱️ Next update in: {mins:02d}:{secs:02d} [{bar}] {progress*100:.0f}%{cooldown_status}"
                    print(countdown, end="\r", flush=True)
                    time.sleep(1)

                print(" " * 150, end="\r")  # Clear the line
                print("Updating now...")
            else:
                print("Processing took longer than update interval, continuing immediately")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nTrading interrupted by user")
            break

        except Exception as e:
            print(f"Error in trading loop: {e}")
            realtime_trader.send_notification(f"⚠️ ERROR: {e}")
            time.sleep(60)

    # Trading completed
    print("\n=== Trading Completed ===")
    final_balance = realtime_trader.get_account_balance()

    return {
        "final_balance": final_balance,
        "profit_loss": final_balance - realtime_trader.initial_investment,
        "return_pct": (final_balance - realtime_trader.initial_investment) / realtime_trader.initial_investment * 100,
    }
