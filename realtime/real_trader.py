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
    ):
        """
        Initialize the RealtimeTrader

        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            initial_investment: Initial investment amount in USD
            daily_profit_target: Daily profit target in USD
            leverage: Leverage to use for trading
            test_mode: Whether to run in test mode (no real trades)
            use_full_investment: Whether to use the full investment amount
            use_full_margin: Whether to use the full margin available
            compound_interest: Whether to compound interest
            enable_pyramiding: Whether to enable pyramiding (adding to winning positions)
            max_pyramid_entries: Maximum number of pyramid entries
            pyramid_threshold_pct: Minimum profit percentage before adding to position
            use_dynamic_take_profit: Whether to use dynamic take profit targets
            trend_following_mode: Whether to use trend following mode
            use_enhanced_signals: Whether to use enhanced signal generation
            signal_confirmation_threshold: Minimum number of indicators confirming a signal
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
        self.last_signal_time = None
        self.signal_cooldown_minutes = 60  # Minimum time between signals
        
        # Add per-symbol balance tracking for compound interest
        self.symbol_balance = initial_investment
        self.total_symbol_profit = 0.0

        # Set leverage (constrain between 1x and 20x)
        self.leverage = max(1, min(20, leverage))

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
        self.max_position_size = 0.95 if use_full_investment else 0.2  # Use 95% of balance if full investment mode
        self.max_daily_loss = 0.1  # Maximum 10% daily loss of initial investment
        self.risk_per_trade = 0.95 if use_full_investment else 0.02  # Use 95% of balance for risk if full investment mode
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
        self.partial_tp_executed = False  # Track if partial take profit has been executed
        self.original_position_size = 0  # Track original position size before partial TP

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
                    
                    # Ensure stop_loss_price and take_profit_price are initialized if they're None
                    if self.stop_loss_price is None:
                        # Calculate default stop loss based on position type and a conservative 1% risk
                        if self.position == "long":
                            self.stop_loss_price = self.entry_price * 0.99  # 1% below entry for long
                        else:
                            self.stop_loss_price = self.entry_price * 1.01  # 1% above entry for short
                        print(f"WARNING: stop_loss_price was None, initialized to ${self.stop_loss_price:.2f}")
                    
                    if self.take_profit_price is None:
                        # Calculate default take profit based on position type and a 2% target
                        if self.position == "long":
                            self.take_profit_price = self.entry_price * 1.02  # 2% above entry for long
                        else:
                            self.take_profit_price = self.entry_price * 0.98  # 2% below entry for short
                        print(f"WARNING: take_profit_price was None, initialized to ${self.take_profit_price:.2f}")
                    
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
            "original_position_size": self.original_position_size if self.original_position_size > 0 else self.position_size
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
        print(f"DEBUG - Position: {self.position}, Stop Loss: {self.stop_loss_price}, Take Profit: {self.take_profit_price}")
        
        # Always verify that stop loss and take profit orders are active first
        if not self.test_mode and (self.stop_loss_order_id or self.take_profit_order_id):
            print(f"Checking orders for {self.symbol} at price ${current_price:.2f}")
            self.verify_stop_loss_take_profit_orders()

        # Calculate current profit/loss for daily target checking
        if self.position == 'long':
            current_profit = self.position_size * (current_price - self.entry_price)
            current_profit_pct = (current_price - self.entry_price) / self.entry_price * 100
            print(f"Current profit for LONG position: ${current_profit:.2f} ({current_profit_pct:.2f}%) (Entry: ${self.entry_price:.2f}, Current: ${current_price:.2f})")
            
            # Check for pyramiding opportunity
            if self.enable_pyramiding and self.pyramid_entries < self.max_pyramid_entries:
                # Only pyramid if we're in profit above threshold
                if current_profit_pct >= self.pyramid_threshold_pct * 100:
                    self.execute_pyramid_entry(current_price, timestamp)
            
            # Check for partial take profit if enabled and not already executed
            if self.partial_tp_enabled and not self.partial_tp_executed and current_price >= self.entry_price * (1 + self.partial_tp_pct):
                print(f"🎯 Partial take profit triggered at ${current_price:.2f} ({self.partial_tp_pct*100:.1f}% gain)")
                self.execute_partial_take_profit(current_price, timestamp)
            
            # Check if we should update the trailing stop loss
            if current_price > self.entry_price and self.stop_loss_price is not None:
                # Calculate potential new stop loss based on current price
                # Use a percentage of the current profit as the trailing distance
                trailing_distance = min(0.005, (current_price - self.entry_price) * 0.3)  # 30% of current profit, max 0.5%
                potential_stop_loss = current_price - (current_price * trailing_distance)
                
                # Only update if the new stop loss would be higher than the current one
                if potential_stop_loss > self.stop_loss_price:
                    old_stop_loss = self.stop_loss_price
                    self.stop_loss_price = potential_stop_loss
                    print(f"🔄 Updated trailing stop loss: ${old_stop_loss:.2f} -> ${self.stop_loss_price:.2f}")
                    
                    # Update the stop loss order in Binance if not in test mode
                    if not self.test_mode:
                        # If we have a stop loss order ID, update it, otherwise recreate it
                        if self.stop_loss_order_id:
                            self.update_stop_loss_order()
                        else:
                            self.recreate_stop_loss_order()
            
            # Check for daily profit target or take profit price
            day_key = timestamp.strftime('%Y-%m-%d')
            if not hasattr(self, 'daily_profits'):
                self.daily_profits = {}
            
            day_profit = self.daily_profits.get(day_key, 0) + current_profit
            
            # Check if daily profit target is reached OR take profit price is hit
            if day_profit >= self.daily_profit_target or (self.take_profit_price is not None and current_price >= self.take_profit_price):
                reason = "DAILY TARGET" if day_profit >= self.daily_profit_target else "TAKE PROFIT"
                print(f"🎯 {reason} reached for LONG position! Current price: ${current_price:.2f}")
                # Close position
                result = self.close_position(current_price, timestamp, reason)
            # Check for stop loss
            elif self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                print(f"🛑 Stop loss triggered for LONG position! Current price: ${current_price:.2f}, Stop loss: ${self.stop_loss_price:.2f}")
                # Close position
                result = self.close_position(current_price, timestamp, "STOP LOSS")
            
        else:  # short
            current_profit = self.position_size * (self.entry_price - current_price)
            current_profit_pct = (self.entry_price - current_price) / self.entry_price * 100
            print(f"Current profit for SHORT position: ${current_profit:.2f} ({current_profit_pct:.2f}%) (Entry: ${self.entry_price:.2f}, Current: ${current_price:.2f})")
            
            # Check for pyramiding opportunity
            if self.enable_pyramiding and self.pyramid_entries < self.max_pyramid_entries:
                # Only pyramid if we're in profit above threshold
                if current_profit_pct >= self.pyramid_threshold_pct * 100:
                    self.execute_pyramid_entry(current_price, timestamp)
            
            # Check for partial take profit if enabled and not already executed
            if self.partial_tp_enabled and not self.partial_tp_executed and current_price <= self.entry_price * (1 - self.partial_tp_pct):
                print(f"🎯 Partial take profit triggered at ${current_price:.2f} ({self.partial_tp_pct*100:.1f}% gain)")
                self.execute_partial_take_profit(current_price, timestamp)
            
            # Check if we should update the trailing stop loss
            if current_price < self.entry_price and self.stop_loss_price is not None:
                # Calculate potential new stop loss based on current price
                # Use a percentage of the current profit as the trailing distance
                trailing_distance = min(0.005, (self.entry_price - current_price) * 0.3)  # 30% of current profit, max 0.5%
                potential_stop_loss = current_price + (current_price * trailing_distance)
                
                # Only update if the new stop loss would be lower than the current one
                if potential_stop_loss < self.stop_loss_price:
                    old_stop_loss = self.stop_loss_price
                    self.stop_loss_price = potential_stop_loss
                    print(f"🔄 Updated trailing stop loss: ${old_stop_loss:.2f} -> ${self.stop_loss_price:.2f}")
                    
                    # Update the stop loss order in Binance if not in test mode
                    if not self.test_mode:
                        # If we have a stop loss order ID, update it, otherwise recreate it
                        if self.stop_loss_order_id:
                            self.update_stop_loss_order()
                        else:
                            self.recreate_stop_loss_order()
            
            # Check for daily profit target or take profit price
            day_key = timestamp.strftime('%Y-%m-%d')
            if not hasattr(self, 'daily_profits'):
                self.daily_profits = {}
            
            day_profit = self.daily_profits.get(day_key, 0) + current_profit
            
            # Check if daily profit target is reached OR take profit price is hit
            if day_profit >= self.daily_profit_target or (self.take_profit_price is not None and current_price <= self.take_profit_price):
                reason = "DAILY TARGET" if day_profit >= self.daily_profit_target else "TAKE PROFIT"
                print(f"🎯 {reason} reached for SHORT position! Current price: ${current_price:.2f}")
                # Close position
                result = self.close_position(current_price, timestamp, reason)
            # Check for stop loss
            elif self.stop_loss_price is not None and current_price >= self.stop_loss_price:
                print(f"🛑 Stop loss triggered for SHORT position! Current price: ${current_price:.2f}, Stop loss: ${self.stop_loss_price:.2f}")
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
                day_key = timestamp.strftime('%Y-%m-%d')
                if not hasattr(self, 'daily_profits'):
                    self.daily_profits = {}
                day_profit = self.daily_profits.get(day_key, 0) + current_profit
                print(f"Daily profit: ${day_profit:.2f} / Target: ${self.daily_profit_target:.2f}")
        
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
                    print(f"Cancelled all open orders for {self.symbol} before partial take profit")
                    
                    # Reset order IDs since we've canceled them
                    self.stop_loss_order_id = None
                    self.take_profit_order_id = None
                except BinanceAPIException as e:
                    print(f"Error cancelling orders before partial take profit: {e}")
                
                # Execute the partial close
                side = 'SELL' if self.position == 'long' else 'BUY'
                print(f"Executing partial take profit: {side} {close_size} {self.symbol} at MARKET price (${current_price:.2f})")
                
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=close_size
                )
                print(f"Partial take profit order executed: {order}")
            else:
                # Test mode - simulate order
                side = 'SELL' if self.position == 'long' else 'BUY'
                print(f"TEST MODE: Simulating partial take profit: {side} {close_size} {self.symbol} at MARKET price (${current_price:.2f})")
            
            # Calculate profit for the closed portion
            if self.position == 'long':
                profit = (current_price - self.entry_price) * close_size
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
            else:  # short
                profit = (self.entry_price - current_price) * close_size
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100
            
            # Update test balance in test mode
            if self.test_mode:
                self.test_balance += profit
                print(f"TEST MODE: Updated balance to ${self.test_balance:.2f} (partial profit: ${profit:.2f})")
            
            # Update per-symbol balance for compound interest
            self.symbol_balance += profit
            self.total_symbol_profit += profit
            
            # Update position size
            remaining_size = self.position_size - close_size
            self.position_size = float("{:0.0{}f}".format(remaining_size, quantity_precision))
            
            # Move stop loss to break-even (entry price)
            self.stop_loss_price = self.entry_price
            print(f"🔄 Moved stop loss to break-even: ${self.stop_loss_price:.2f}")
            
            # Add to trade history
            partial_close_record = {
                'timestamp': timestamp,
                'action': 'PARTIAL_CLOSE',
                'price': current_price,
                'size': close_size,
                'value': close_size * current_price,
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': 'PARTIAL_TAKE_PROFIT',
                'order_type': 'MARKET'
            }
            
            self.trade_history.append(partial_close_record)
            if self.test_mode:
                self.test_trades.append(partial_close_record)
            
            # Update daily profits
            day_key = timestamp.strftime('%Y-%m-%d')
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
            
            print(f"✅ Partial take profit executed successfully. Remaining position: {self.position_size} units with stop loss at break-even.")
            
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
            
            print(f"Verifying orders for {self.symbol}. Found {len(open_orders)} open orders.")
            
            # Debug: Print all order details
            self.debug_print_orders(open_orders)
            
            for order in open_orders:
                order_id = str(order['orderId'])
                order_type = order.get('type', 'UNKNOWN')
                order_status = order.get('status', 'UNKNOWN')
                
                if order_id == str(self.stop_loss_order_id):
                    stop_loss_found = True
                    print(f"✅ Stop loss order confirmed active: ID={order_id}, Type={order_type}, Status={order_status}, Price={order.get('stopPrice', 'N/A')}")
                elif order_id == str(self.take_profit_order_id):
                    take_profit_found = True
                    print(f"✅ Take profit order confirmed active: ID={order_id}, Type={order_type}, Status={order_status}, Price={order.get('stopPrice', 'N/A')}")
                else:
                    print(f"Found other order: ID={order_id}, Type={order_type}, Status={order_status}")
            
            # If orders are not found, recreate them
            if not stop_loss_found and self.stop_loss_order_id:
                print(f"⚠️ Warning: Stop loss order {self.stop_loss_order_id} not found in open orders. Recreating...")
                self.recreate_stop_loss_order()
                
            if not take_profit_found and self.take_profit_order_id:
                print(f"⚠️ Warning: Take profit order {self.take_profit_order_id} not found in open orders. Recreating...")
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
            
        print(f"DEBUG - Recreating stop loss order for {self.position.upper()} position with size {self.position_size}")
            
        try:
            # Format price with correct precision
            price_precision = self.get_price_precision()
            stop_loss_price = float("{:0.0{}f}".format(self.stop_loss_price, price_precision))
            
            # For LONG positions: SELL to close
            # For SHORT positions: BUY to close
            close_side = 'SELL' if self.position == 'long' else 'BUY'
            
            print(f"Recreating stop loss order: {self.symbol} {close_side} at {stop_loss_price}")
            
            # Place stop loss order with exact parameters from Binance documentation
            stop_loss_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='STOP_MARKET',
                stopPrice=stop_loss_price,
                reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                quantity=self.position_size,
                workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                timeInForce="GTC"  # Good Till Cancelled
            )
            self.stop_loss_order_id = stop_loss_order['orderId']
            print(f"✅ Stop loss order recreated: {stop_loss_order}")
            
        except BinanceAPIException as e:
            print(f"❌ Error recreating stop loss order: {e}")
            # Log detailed error information
            print(f"Error details: Symbol={self.symbol}, Position={self.position}, Size={self.position_size}, Stop Price={self.stop_loss_price}")

    def recreate_take_profit_order(self):
        """Recreate take profit order if it's missing"""
        if self.test_mode or not self.has_open_position():
            return
            
        try:
            # Format price with correct precision
            price_precision = self.get_price_precision()
            take_profit_price = float("{:0.0{}f}".format(self.take_profit_price, price_precision))
            
            # For LONG positions: SELL to close
            # For SHORT positions: BUY to close
            close_side = 'SELL' if self.position == 'long' else 'BUY'
            
            print(f"Recreating take profit order: {self.symbol} {close_side} at {take_profit_price}")
            
            # Place take profit order with exact parameters from Binance documentation
            take_profit_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit_price,
                reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                quantity=self.position_size,
                workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                timeInForce="GTC"  # Good Till Cancelled
            )
            self.take_profit_order_id = take_profit_order['orderId']
            print(f"✅ Take profit order recreated: {take_profit_order}")
            
        except BinanceAPIException as e:
            print(f"❌ Error recreating take profit order: {e}")

    def get_symbol_info(self):
        """Get symbol information including precision requirements"""
        try:
            # For spot trading
            exchange_info = self.client.get_exchange_info()
            
            # Find the symbol info
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
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
            for symbol_info in futures_exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    return symbol_info
                    
            return None
        except BinanceAPIException as e:
            print(f"Error getting futures symbol info: {e}")
            return None

    def get_quantity_precision(self):
        """Get the quantity precision for the symbol"""
        # Try futures first
        futures_info = self.get_futures_symbol_info()
        if futures_info and 'quantityPrecision' in futures_info:
            return futures_info['quantityPrecision']
        
        # Fall back to spot trading calculation
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 5 decimal places if we can't get the info
            return 5
            
        # Get the lot size filter
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                # Calculate precision based on step size
                if step_size == 1.0:
                    return 0
                precision = 0
                step_size_str = "{:0.8f}".format(step_size)
                while step_size_str[len(step_size_str) - 1 - precision] == '0':
                    precision += 1
                return 8 - precision
                
        # Default to 5 decimal places if we can't find the LOT_SIZE filter
        return 5

    def get_price_precision(self):
        """Get the price precision for the symbol"""
        # Try futures first
        futures_info = self.get_futures_symbol_info()
        if futures_info and 'pricePrecision' in futures_info:
            return futures_info['pricePrecision']
        
        # Fall back to spot trading calculation
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 2 decimal places if we can't get the info
            return 2
            
        # Get the price filter
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter['tickSize'])
                # Calculate precision based on tick size
                if tick_size == 1.0:
                    return 0
                precision = 0
                tick_size_str = "{:0.8f}".format(tick_size)
                while tick_size_str[len(tick_size_str) - 1 - precision] == '0':
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
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'MIN_NOTIONAL':
                return float(filter['minNotional'])
                
        # Default to 10 if we can't find the MIN_NOTIONAL filter
        return 10.0

    def get_min_quantity(self):
        """Get the minimum quantity for the symbol"""
        symbol_info = self.get_symbol_info()
        if not symbol_info:
            # Default to 0.001 if we can't get the info
            return 0.001
            
        # Get the lot size filter
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                return float(filter['minQty'])
                
        # Default to 0.001 if we can't find the LOT_SIZE filter
        return 0.001

    def calculate_dynamic_position_size(self, current_price, atr_value=None):
        """
        Calculate position size dynamically based on volatility and recent performance
        
        Args:
            current_price: Current price of the asset
            atr_value: Average True Range value (if None, will be calculated)
            
        Returns:
            Adjusted position size
        """
        # Get base position size using current method
        base_position_size = self.calculate_position_size(current_price)
        
        # If no ATR provided, use a default volatility adjustment
        if atr_value is None:
            # Get latest data to calculate ATR
            latest_df = self.get_latest_data(lookback_candles=14)
            if latest_df is not None and len(latest_df) >= 14:
                # Calculate ATR
                high = latest_df['high'].values
                low = latest_df['low'].values
                close = latest_df['close'].values
                
                # Calculate True Range
                tr1 = np.abs(high[1:] - low[1:])
                tr2 = np.abs(high[1:] - close[:-1])
                tr3 = np.abs(low[1:] - close[:-1])
                tr = np.maximum(np.maximum(tr1, tr2), tr3)
                
                # 14-period ATR
                atr_value = np.mean(tr[-14:])
            else:
                # Default to 2% volatility if we can't calculate ATR
                atr_value = current_price * 0.02
        
        # Calculate volatility ratio (ATR as percentage of price)
        volatility_ratio = atr_value / current_price
        
        # Adjust position size based on volatility
        # Lower position size when volatility is high, increase when volatility is low
        volatility_adjustment = 1.0
        if volatility_ratio > 0.03:  # High volatility
            volatility_adjustment = 0.7  # Reduce position size by 30%
        elif volatility_ratio < 0.01:  # Low volatility
            volatility_adjustment = 1.3  # Increase position size by 30%
        
        # Performance adjustment based on recent trades
        performance_adjustment = 1.0
        if len(self.trade_history) >= 3:
            # Get last 3 trades
            recent_trades = self.trade_history[-3:]
            profitable_trades = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
            
            if profitable_trades == 3:
                # All recent trades profitable, increase size
                performance_adjustment = 1.2
            elif profitable_trades == 0:
                # All recent trades losing, decrease size
                performance_adjustment = 0.8
        
        # Calculate final position size with adjustments
        adjusted_position_size = base_position_size * volatility_adjustment * performance_adjustment
        
        # Ensure position size doesn't exceed maximum risk
        max_position_size = self.calculate_position_size(current_price, risk_pct=0.02)  # Max 2% risk
        adjusted_position_size = min(adjusted_position_size, max_position_size)
        
        # Format with correct precision
        quantity_precision = self.get_quantity_precision()
        adjusted_position_size = float("{:0.0{}f}".format(adjusted_position_size, quantity_precision))
        
        print(f"Dynamic position sizing: Base={base_position_size}, Volatility Adj={volatility_adjustment:.2f}, " +
              f"Performance Adj={performance_adjustment:.2f}, Final={adjusted_position_size}")
        
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
                atr = latest_df['atr'].iloc[-1] if 'atr' in latest_df.columns else (current_price * 0.01)
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
        """
        Calculate dynamic take profit levels based on market conditions
        
        Args:
            current_price: Current price of the asset
            
        Returns:
            Dynamic take profit price
        """
        # Get latest data to analyze market conditions
        latest_df = self.get_latest_data(lookback_candles=20)
        
        if latest_df is None or len(latest_df) < 20:
            # Fallback to standard take profit if not enough data
            if self.position == 'long':
                return current_price * (1 + self.take_profit_pct)
            else:  # short
                return current_price * (1 - self.take_profit_pct)
        
        # Calculate volatility using ATR
        high = latest_df['high'].values
        low = latest_df['low'].values
        close = latest_df['close'].values
        
        # Calculate True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # 14-period ATR
        atr = np.mean(tr[-14:])
        
        # Calculate volatility ratio (ATR as percentage of price)
        volatility_ratio = atr / current_price
        
        # Analyze trend strength using moving averages
        if 'ema_8' in latest_df.columns and 'ema_21' in latest_df.columns:
            ema_8 = latest_df['ema_8'].iloc[-1]
            ema_21 = latest_df['ema_21'].iloc[-1]
            ema_diff = (ema_8 - ema_21) / ema_21 * 100  # Percentage difference
            
            # Determine trend direction and strength
            if ema_diff > 0.5:  # Strong uptrend
                self.trend_direction = 1
                self.trend_strength = min(100, abs(ema_diff) * 20)  # Scale to 0-100
            elif ema_diff < -0.5:  # Strong downtrend
                self.trend_direction = -1
                self.trend_strength = min(100, abs(ema_diff) * 20)  # Scale to 0-100
            else:  # Neutral or weak trend
                self.trend_direction = 0 if abs(ema_diff) < 0.1 else (1 if ema_diff > 0 else -1)
                self.trend_strength = min(100, abs(ema_diff) * 10)  # Scale to 0-100
        else:
            # Simple trend detection using price action
            price_5_periods_ago = latest_df['close'].iloc[-5]
            price_change_pct = (current_price - price_5_periods_ago) / price_5_periods_ago * 100
            
            if abs(price_change_pct) < 0.5:  # Sideways market
                self.trend_direction = 0
                self.trend_strength = 20
            else:
                self.trend_direction = 1 if price_change_pct > 0 else -1
                self.trend_strength = min(100, abs(price_change_pct) * 10)  # Scale to 0-100
        
        # Adjust take profit based on trend strength and volatility
        base_tp_pct = self.take_profit_pct
        
        # In strong trend, extend take profit target
        if (self.position == 'long' and self.trend_direction == 1) or (self.position == 'short' and self.trend_direction == -1):
            # Aligned with trend - extend take profit
            trend_factor = 1.0 + (self.trend_strength / 100)  # 1.0 to 2.0
            volatility_factor = 1.0 + (volatility_ratio * 10)  # Adjust based on volatility
            adjusted_tp_pct = base_tp_pct * trend_factor * volatility_factor
            
            # Cap the maximum take profit percentage
            adjusted_tp_pct = min(adjusted_tp_pct, base_tp_pct * 3)
            
            print(f"Dynamic TP: Extended target due to strong {'up' if self.trend_direction == 1 else 'down'}trend " +
                  f"(Strength: {self.trend_strength:.1f}%, Volatility: {volatility_ratio*100:.2f}%)")
        else:
            # Counter-trend or neutral - reduce take profit
            trend_factor = 0.8  # Reduce take profit
            volatility_factor = 1.0 + (volatility_ratio * 5)  # Still adjust for volatility
            adjusted_tp_pct = base_tp_pct * trend_factor * volatility_factor
            
            print(f"Dynamic TP: Reduced target due to {'counter-trend' if self.trend_direction != 0 else 'neutral'} " +
                  f"conditions (Strength: {self.trend_strength:.1f}%, Volatility: {volatility_ratio*100:.2f}%)")
        
        # Calculate final take profit price
        if self.position == 'long':
            take_profit_price = current_price * (1 + adjusted_tp_pct)
        else:  # short
            take_profit_price = current_price * (1 - adjusted_tp_pct)
        
        # Format with correct precision
        price_precision = self.get_price_precision()
        take_profit_price = float("{:0.0{}f}".format(take_profit_price, price_precision))
        
        print(f"Dynamic take profit calculated: ${take_profit_price:.2f} " +
              f"(Base: {base_tp_pct*100:.1f}%, Adjusted: {adjusted_tp_pct*100:.1f}%)")
        
        return take_profit_price

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
        if 'ema_8' in latest_df.columns and 'ema_21' in latest_df.columns and 'ema_50' in latest_df.columns:
            ema_8 = latest_df['ema_8'].iloc[-1]
            ema_21 = latest_df['ema_21'].iloc[-1]
            ema_50 = latest_df['ema_50'].iloc[-1]
            
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
            prices = latest_df['close'].values
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
        current_price = latest_df['close'].iloc[-1]
        price_10_ago = latest_df['close'].iloc[-10]
        price_30_ago = latest_df['close'].iloc[-30]
        
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
        if 'volume' in latest_df.columns:
            recent_volume = np.mean(latest_df['volume'].iloc[-5:])
            avg_volume = np.mean(latest_df['volume'].iloc[-20:])
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
        final_strength = (ma_strength * ma_weight) + (pa_strength * pa_weight) + (volume_strength * vol_weight)
        
        # Adjust strength based on agreement between indicators
        if ma_direction == pa_direction and ma_direction != 0:
            final_strength *= 1.2  # Boost strength when indicators agree
        
        # Cap at 100
        final_strength = min(100, final_strength)
        
        print(f"Market trend analysis: Direction={final_direction}, Strength={final_strength:.1f}%")
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
        if latest_df is None or len(latest_df) < 50:
            return (0, 0)  # No signal if not enough data
            
        # Get the latest candle
        latest = latest_df.iloc[-1]
        
        # Initialize signal counters
        long_signals = 0
        short_signals = 0
        total_indicators = 0
        
        # 1. Moving Average Crossover
        if 'ema_8' in latest_df.columns and 'ema_21' in latest_df.columns:
            total_indicators += 1
            
            # Current values
            ema_8_current = latest['ema_8']
            ema_21_current = latest['ema_21']
            
            # Previous values
            ema_8_prev = latest_df.iloc[-2]['ema_8']
            ema_21_prev = latest_df.iloc[-2]['ema_21']
            
            # Check for crossover
            if ema_8_prev <= ema_21_prev and ema_8_current > ema_21_current:
                # Bullish crossover
                long_signals += 1
                print("EMA Crossover: BULLISH (8 EMA crossed above 21 EMA)")
            elif ema_8_prev >= ema_21_prev and ema_8_current < ema_21_current:
                # Bearish crossover
                short_signals += 1
                print("EMA Crossover: BEARISH (8 EMA crossed below 21 EMA)")
                
        # 2. RSI
        if 'rsi' in latest_df.columns:
            total_indicators += 1
            rsi = latest['rsi']
            
            if rsi < 30:
                # Oversold - bullish
                long_signals += 1
                print(f"RSI: BULLISH (Oversold at {rsi:.1f})")
            elif rsi > 70:
                # Overbought - bearish
                short_signals += 1
                print(f"RSI: BEARISH (Overbought at {rsi:.1f})")
                
        # 3. MACD
        if 'macd' in latest_df.columns and 'macd_signal' in latest_df.columns:
            total_indicators += 1
            
            # Current values
            macd_current = latest['macd']
            signal_current = latest['macd_signal']
            
            # Previous values
            macd_prev = latest_df.iloc[-2]['macd']
            signal_prev = latest_df.iloc[-2]['macd_signal']
            
            # Check for crossover
            if macd_prev <= signal_prev and macd_current > signal_current:
                # Bullish crossover
                long_signals += 1
                print(f"MACD: BULLISH (MACD crossed above signal line)")
            elif macd_prev >= signal_prev and macd_current < signal_current:
                # Bearish crossover
                short_signals += 1
                print(f"MACD: BEARISH (MACD crossed below signal line)")
                
        # 4. Bollinger Bands
        if 'bb_upper' in latest_df.columns and 'bb_lower' in latest_df.columns:
            total_indicators += 1
            
            close = latest['close']
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            
            # Previous close
            prev_close = latest_df.iloc[-2]['close']
            
            if prev_close <= bb_lower and close > bb_lower:
                # Price bouncing off lower band - bullish
                long_signals += 1
                print(f"Bollinger Bands: BULLISH (Price bounced off lower band)")
            elif prev_close >= bb_upper and close < bb_upper:
                # Price bouncing off upper band - bearish
                short_signals += 1
                print(f"Bollinger Bands: BEARISH (Price bounced off upper band)")
                
        # 5. Stochastic Oscillator
        if 'stoch_k' in latest_df.columns and 'stoch_d' in latest_df.columns:
            total_indicators += 1
            
            # Current values
            k_current = latest['stoch_k']
            d_current = latest['stoch_d']
            
            # Previous values
            k_prev = latest_df.iloc[-2]['stoch_k']
            d_prev = latest_df.iloc[-2]['stoch_d']
            
            if k_prev <= d_prev and k_current > d_current and k_current < 20:
                # Bullish crossover in oversold territory
                long_signals += 1
                print(f"Stochastic: BULLISH (K crossed above D in oversold territory)")
            elif k_prev >= d_prev and k_current < d_current and k_current > 80:
                # Bearish crossover in overbought territory
                short_signals += 1
                print(f"Stochastic: BEARISH (K crossed below D in overbought territory)")
                
        # 6. ADX for trend strength
        if 'adx' in latest_df.columns:
            adx = latest['adx']
            print(f"ADX: {adx:.1f} - {'Strong' if adx > 25 else 'Weak'} trend")
            
            # ADX doesn't give direction, just confirms strength of other signals
            if adx > 25:
                # Strong trend - boost existing signals
                long_signals = long_signals * 1.2 if long_signals > short_signals else long_signals
                short_signals = short_signals * 1.2 if short_signals > long_signals else short_signals
                
        # 7. Volume analysis
        if 'volume' in latest_df.columns:
            total_indicators += 1
            
            # Current volume
            current_volume = latest['volume']
            
            # Average volume (10 periods)
            avg_volume = latest_df['volume'].iloc[-10:].mean()
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume
            
            # Price change
            price_change = latest['close'] - latest_df.iloc[-2]['close']
            
            # High volume with price increase - bullish
            if volume_ratio > 1.5 and price_change > 0:
                long_signals += 1
                print(f"Volume: BULLISH (High volume with price increase, ratio: {volume_ratio:.2f})")
            # High volume with price decrease - bearish
            elif volume_ratio > 1.5 and price_change < 0:
                short_signals += 1
                print(f"Volume: BEARISH (High volume with price decrease, ratio: {volume_ratio:.2f})")
                
        # 8. Support/Resistance breakouts
        # This is a simplified version - in a real system you'd have more sophisticated S/R detection
        if len(latest_df) >= 20:
            total_indicators += 1
            
            # Find recent highs and lows
            recent_high = latest_df['high'].iloc[-20:].max()
            recent_low = latest_df['low'].iloc[-20:].min()
            
            current_close = latest['close']
            prev_close = latest_df.iloc[-2]['close']
            
            # Breakout above resistance
            if prev_close < recent_high and current_close > recent_high:
                long_signals += 1
                print(f"Support/Resistance: BULLISH (Breakout above resistance at {recent_high:.2f})")
            # Breakdown below support
            elif prev_close > recent_low and current_close < recent_low:
                short_signals += 1
                print(f"Support/Resistance: BEARISH (Breakdown below support at {recent_low:.2f})")
                
        # Calculate final signal
        # Require at least the threshold number of confirming indicators
        if long_signals >= self.signal_confirmation_threshold and long_signals > short_signals:
            signal = 1  # Long
            confidence = min(100, (long_signals / total_indicators) * 100)
        elif short_signals >= self.signal_confirmation_threshold and short_signals > long_signals:
            signal = -1  # Short
            confidence = min(100, (short_signals / total_indicators) * 100)
        else:
            signal = 0  # No clear signal
            confidence = 0
            
        # Check for signal cooldown period
        if signal != 0 and self.last_signal_time is not None:
            time_since_last_signal = (datetime.now() - self.last_signal_time).total_seconds() / 60
            if time_since_last_signal < self.signal_cooldown_minutes:
                print(f"Signal ignored: Cooldown period active ({time_since_last_signal:.1f} minutes since last signal)")
                return (0, 0)
                
        # Update last signal time if we have a valid signal
        if signal != 0:
            self.last_signal_time = datetime.now()
            
        # Print summary
        print(f"Enhanced Signal Analysis: Long indicators: {long_signals}, Short indicators: {short_signals}, Total: {total_indicators}")
        print(f"Final Signal: {'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'} with {confidence:.1f}% confidence")
        
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
            print(f"Using compound interest. Symbol balance: ${self.symbol_balance:.2f}")
            print(f"Initial investment: ${self.initial_investment:.2f}")
            print(f"Total symbol profit: ${self.total_symbol_profit:.2f}")
            print(f"Compound interest: {self.total_symbol_profit/self.initial_investment*100:.2f}%")
            
        # If enhanced signals are enabled, verify the signal with multiple indicators
        if self.use_enhanced_signals:
            # Get latest data for signal analysis
            latest_df = self.get_latest_data(lookback_candles=50)
            
            if latest_df is not None and len(latest_df) >= 50:
                enhanced_signal, confidence = self.generate_enhanced_signal(latest_df)
                
                # Only proceed if enhanced signal confirms the original signal
                if enhanced_signal != signal:
                    print(f"Signal rejected: Enhanced signal ({enhanced_signal}) does not match original signal ({signal})")
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
                if (signal == 1 and trend_direction != 1) or (signal == -1 and trend_direction != -1):
                    print(f"Signal {signal} ignored: against current market trend (direction: {trend_direction}, strength: {trend_strength:.1f}%)")
                    return None
                else:
                    print(f"Signal {signal} aligned with market trend (direction: {trend_direction}, strength: {trend_strength:.1f}%)")
            else:
                print(f"Weak market trend detected (strength: {trend_strength:.1f}%), proceeding with signal {signal}")

        # Determine position type based on signal
        position = 'long' if signal == 1 else 'short'
        
        # Calculate position size using dynamic sizing for better risk management
        position_size = self.calculate_dynamic_position_size(current_price)
        
        # Calculate stop loss and take profit prices
        if position == 'long':
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
            print(f"Symbol balance too low (${self.symbol_balance:.2f}), trading paused")
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
            print(f"Controlling position worth: {investment_amount * self.leverage:.2f} USD")
        elif self.use_full_investment:
            # Use almost full balance for position size (95% to leave room for fees)
            # If compound interest is enabled, use symbol balance, otherwise use initial investment
            if self.compound_interest:
                investment_amount = self.symbol_balance * 0.95
                print(f"Using full investment mode with compound interest: {investment_amount:.2f} USD for position (based on symbol balance)")
            else:
                investment_amount = self.initial_investment * 0.95
                print(f"Using full investment mode: {investment_amount:.2f} USD for position")
            position_size = investment_amount / current_price
        else:
            # Calculate maximum position size based on current balance
            max_position_value = self.symbol_balance * self.max_position_size if self.compound_interest else account_balance * self.max_position_size
            
            # Calculate position size based on risk per trade
            # If compound interest is enabled, calculate risk amount based on symbol balance
            # Otherwise, use initial investment
            if self.compound_interest:
                risk_amount = self.symbol_balance * self.risk_per_trade
                print(f"Using compound interest: Risk amount ${risk_amount:.2f} based on symbol balance ${self.symbol_balance:.2f}")
            else:
                risk_amount = self.initial_investment * self.risk_per_trade
                print(f"Using fixed investment: Risk amount ${risk_amount:.2f} based on initial investment ${self.initial_investment:.2f}")
            
            # Get ATR for dynamic stop loss and take profit
            latest_df = self.get_latest_data(lookback_candles=20)
            atr = latest_df['ATR'].iloc[-1]
            
            if signal == 1:  # BUY signal
                # Use ATR for stop loss (1.0x ATR instead of 1.5x)
                self.stop_loss_pct = min(0.01, (1.0 * atr) / current_price)  # Cap at 1% instead of 3%
                # Use ATR for take profit (2x ATR instead of 3x)
                self.take_profit_pct = min(0.06, (2 * atr) / current_price)  # Cap at 6% instead of 10%
                
                # Calculate position size based on risk
                position_size = (risk_amount / self.stop_loss_pct) / current_price
                
                # Calculate stop loss and take profit prices
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.take_profit_price = current_price * (1 + self.take_profit_pct)
                
                # Debug logging for stop loss and take profit calculation
                print(f"DEBUG - Calculated stop loss: ${self.stop_loss_price:.2f} ({self.stop_loss_pct*100:.2f}% below entry)")
                print(f"DEBUG - Calculated take profit: ${self.take_profit_price:.2f} ({self.take_profit_pct*100:.2f}% above entry)")
                
            elif signal == -1:  # SELL signal
                # Use ATR for stop loss (1.0x ATR instead of 1.5x)
                self.stop_loss_pct = min(0.01, (1.0 * atr) / current_price)  # Cap at 1% instead of 3%
                # Use ATR for take profit (2x ATR instead of 3x)
                self.take_profit_pct = min(0.06, (2 * atr) / current_price)  # Cap at 6% instead of 10%
                
                # Calculate position size based on risk
                position_size = (risk_amount / self.stop_loss_pct) / current_price
                
                # Calculate stop loss and take profit prices
                self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
                self.take_profit_price = current_price * (1 - self.take_profit_pct)
                
                # Debug logging for stop loss and take profit calculation
                print(f"DEBUG - Calculated stop loss: ${self.stop_loss_price:.2f} ({self.stop_loss_pct*100:.2f}% above entry)")
                print(f"DEBUG - Calculated take profit: ${self.take_profit_price:.2f} ({self.take_profit_pct*100:.2f}% below entry)")
            
            # Ensure position size doesn't exceed maximum allowed
            position_value = position_size * current_price
            if position_value > max_position_value:
                position_size = max_position_value / current_price
        
        # Format the position size with the correct precision using the direct method
        position_size = float("{:0.0{}f}".format(position_size, quantity_precision))
        
        # Ensure position size meets minimum quantity requirement
        if position_size < min_quantity:
            print(f"Warning: Position size {position_size} is below minimum quantity {min_quantity}")
            position_size = min_quantity
        
        # Ensure order value meets minimum notional requirement
        order_value = position_size * current_price
        if order_value < min_notional:
            print(f"Warning: Order value ${order_value:.2f} is below minimum notional ${min_notional}")
            # Calculate the minimum position size needed to meet the minimum notional value
            min_position_size = min_notional / current_price
            # Format to the correct precision
            min_position_size = float("{:0.0{}f}".format(min_position_size, quantity_precision))
            # Ensure it's at least the minimum quantity
            min_position_size = max(min_position_size, min_quantity)
            
            print(f"Adjusting position size from {position_size} to {min_position_size} to meet minimum requirements")
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
                    print(f"Opening LONG position at MARKET price for {position_size} {self.symbol}")
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side='BUY',
                        type='MARKET',  # MARKET order type ensures execution at current market price
                        quantity=position_size
                    )
                    print(f"Order executed: {order}")
                else:
                    # Test mode - simulate order
                    print(f"TEST MODE: Simulating BUY order for {position_size} {self.symbol} at MARKET price (${current_price:.{price_precision}f})")
                
                # Update position tracking
                self.position = 'long'
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
                        stop_loss_price = float("{:0.0{}f}".format(self.stop_loss_price, price_precision))
                        take_profit_price = float("{:0.0{}f}".format(self.take_profit_price, price_precision))
                        
                        # For LONG positions: SELL to close
                        # For SHORT positions: BUY to close
                        close_side = 'SELL' if signal == 1 else 'BUY'
                        
                        # Place stop loss order
                        stop_loss_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type='STOP_MARKET',
                            stopPrice=stop_loss_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC"  # Good Till Cancelled
                        )
                        self.stop_loss_order_id = stop_loss_order['orderId']
                        print(f"Stop loss order placed: {stop_loss_order}")
                        
                        # Place take profit order
                        take_profit_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type='TAKE_PROFIT_MARKET',
                            stopPrice=take_profit_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC"  # Good Till Cancelled
                        )
                        self.take_profit_order_id = take_profit_order['orderId']
                        print(f"Take profit order placed: {take_profit_order}")
                    except BinanceAPIException as order_error:
                        print(f"Error placing stop loss or take profit orders: {order_error}")
                        # Continue with the trade even if setting the orders fails
                
                # Add to trade history
                trade_record = {
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'value': position_size * current_price,
                    'stop_loss': self.stop_loss_price,
                    'take_profit': self.take_profit_price,
                    'order_type': 'MARKET'  # Record that this was a market order
                }
                
                self.trade_history.append(trade_record)
                if self.test_mode:
                    self.test_trades.append(trade_record)
                
                return f"BUY: Opened long position at MARKET price (${current_price:.{price_precision}f}) with {position_size} units"
                
            elif signal == -1:  # SELL signal
                if not self.test_mode:
                    # For futures trading - open short position at MARKET price (executes immediately at current market price)
                    print(f"Opening SHORT position at MARKET price for {position_size} {self.symbol}")
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side='SELL',
                        type='MARKET',  # MARKET order type ensures execution at current market price
                        quantity=position_size
                    )
                    print(f"Order executed: {order}")
                else:
                    # Test mode - simulate order
                    print(f"TEST MODE: Simulating SELL order for {position_size} {self.symbol} at MARKET price (${current_price:.{price_precision}f})")
                
                # Update position tracking
                self.position = 'short'
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
                        stop_loss_price = float("{:0.0{}f}".format(self.stop_loss_price, price_precision))
                        take_profit_price = float("{:0.0{}f}".format(self.take_profit_price, price_precision))
                        
                        # For LONG positions: SELL to close
                        # For SHORT positions: BUY to close
                        close_side = 'SELL' if signal == 1 else 'BUY'
                        
                        # Place stop loss order
                        stop_loss_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type='STOP_MARKET',
                            stopPrice=stop_loss_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC"  # Good Till Cancelled
                        )
                        self.stop_loss_order_id = stop_loss_order['orderId']
                        print(f"Stop loss order placed: {stop_loss_order}")
                        
                        # Place take profit order
                        take_profit_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type='TAKE_PROFIT_MARKET',
                            stopPrice=take_profit_price,
                            reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                            quantity=position_size,
                            workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                            timeInForce="GTC"  # Good Till Cancelled
                        )
                        self.take_profit_order_id = take_profit_order['orderId']
                        print(f"Take profit order placed: {take_profit_order}")
                    except BinanceAPIException as order_error:
                        print(f"Error placing stop loss or take profit orders: {order_error}")
                        # Continue with the trade even if setting the orders fails
                
                # Add to trade history
                trade_record = {
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': current_price,
                    'size': position_size,
                    'value': position_size * current_price,
                    'stop_loss': self.stop_loss_price,
                    'take_profit': self.take_profit_price,
                    'order_type': 'MARKET'  # Record that this was a market order
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
                if futures_info and 'quantityPrecision' in futures_info:
                    adjusted_precision = futures_info['quantityPrecision']
                    adjusted_position_size = float("{:0.0{}f}".format(position_size, adjusted_precision))
                    print(f"Adjusted position size from {position_size} to {adjusted_position_size} using futures precision")
                    
                    # Ensure adjusted position size is greater than zero
                    if adjusted_position_size <= 0:
                        adjusted_position_size = float("{:0.0{}f}".format(min_quantity, adjusted_precision))
                        print(f"Adjusted position size was zero, setting to minimum quantity: {adjusted_position_size}")
                    
                    # Try again with the adjusted position size
                    if not self.test_mode:
                        try:
                            side = 'BUY' if signal == 1 else 'SELL'
                            print(f"Retrying with adjusted position size: {adjusted_position_size}")
                            order = self.client.futures_create_order(
                                symbol=self.symbol,
                                side=side,
                                type='MARKET',
                                quantity=adjusted_position_size
                            )
                            print(f"Order executed: {order}")
                            
                            # Update position tracking
                            self.position = 'long' if signal == 1 else 'short'
                            self.entry_price = current_price
                            self.position_size = adjusted_position_size
                            self.entry_time = timestamp
                            
                            # Set stop loss and take profit prices
                            self.stop_loss_price = current_price * (1 - self.stop_loss_pct) if signal == 1 else current_price * (1 + self.stop_loss_pct)
                            self.take_profit_price = current_price * (1 + self.take_profit_pct) if signal == 1 else current_price * (1 - self.take_profit_pct)
                            
                            # Place actual stop loss and take profit orders in Binance
                            if not self.test_mode:
                                try:
                                    # Format prices with correct precision
                                    price_precision = self.get_price_precision()
                                    stop_loss_price = float("{:0.0{}f}".format(self.stop_loss_price, price_precision))
                                    take_profit_price = float("{:0.0{}f}".format(self.take_profit_price, price_precision))
                                    
                                    # For LONG positions: SELL to close
                                    # For SHORT positions: BUY to close
                                    close_side = 'SELL' if signal == 1 else 'BUY'
                                    
                                    # Place stop loss order
                                    stop_loss_order = self.client.futures_create_order(
                                        symbol=self.symbol,
                                        side=close_side,
                                        type='STOP_MARKET',
                                        stopPrice=stop_loss_price,
                                        reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                        quantity=adjusted_position_size,
                                        workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                        timeInForce="GTC"  # Good Till Cancelled
                                    )
                                    self.stop_loss_order_id = stop_loss_order['orderId']
                                    print(f"Stop loss order placed: {stop_loss_order}")
                                    
                                    # Place take profit order
                                    take_profit_order = self.client.futures_create_order(
                                        symbol=self.symbol,
                                        side=close_side,
                                        type='TAKE_PROFIT_MARKET',
                                        stopPrice=take_profit_price,
                                        reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                        quantity=adjusted_position_size,
                                        workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                        timeInForce="GTC"  # Good Till Cancelled
                                    )
                                    self.take_profit_order_id = take_profit_order['orderId']
                                    print(f"Take profit order placed: {take_profit_order}")
                                except BinanceAPIException as order_error:
                                    print(f"Error placing stop loss or take profit orders: {order_error}")
                                    # Continue with the trade even if setting the orders fails
                            
                            # Add to trade history
                            trade_record = {
                                'timestamp': timestamp,
                                'action': 'BUY' if signal == 1 else 'SELL',
                                'price': current_price,
                                'size': adjusted_position_size,
                                'value': adjusted_position_size * current_price,
                                'stop_loss': self.stop_loss_price,
                                'take_profit': self.take_profit_price,
                                'order_type': 'MARKET'
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
                print("Error: Position size is zero or negative. Trying with minimum quantity...")
                
                # Try with minimum quantity
                min_valid_quantity = min_quantity
                # Ensure it meets minimum notional requirement
                if min_valid_quantity * current_price < min_notional:
                    min_valid_quantity = min_notional / current_price
                
                print(f"Retrying with minimum valid quantity: {min_valid_quantity}")
                
                if not self.test_mode:
                    try:
                        side = 'BUY' if signal == 1 else 'SELL'
                        order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=side,
                            type='MARKET',
                            quantity=min_valid_quantity
                        )
                        print(f"Order executed: {order}")
                        
                        # Update position tracking
                        self.position = 'long' if signal == 1 else 'short'
                        self.entry_price = current_price
                        self.position_size = min_valid_quantity
                        self.entry_time = timestamp
                        
                        # Set stop loss and take profit prices
                        self.stop_loss_price = current_price * (1 - self.stop_loss_pct) if signal == 1 else current_price * (1 + self.stop_loss_pct)
                        self.take_profit_price = current_price * (1 + self.take_profit_pct) if signal == 1 else current_price * (1 - self.take_profit_pct)
                        
                        # Place actual stop loss and take profit orders in Binance
                        if not self.test_mode:
                            try:
                                # Format prices with correct precision
                                price_precision = self.get_price_precision()
                                stop_loss_price = float("{:0.0{}f}".format(self.stop_loss_price, price_precision))
                                take_profit_price = float("{:0.0{}f}".format(self.take_profit_price, price_precision))
                                
                                # For LONG positions: SELL to close
                                # For SHORT positions: BUY to close
                                close_side = 'SELL' if signal == 1 else 'BUY'
                                
                                # Place stop loss order
                                stop_loss_order = self.client.futures_create_order(
                                    symbol=self.symbol,
                                    side=close_side,
                                    type='STOP_MARKET',
                                    stopPrice=stop_loss_price,
                                    reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                    quantity=min_valid_quantity,
                                    workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                    timeInForce="GTC"  # Good Till Cancelled
                                )
                                self.stop_loss_order_id = stop_loss_order['orderId']
                                print(f"Stop loss order placed: {stop_loss_order}")
                                
                                # Place take profit order
                                take_profit_order = self.client.futures_create_order(
                                    symbol=self.symbol,
                                    side=close_side,
                                    type='TAKE_PROFIT_MARKET',
                                    stopPrice=take_profit_price,
                                    reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                                    quantity=min_valid_quantity,
                                    workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                                    timeInForce="GTC"  # Good Till Cancelled
                                )
                                self.take_profit_order_id = take_profit_order['orderId']
                                print(f"Take profit order placed: {take_profit_order}")
                            except BinanceAPIException as order_error:
                                print(f"Error placing stop loss or take profit orders: {order_error}")
                                # Continue with the trade even if setting the orders fails
                        
                        # Add to trade history
                        trade_record = {
                            'timestamp': timestamp,
                            'action': 'BUY' if signal == 1 else 'SELL',
                            'price': current_price,
                            'size': min_valid_quantity,
                            'value': min_valid_quantity * current_price,
                            'stop_loss': self.stop_loss_price,
                            'take_profit': self.take_profit_price,
                            'order_type': 'MARKET'
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
            pyramid_position_size = float("{:0.0{}f}".format(pyramid_position_size, quantity_precision))
            
            # Ensure pyramid size is greater than zero
            if pyramid_position_size <= 0:
                print("Error: Pyramid position size is zero or negative.")
                return
                
            # Execute the pyramid entry
            if not self.test_mode:
                # For LONG positions: BUY to add
                # For SHORT positions: SELL to add
                side = 'BUY' if self.position == 'long' else 'SELL'
                print(f"Executing pyramid entry: {side} {pyramid_position_size} {self.symbol} at MARKET price (${current_price:.2f})")
                
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=pyramid_position_size
                )
                print(f"Pyramid entry order executed: {order}")
            else:
                # Test mode - simulate order
                side = 'BUY' if self.position == 'long' else 'SELL'
                print(f"TEST MODE: Simulating pyramid entry: {side} {pyramid_position_size} {self.symbol} at MARKET price (${current_price:.2f})")
            
            # Store pyramid entry details
            self.pyramid_entries += 1
            self.pyramid_entry_prices.append(current_price)
            self.pyramid_position_sizes.append(pyramid_position_size)
            
            # Update average entry price and total position size
            total_position_value = (self.entry_price * self.position_size) + (current_price * pyramid_position_size)
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
                'timestamp': timestamp,
                'action': 'PYRAMID_ENTRY',
                'price': current_price,
                'size': pyramid_position_size,
                'value': pyramid_position_size * current_price,
                'old_entry_price': old_entry_price,
                'new_entry_price': new_entry_price,
                'old_position_size': old_position_size,
                'new_position_size': new_position_size,
                'pyramid_count': self.pyramid_entries
            }
            
            self.trade_history.append(pyramid_entry_record)
            if self.test_mode:
                self.test_trades.append(pyramid_entry_record)
            
            # Send notification
            message = (
                f"🔺 Pyramid Entry #{self.pyramid_entries}\n"
                f"Symbol: {self.symbol}\n"
                f"Type: {self.position.upper()}\n"
                f"Entry Price: ${current_price:.2f}\n"
                f"Added Size: {pyramid_position_size} units\n"
                f"New Position Size: {new_position_size} units\n"
                f"New Average Entry: ${new_entry_price:.2f}\n"
                f"Account Balance: ${self.get_account_balance():.2f}\n"
                f"Symbol Balance: ${self.symbol_balance:.2f}"
            )
            self.send_notification(message)
            
            print(f"✅ Pyramid entry #{self.pyramid_entries} executed successfully. New position size: {new_position_size} units with average entry: ${new_entry_price:.2f}")
            
        except BinanceAPIException as e:
            print(f"❌ Error executing pyramid entry: {e}")

    def close_position(self, current_price, timestamp, reason='manual'):
        """Close the current position"""
        if not self.has_open_position():
            return None

        # Calculate profit/loss
        if self.position == 'long':
            profit = self.position_size * (current_price - self.entry_price)
            profit_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # short
            profit = self.position_size * (self.entry_price - current_price)
            profit_pct = (self.entry_price - current_price) / self.entry_price * 100

        # Update test balance in test mode
        if self.test_mode:
            self.test_balance += profit
            print(f"TEST MODE: Updated balance to ${self.test_balance:.2f} (profit: ${profit:.2f})")

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
                close_side = 'SELL' if self.position == 'long' else 'BUY'

                # Execute market order to close position
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=close_side,
                    type='MARKET',
                    quantity=self.position_size,
                    reduceOnly="true"  # Must be string "true" or "false" per Binance docs
                )
                print(f"Position closed with order: {order}")
            except BinanceAPIException as e:
                print(f"Error closing position: {e}")

        # Add to trade history
        close_record = {
            'timestamp': timestamp,
            'action': 'CLOSE',
            'price': current_price,
            'size': self.position_size,
            'value': self.position_size * current_price,
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason,
            'order_type': 'MARKET',
            'pyramid_entries': self.pyramid_entries
        }

        self.trade_history.append(close_record)
        if self.test_mode:
            self.test_trades.append(close_record)

        # Update daily profits
        day_key = timestamp.strftime('%Y-%m-%d')
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
                if trade.get('reason') == 'PARTIAL_TAKE_PROFIT':
                    partial_profit = trade.get('profit', 0)
                    partial_price = trade.get('price', 0)
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
            "pyramid_entries": self.pyramid_entries
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
                os.path.join(self.results_dir, f"{mode_prefix}{self.symbol}_trade_history.csv"),
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
                os.path.join(self.results_dir, f"{mode_prefix}{self.symbol}_daily_profits.csv"),
                index=False,
            )

        # Calculate days that met target
        days_met_target = sum(1 for profit in self.daily_profits.values() if profit >= self.daily_profit_target)
        
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

    def update_stop_loss_order(self):
        """Update the stop loss order in Binance with the new stop loss price"""
        if self.test_mode or not self.has_open_position() or not self.stop_loss_order_id or self.stop_loss_price is None:
            if self.stop_loss_price is None:
                print("Cannot update stop loss order: stop_loss_price is None")
            elif not self.stop_loss_order_id:
                print("Cannot update stop loss order: stop_loss_order_id is None, will recreate instead")
                self.recreate_stop_loss_order()
            elif not self.has_open_position():
                print("Cannot update stop loss order: no open position")
            elif self.test_mode:
                print("Cannot update stop loss order: in test mode")
            return
            
        try:
            print(f"DEBUG - Updating stop loss order: ID={self.stop_loss_order_id}, Current price={self.stop_loss_price}")
            
            # Cancel the existing stop loss order
            try:
                cancel_result = self.client.futures_cancel_order(
                    symbol=self.symbol,
                    orderId=self.stop_loss_order_id
                )
                print(f"Cancelled existing stop loss order: {self.stop_loss_order_id}, Result: {cancel_result}")
            except BinanceAPIException as cancel_error:
                print(f"❌ Error cancelling stop loss order: {cancel_error}")
                if "Unknown order" in str(cancel_error):
                    print(f"Order {self.stop_loss_order_id} already cancelled or does not exist. Will create new order.")
                    self.stop_loss_order_id = None
                    self.recreate_stop_loss_order()
                    return
                else:
                    # Re-raise for other errors
                    raise
            
            # Format price with correct precision
            price_precision = self.get_price_precision()
            stop_loss_price = float("{:0.0{}f}".format(self.stop_loss_price, price_precision))
            
            # For LONG positions: SELL to close
            # For SHORT positions: BUY to close
            close_side = 'SELL' if self.position == 'long' else 'BUY'
            
            print(f"Creating new stop loss order: {self.symbol} {close_side} at {stop_loss_price}, Size: {self.position_size}")
            
            # Place new stop loss order
            stop_loss_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='STOP_MARKET',
                stopPrice=stop_loss_price,
                reduceOnly="true",  # Must be string "true" or "false" per Binance docs
                quantity=self.position_size,
                workingType="MARK_PRICE",  # Use MARK_PRICE for more reliable triggering
                timeInForce="GTC"  # Good Till Cancelled
            )
            self.stop_loss_order_id = stop_loss_order['orderId']
            print(f"✅ New trailing stop loss order placed: {stop_loss_order}")
            
        except BinanceAPIException as e:
            print(f"❌ Error updating stop loss order: {e}")
            print(f"Error code: {e.code}, Error message: {e.message}")
            print(f"Error details: Symbol={self.symbol}, Position={self.position}, Size={self.position_size}, Stop Price={self.stop_loss_price}")
            
            # If we get an "Unknown order" error, reset the order ID and try to recreate
            if "Unknown order" in str(e):
                print("Resetting stop_loss_order_id and recreating order")
                self.stop_loss_order_id = None
                self.recreate_stop_loss_order()
        except Exception as e:
            print(f"❌ Unexpected error updating stop loss order: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: Symbol={self.symbol}, Position={self.position}, Size={self.position_size}, Stop Price={self.stop_loss_price}")


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
    print(f"Daily profit target: ${realtime_trader.daily_profit_target:.2f}")

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
        f"Daily Profit Target: ${realtime_trader.daily_profit_target:.2f}\n"
        f"Duration: {duration_hours} hours\n"
        f"Update Interval: {update_interval_minutes} minutes\n"
        f"Enhanced Signals: {'Enabled' if realtime_trader.use_enhanced_signals else 'Disabled'}\n"
        f"Trend Following: {'Enabled' if realtime_trader.trend_following_mode else 'Disabled'}\n"
        f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Track the current day for daily resets
    current_day = datetime.now().date()

    # Main trading loop
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            print(f"\n=== Update: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            # Check if day has changed and reset daily tracking
            if current_time.date() != current_day:
                realtime_trader.daily_loss = 0
                realtime_trader.trading_disabled = False
                current_day = current_time.date()
                realtime_trader.last_reset_day = current_day
                
                # Reset daily profit tracking for the new day
                day_key = current_time.strftime('%Y-%m-%d')
                if not hasattr(realtime_trader, 'daily_profits'):
                    realtime_trader.daily_profits = {}
                realtime_trader.daily_profits[day_key] = 0
                
                print(f"New day started. Daily tracking reset.")
                
                # Send notification about new day
                realtime_trader.send_notification(
                    f"📅 NEW TRADING DAY STARTED\n"
                    f"Symbol: {realtime_trader.symbol}\n"
                    f"Date: {day_key}\n"
                    f"Daily Profit Target: ${realtime_trader.daily_profit_target:.2f}\n"
                    f"Current Balance: ${realtime_trader.get_account_balance():.2f}\n"
                    f"Symbol Balance: ${realtime_trader.symbol_balance:.2f}"
                )

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
                    if "TAKE PROFIT" in tp_sl_result or "STOP LOSS" in tp_sl_result or "DAILY TARGET" in tp_sl_result:
                        # Extract position type and profit/loss from result
                        position_type = "LONG" if "LONG" in tp_sl_result else "SHORT"
                        
                        if "DAILY TARGET" in tp_sl_result:
                            reason = "DAILY TARGET"
                        elif "TAKE PROFIT" in tp_sl_result:
                            reason = "TAKE PROFIT"
                        else:
                            reason = "STOP LOSS"

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
                                f"Partial TP: {realtime_trader.partial_tp_pct*100:.1f}% ({realtime_trader.partial_tp_size*100:.0f}% of position)\n"
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
                        # Format stop loss and take profit values
                        stop_loss_str = 'N/A' if position_info['stop_loss'] is None else '$%.2f' % position_info['stop_loss']
                        take_profit_str = 'N/A' if position_info['take_profit'] is None else '$%.2f' % position_info['take_profit']
                        
                        # Add partial take profit status
                        partial_tp_status = "EXECUTED" if position_info['partial_tp_executed'] else "PENDING"
                        partial_tp_info = f"{position_info['partial_tp_pct']*100:.1f}% ({position_info['partial_tp_size']*100:.0f}% of position)"
                        
                        # Show original position size if partial TP executed
                        position_size_info = position_info['position_size']
                        if position_info['partial_tp_executed']:
                            position_size_info = f"{position_size_info:.6f} (original: {position_info['original_position_size']:.6f})"
                        else:
                            position_size_info = f"{position_size_info:.6f}"
                        
                        update_message = (
                            f"{emoji} POSITION UPDATE ({position_info['position'].upper()})\n"
                            f"Symbol: {realtime_trader.symbol}\n"
                            f"Current Price: ${position_info['current_price']:.2f}\n"
                            f"Entry Price: ${position_info['entry_price']:.2f}\n"
                            f"Position Size: {position_size_info}\n"
                            f"Unrealized P/L: ${profit_loss:.2f} ({profit_pct:.2f}%)\n"
                            f"Stop Loss: {stop_loss_str}\n"
                            f"Take Profit: {take_profit_str}\n"
                            f"Partial TP: {partial_tp_info} - {partial_tp_status}"
                        )
                        realtime_trader.send_notification(update_message)
                else:
                    print("Position info not available, will retry next update")

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
