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
        leverage=15,
        test_mode=False,
        use_full_investment=False,
        use_full_margin=False,
    ):
        """
        Initialize the real-time trader

        Args:
            symbol: Trading pair to trade
            initial_investment: Starting capital in USD
            daily_profit_target: Target profit per day in USD
            leverage: Margin trading leverage (15x-20x)
            test_mode: If True, run in test mode with fake balance
            use_full_investment: If True, use full investment for position size
            use_full_margin: If True, use full investment as margin (very high risk)
        """
        self.symbol = symbol
        self.initial_investment = initial_investment
        self.daily_profit_target = daily_profit_target
        self.test_mode = test_mode
        self.use_full_investment = use_full_investment
        self.use_full_margin = use_full_margin

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
        self.max_position_size = 0.95 if use_full_investment else 0.2  # Use 95% of balance if full investment mode
        self.max_daily_loss = 0.1  # Maximum 10% daily loss of initial investment
        self.risk_per_trade = 0.95 if use_full_investment else 0.02  # Use 95% of balance for risk if full investment mode
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
        self.stop_loss_pct = 0.02  # Default 2%
        self.take_profit_pct = 0.04  # Default 4%

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

        # Check for take profit (with null check)
        if self.position == "long" and self.take_profit_price is not None and current_price >= self.take_profit_price:
            result = self.close_position(current_price, timestamp, "TAKE PROFIT")
        elif self.position == "short" and self.take_profit_price is not None and current_price <= self.take_profit_price:
            result = self.close_position(current_price, timestamp, "TAKE PROFIT")

        # Check for stop loss (with null check)
        elif self.position == "long" and self.stop_loss_price is not None and current_price <= self.stop_loss_price:
            result = self.close_position(current_price, timestamp, "STOP LOSS")
        elif self.position == "short" and self.stop_loss_price is not None and current_price >= self.stop_loss_price:
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
        
        # Get the correct precision for the quantity and price
        quantity_precision = self.get_quantity_precision()
        price_precision = self.get_price_precision()
        min_quantity = self.get_min_quantity()
        min_notional = self.get_min_notional()
        
        # Calculate position size based on full investment, full margin, or risk management
        if self.use_full_margin:
            # Calculate position size to use full investment as margin
            # Formula: position_size = (investment * leverage) / current_price
            position_size = (self.initial_investment * self.leverage) / current_price
            print(f"Using FULL MARGIN mode: {self.initial_investment:.2f} USD as margin")
            print(f"Controlling position worth: {self.initial_investment * self.leverage:.2f} USD")
        elif self.use_full_investment:
            # Use almost full balance for position size (95% to leave room for fees)
            position_size = (account_balance * 0.95) / current_price
            print(f"Using full investment mode: {account_balance * 0.95:.2f} USD for position")
        else:
            # Calculate maximum position size based on current balance
            max_position_value = account_balance * self.max_position_size
            
            # Calculate position size based on risk per trade
            risk_amount = account_balance * self.risk_per_trade
            
            # Get ATR for dynamic stop loss and take profit
            latest_df = self.get_latest_data(lookback_candles=20)
            atr = latest_df['ATR'].iloc[-1]
            
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
                    min_valid_quantity = float("{:0.0{}f}".format(min_valid_quantity, quantity_precision))
                
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

    def close_position(self, current_price, timestamp, reason='manual'):
        """Close the current position"""
        if not self.has_open_position():
            return None
        
        try:
            # Get the correct precision for the quantity and price
            quantity_precision = self.get_quantity_precision()
            price_precision = self.get_price_precision()
            
            # Format the position size with the correct precision using the direct method
            position_size = float("{:0.0{}f}".format(self.position_size, quantity_precision))
            
            if not self.test_mode:
                # For futures trading - close position at MARKET price (executes immediately at current market price)
                side = 'SELL' if self.position == 'long' else 'BUY'
                print(f"Closing {self.position.upper()} position at MARKET price for {position_size} {self.symbol}")
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',  # MARKET order type ensures execution at current market price
                    quantity=position_size
                )
                print(f"Order executed: {order}")
            else:
                # Test mode - simulate order
                side = 'SELL' if self.position == 'long' else 'BUY'
                print(f"TEST MODE: Simulating {side} order to close {self.position} position for {position_size} {self.symbol} at MARKET price (${current_price:.{price_precision}f})")
            
            # Calculate profit/loss
            if self.position == 'long':
                profit = (current_price - self.entry_price) * self.position_size
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
            else:  # short
                profit = (self.entry_price - current_price) * self.position_size
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100
            
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
                'order_type': 'MARKET'  # Record that this was a market order
            }
            
            self.trade_history.append(close_record)
            if self.test_mode:
                self.test_trades.append(close_record)
            
            # Send notification
            message = (
                f"🔔 Position closed: {reason}\n"
                f"Symbol: {self.symbol}\n"
                f"Type: {self.position.upper()}\n"
                f"Entry: ${self.entry_price:.2f}\n"
                f"Exit: ${current_price:.2f}\n"
                f"Profit: ${profit:.2f} ({profit_pct:.2f}%)\n"
                f"Balance: ${self.get_account_balance():.2f}"
            )
            self.send_notification(message)
            
            # Reset position tracking
            self.position = None
            self.entry_price = None
            self.position_size = 0  # Set to 0 instead of None
            self.entry_time = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            # Return result
            result = {
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': reason,
                'order_type': 'MARKET'  # Include the order type in the result
            }
            
            return result
            
        except BinanceAPIException as e:
            error_message = str(e)
            print(f"Error closing position: {error_message}")
            
            # Handle precision errors specifically
            if "Precision is over the maximum defined for this asset" in error_message:
                print("Precision error detected when closing position. Trying to adjust position size...")
                
                # Try to get futures precision directly
                futures_info = self.get_futures_symbol_info()
                if futures_info and 'quantityPrecision' in futures_info:
                    adjusted_precision = futures_info['quantityPrecision']
                    adjusted_position_size = float("{:0.0{}f}".format(self.position_size, adjusted_precision))
                    print(f"Adjusted position size from {self.position_size} to {adjusted_position_size} using futures precision")
                    
                    # Update position size and try again
                    self.position_size = adjusted_position_size
                    return self.close_position(current_price, timestamp, reason)
            
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
                            f"Current Price: ${position_info['current_price']:,.2f}\n"
                            f"Entry Price: ${position_info['entry_price']:,.2f}\n"
                            f"Unrealized P/L: ${profit_loss:,.2f} ({profit_pct:.2f}%)\n"
                            f"Stop Loss: ${position_info['stop_loss'] if position_info['stop_loss'] is not None else 'N/A':,.2f}\n"
                            f"Take Profit: ${position_info['take_profit'] if position_info['take_profit'] is not None else 'N/A':,.2f}"
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
