"""
Order execution module for handling trade execution.
"""

import time
import math
from datetime import datetime, timedelta
from binance.exceptions import BinanceAPIException
from ..utils.indicators import calculate_atr
from ..utils.reporting import display_countdown, format_time_remaining

class OrderExecutor:
    def __init__(self, trader):
        """
        Initialize the OrderExecutor with a reference to the trader
        
        Args:
            trader: The RealtimeTrader instance
        """
        self.trader = trader
        self.client = trader.client
        self.symbol = trader.symbol
        self.leverage = trader.leverage
        self.test_mode = trader.test_mode
        self.use_dynamic_take_profit = trader.use_dynamic_take_profit
        self.fixed_tp = trader.fixed_tp
        self.fixed_sl = trader.fixed_sl
        self.last_countdown_time = None
        self.next_run_time = None
    
    def execute_trade(self, signal, current_price, timestamp):
        """
        Execute a trade based on the given signal
        
        Args:
            signal: Trading signal ("LONG" or "SHORT")
            current_price: Current price of the trading symbol
            timestamp: Current timestamp
            
        Returns:
            bool: True if trade was executed, False otherwise
        """
        if signal not in ["LONG", "SHORT"]:
            print(f"Invalid signal: {signal}")
            return False
        
        # Check if there's already an open position
        if self.trader.position_manager.has_open_position():
            print("Cannot execute trade: Position already open")
            return False
        
        # Calculate position size
        position_size = self.calculate_position_size(current_price)
        
        if position_size <= 0:
            print("Cannot execute trade: Invalid position size")
            return False
        
        # Get quantity precision
        quantity_precision = self.trader.account_manager.get_quantity_precision()
        quantity = round(position_size / current_price, quantity_precision)
        
        # Ensure minimum quantity
        min_qty = self.trader.account_manager.get_min_quantity()
        if quantity < min_qty:
            print(f"Cannot execute trade: Quantity {quantity} is below minimum {min_qty}")
            return False
        
        # Determine order side
        order_side = "BUY" if signal == "LONG" else "SELL"
        
        try:
            # Set leverage
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            
            # Place market order
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type="MARKET",
                quantity=quantity
            )
            
            print(f"🟢 {signal} POSITION OPENED")
            print(f"Bought {quantity} {self.symbol} at {current_price:.2f}")
            
            # Record the trade
            self.trader.trading_history.append({
                "timestamp": timestamp,
                "action": f"OPEN_{signal}",
                "price": current_price,
                "quantity": quantity,
                "reason": "signal"
            })
            
            # Send notification
            self.trader.send_notification(
                f"🟢 {signal} POSITION OPENED on {self.symbol}\n"
                f"Quantity: {quantity}\n"
                f"Price: {current_price:.2f}\n"
                f"Leverage: {self.leverage}x"
            )
            
            # Reset pyramid entries counter
            self.trader.pyramid_entries = 0
            
            # Set up take profit and stop loss orders if not using dynamic TP/SL
            if not self.use_dynamic_take_profit and (self.fixed_tp > 0 or self.fixed_sl > 0):
                self.place_take_profit_stop_loss_orders(signal, current_price, quantity)
            
            return True
            
        except BinanceAPIException as e:
            print(f"Error executing trade: {e}")
            return False
    
    def place_take_profit_stop_loss_orders(self, signal, entry_price, quantity):
        """
        Place take profit and stop loss orders
        
        Args:
            signal: Trading signal ("LONG" or "SHORT")
            entry_price: Entry price of the position
            quantity: Position quantity
            
        Returns:
            bool: True if orders were placed, False otherwise
        """
        # Get price precision
        price_precision = self.trader.account_manager.get_price_precision()
        
        # Calculate take profit and stop loss prices
        if signal == "LONG":
            if self.fixed_tp > 0:
                tp_price = round(entry_price * (1 + self.fixed_tp / (100 * self.leverage)), price_precision)
            else:
                tp_price = None
                
            if self.fixed_sl > 0:
                sl_price = round(entry_price * (1 - self.fixed_sl / (100 * self.leverage)), price_precision)
            else:
                sl_price = None
        else:  # SHORT
            if self.fixed_tp > 0:
                tp_price = round(entry_price * (1 - self.fixed_tp / (100 * self.leverage)), price_precision)
            else:
                tp_price = None
                
            if self.fixed_sl > 0:
                sl_price = round(entry_price * (1 + self.fixed_sl / (100 * self.leverage)), price_precision)
            else:
                sl_price = None
        
        # Place take profit order
        if tp_price:
            try:
                tp_side = "SELL" if signal == "LONG" else "BUY"
                
                tp_order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=tp_side,
                    type="LIMIT",
                    timeInForce="GTC",
                    price=tp_price,
                    quantity=quantity,
                    reduceOnly=True
                )
                
                print(f"Take profit order placed at {tp_price:.2f}")
                
            except BinanceAPIException as e:
                print(f"Error placing take profit order: {e}")
        
        # Place stop loss order
        if sl_price:
            try:
                sl_side = "SELL" if signal == "LONG" else "BUY"
                
                sl_order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=sl_side,
                    type="STOP_MARKET",
                    timeInForce="GTC",
                    stopPrice=sl_price,
                    closePosition=True
                )
                
                print(f"Stop loss order placed at {sl_price:.2f}")
                
            except BinanceAPIException as e:
                print(f"Error placing stop loss order: {e}")
        
        return True
    
    def calculate_position_size(self, current_price, risk_pct=None):
        """
        Calculate the position size based on account balance and risk parameters
        
        Args:
            current_price: Current price of the trading symbol
            risk_pct: Risk percentage (optional)
            
        Returns:
            float: Position size in quote currency
        """
        # Get account balance
        if self.test_mode:
            # Use initial investment as account balance in test mode
            account_balance = self.trader.initial_investment
            print(f"Test mode: Using initial investment of {account_balance} USDT as account balance")
        else:
            account_balance = self.trader.account_manager.get_account_balance()
        
        if account_balance <= 0:
            print("Cannot calculate position size: Zero or negative account balance")
            # In test mode, provide a default value
            if self.test_mode:
                default_balance = 1000.0
                print(f"Test mode: Using default balance of {default_balance} USDT")
                account_balance = default_balance
            else:
                return 0
        
        # Determine investment amount
        if self.trader.use_full_investment:
            # Use the entire initial investment
            investment = self.trader.initial_investment
        elif self.trader.use_full_margin:
            # Use the entire available balance
            investment = account_balance
        else:
            # Use the initial investment or a percentage of the balance
            investment = min(self.trader.initial_investment, account_balance)
        
        # Apply compound interest if enabled
        if self.trader.compound_interest and self.trader.total_profit > 0:
            profit_multiplier = 1 + (self.trader.total_profit / 100)
            investment *= profit_multiplier
            print(f"Applying compound interest: {investment:.2f} (x{profit_multiplier:.2f})")
        
        # Apply risk percentage if specified
        if risk_pct is not None:
            position_size = investment * (risk_pct / 100)
        else:
            position_size = investment
        
        # Apply leverage
        position_size *= self.leverage
        
        # Ensure minimum notional value
        if self.test_mode:
            # Skip minimum notional check in test mode
            min_notional = 5.0  # Default value
        else:
            min_notional = self.trader.account_manager.get_min_notional()
            
        if position_size < min_notional:
            print(f"Warning: Position size {position_size:.2f} is below minimum notional {min_notional:.2f}")
            position_size = min_notional
        
        # Cap position size to available balance
        max_position = account_balance * self.leverage
        if position_size > max_position:
            print(f"Warning: Position size {position_size:.2f} exceeds maximum {max_position:.2f}")
            position_size = max_position
        
        print(f"Calculated position size: {position_size:.2f} USDT")
        return position_size
    
    def calculate_dynamic_position_size(self, current_price, atr_value=None):
        """
        Calculate a dynamic position size based on market volatility
        
        Args:
            current_price: Current price of the trading symbol
            atr_value: ATR value (optional)
            
        Returns:
            float: Position size in quote currency
        """
        # Get account balance
        account_balance = self.trader.account_manager.get_account_balance()
        
        if account_balance <= 0:
            print("Cannot calculate position size: Zero or negative account balance")
            return 0
        
        # Get ATR if not provided
        if atr_value is None:
            # Get latest market data
            df = self.trader.account_manager.get_latest_data(lookback_candles=100)
            
            if df is None or len(df) < 50:
                # Use default position size if data is not available
                return self.calculate_position_size(current_price)
            
            # Calculate ATR if not already present
            if "atr" not in df.columns:
                df["atr"] = calculate_atr(df)
            
            # Get the latest ATR value
            atr_value = df["atr"].iloc[-1]
        
        # Calculate ATR as percentage of price
        atr_pct = (atr_value / current_price) * 100
        
        # Adjust risk percentage based on volatility
        if atr_pct > 3.0:
            # High volatility - reduce risk
            risk_pct = 1.0
        elif atr_pct > 1.5:
            # Medium volatility
            risk_pct = 2.0
        else:
            # Low volatility - increase risk
            risk_pct = 3.0
        
        # Calculate position size with adjusted risk
        return self.calculate_position_size(current_price, risk_pct=risk_pct)
    
    def verify_stop_loss_take_profit_orders(self):
        """
        Verify that stop loss and take profit orders exist and recreate them if needed
        
        Returns:
            bool: True if orders are verified, False otherwise
        """
        if not self.trader.position_manager.has_open_position():
            return False
        
        try:
            # Get open orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            
            # Check for take profit and stop loss orders
            has_tp_order = False
            has_sl_order = False
            
            for order in open_orders:
                if order["type"] == "LIMIT" and order["reduceOnly"]:
                    has_tp_order = True
                elif order["type"] == "STOP_MARKET" or order["type"] == "STOP":
                    has_sl_order = True
            
            # Recreate missing orders if needed
            if not has_tp_order and self.fixed_tp > 0:
                self.recreate_take_profit_order()
            
            if not has_sl_order and self.fixed_sl > 0:
                self.recreate_stop_loss_order()
            
            return True
            
        except BinanceAPIException as e:
            print(f"Error verifying TP/SL orders: {e}")
            return False
    
    def recreate_stop_loss_order(self):
        """
        Recreate a stop loss order
        
        Returns:
            bool: True if order was recreated, False otherwise
        """
        if not self.trader.position_manager.has_open_position():
            return False
        
        position_info = self.trader.position_manager.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        entry_price = position_info["entry_price"]
        
        # Get price precision
        price_precision = self.trader.account_manager.get_price_precision()
        
        # Calculate stop loss price
        if position_type == "LONG":
            sl_price = round(entry_price * (1 - self.fixed_sl / (100 * self.leverage)), price_precision)
        else:  # SHORT
            sl_price = round(entry_price * (1 + self.fixed_sl / (100 * self.leverage)), price_precision)
        
        try:
            # Cancel existing stop loss orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            for order in open_orders:
                if order["type"] == "STOP_MARKET" or order["type"] == "STOP":
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=order["orderId"]
                    )
            
            # Place new stop loss order
            sl_side = "SELL" if position_type == "LONG" else "BUY"
            
            sl_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=sl_side,
                type="STOP_MARKET",
                timeInForce="GTC",
                stopPrice=sl_price,
                closePosition=True
            )
            
            print(f"Stop loss order recreated at {sl_price:.2f}")
            return True
            
        except BinanceAPIException as e:
            print(f"Error recreating stop loss order: {e}")
            return False
    
    def recreate_take_profit_order(self):
        """
        Recreate a take profit order
        
        Returns:
            bool: True if order was recreated, False otherwise
        """
        if not self.trader.position_manager.has_open_position():
            return False
        
        position_info = self.trader.position_manager.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        entry_price = position_info["entry_price"]
        amount = position_info["amount"]
        
        # Get price precision
        price_precision = self.trader.account_manager.get_price_precision()
        
        # Calculate take profit price
        if position_type == "LONG":
            tp_price = round(entry_price * (1 + self.fixed_tp / (100 * self.leverage)), price_precision)
        else:  # SHORT
            tp_price = round(entry_price * (1 - self.fixed_tp / (100 * self.leverage)), price_precision)
        
        try:
            # Cancel existing take profit orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            for order in open_orders:
                if order["type"] == "LIMIT" and order["reduceOnly"]:
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=order["orderId"]
                    )
            
            # Place new take profit order
            tp_side = "SELL" if position_type == "LONG" else "BUY"
            
            tp_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=tp_side,
                type="LIMIT",
                timeInForce="GTC",
                price=tp_price,
                quantity=amount,
                reduceOnly=True
            )
            
            print(f"Take profit order recreated at {tp_price:.2f}")
            return True
            
        except BinanceAPIException as e:
            print(f"Error recreating take profit order: {e}")
            return False
    
    def update_stop_loss_order(self):
        """
        Update the stop loss order to trail the price
        
        Returns:
            bool: True if order was updated, False otherwise
        """
        if not self.trader.position_manager.has_open_position():
            return False
        
        position_info = self.trader.position_manager.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        entry_price = position_info["entry_price"]
        current_price = position_info["mark_price"]
        
        # Calculate profit percentage
        if position_type == "LONG":
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price * 100
        
        # Only update if in profit
        if profit_pct < 1.0:
            return False
        
        # Get price precision
        price_precision = self.trader.account_manager.get_price_precision()
        
        # Calculate new stop loss price
        if position_type == "LONG":
            # Trail by 50% of the profit
            trail_pct = profit_pct * 0.5
            new_sl_price = round(entry_price * (1 + trail_pct / 100), price_precision)
        else:  # SHORT
            # Trail by 50% of the profit
            trail_pct = profit_pct * 0.5
            new_sl_price = round(entry_price * (1 - trail_pct / 100), price_precision)
        
        try:
            # Cancel existing stop loss orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            for order in open_orders:
                if order["type"] == "STOP_MARKET" or order["type"] == "STOP":
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=order["orderId"]
                    )
            
            # Place new stop loss order
            sl_side = "SELL" if position_type == "LONG" else "BUY"
            
            sl_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=sl_side,
                type="STOP_MARKET",
                timeInForce="GTC",
                stopPrice=new_sl_price,
                closePosition=True
            )
            
            print(f"Stop loss updated to {new_sl_price:.2f} (trailing by {trail_pct:.2f}%)")
            return True
            
        except BinanceAPIException as e:
            print(f"Error updating stop loss order: {e}")
            return False
    
    def display_next_update_countdown(self, interval_seconds):
        """
        Display countdown until next update
        
        Args:
            interval_seconds: Interval between updates in seconds
            
        Returns:
            datetime: Updated next run time
        """
        now = datetime.now()
        
        # Initialize next_run_time if not set
        if self.next_run_time is None:
            self.next_run_time = now + timedelta(seconds=interval_seconds)
            print(f"Next update in {interval_seconds:.0f} seconds")
            print("Countdown started...")
        
        # Update the countdown display
        self.next_run_time = display_countdown(self.next_run_time, interval_seconds, 
                                              f"⏱️ Next {self.symbol} update in")
        
        return self.next_run_time