"""
Position management module for handling trading positions.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from binance.exceptions import BinanceAPIException
from ..utils.indicators import calculate_atr

class PositionManager:
    def __init__(self, trader):
        """
        Initialize the PositionManager with a reference to the trader
        
        Args:
            trader: The RealtimeTrader instance
        """
        self.trader = trader
        self.client = trader.client
        self.symbol = trader.symbol
        self.use_dynamic_take_profit = trader.use_dynamic_take_profit
        self.fixed_tp = trader.fixed_tp
        self.fixed_sl = trader.fixed_sl
        self.enable_pyramiding = trader.enable_pyramiding
        self.max_pyramid_entries = trader.max_pyramid_entries
        self.pyramid_threshold_pct = trader.pyramid_threshold_pct
        self.use_scalping_mode = trader.use_scalping_mode
        self.scalping_tp_factor = trader.scalping_tp_factor
        self.scalping_sl_factor = trader.scalping_sl_factor
    
    def has_open_position(self):
        """
        Check if there is an open position for the trading symbol
        
        Returns:
            bool: True if there is an open position, False otherwise
        """
        try:
            positions = self.client.futures_position_information()
            
            for position in positions:
                if position["symbol"] == self.symbol:
                    amount = float(position["positionAmt"])
                    if amount != 0:
                        return True
            
            return False
            
        except BinanceAPIException as e:
            print(f"Error checking open position: {e}")
            return False
    
    def get_position_info(self):
        """
        Get information about the current position
        
        Returns:
            dict: Position information or None if no position
        """
        try:
            positions = self.client.futures_position_information()
            
            for position in positions:
                if position["symbol"] == self.symbol:
                    amount = float(position["positionAmt"])
                    if amount != 0:
                        # Determine position type
                        position_type = "LONG" if amount > 0 else "SHORT"
                        
                        # Calculate unrealized PNL percentage
                        entry_price = float(position["entryPrice"])
                        mark_price = float(position["markPrice"])
                        leverage = float(position["leverage"])
                        
                        if position_type == "LONG":
                            pnl_pct = (mark_price - entry_price) / entry_price * 100 * leverage
                        else:
                            pnl_pct = (entry_price - mark_price) / entry_price * 100 * leverage
                        
                        # Return position details
                        return {
                            "type": position_type,
                            "amount": abs(amount),
                            "entry_price": entry_price,
                            "mark_price": mark_price,
                            "leverage": leverage,
                            "pnl_pct": pnl_pct,
                            "liquidation_price": float(position["liquidationPrice"]),
                            "margin_type": position["marginType"],
                            "isolated_margin": float(position["isolatedMargin"]),
                            "unrealized_profit": float(position["unRealizedProfit"])
                        }
            
            return None
            
        except BinanceAPIException as e:
            print(f"Error getting position info: {e}")
            return None
    
    def check_take_profit_stop_loss(self, current_price, timestamp):
        """
        Check if take profit or stop loss conditions are met
        
        Args:
            current_price: Current price of the trading symbol
            timestamp: Current timestamp
            
        Returns:
            bool: True if position was closed, False otherwise
        """
        if not self.has_open_position():
            return False
        
        position_info = self.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        entry_price = position_info["entry_price"]
        pnl_pct = position_info["pnl_pct"]
        
        print(f"\n--- Position Status ---")
        print(f"Position: {position_type}")
        print(f"Entry Price: {entry_price:.2f}")
        print(f"Current Price: {current_price:.2f}")
        print(f"PNL: {pnl_pct:.2f}%")
        
        # Check for take profit
        take_profit_hit = False
        stop_loss_hit = False
        
        # Dynamic take profit based on market conditions
        if self.use_dynamic_take_profit:
            # Get dynamic TP/SL levels
            tp_pct, sl_pct = self.calculate_dynamic_take_profit(current_price)
            
            print(f"Dynamic TP: {tp_pct:.2f}%, SL: {sl_pct:.2f}%")
            
            # Check if TP or SL hit
            if position_type == "LONG":
                if pnl_pct >= tp_pct:
                    take_profit_hit = True
                elif pnl_pct <= -sl_pct:
                    stop_loss_hit = True
            else:  # SHORT
                if pnl_pct >= tp_pct:
                    take_profit_hit = True
                elif pnl_pct <= -sl_pct:
                    stop_loss_hit = True
        
        # Fixed take profit/stop loss
        elif self.fixed_tp > 0 or self.fixed_sl > 0:
            if self.fixed_tp > 0 and pnl_pct >= self.fixed_tp:
                take_profit_hit = True
                print(f"Fixed TP hit: {pnl_pct:.2f}% >= {self.fixed_tp:.2f}%")
            
            if self.fixed_sl > 0 and pnl_pct <= -self.fixed_sl:
                stop_loss_hit = True
                print(f"Fixed SL hit: {pnl_pct:.2f}% <= -{self.fixed_sl:.2f}%")
        
        # Default TP/SL if none specified
        else:
            # Default take profit at 5%
            if pnl_pct >= 5.0:
                take_profit_hit = True
                print(f"Default TP hit: {pnl_pct:.2f}% >= 5.0%")
            
            # Default stop loss at 3%
            if pnl_pct <= -3.0:
                stop_loss_hit = True
                print(f"Default SL hit: {pnl_pct:.2f}% <= -3.0%")
        
        # Execute take profit or stop loss if hit
        if take_profit_hit:
            print(f"🎯 TAKE PROFIT triggered at {pnl_pct:.2f}%")
            self.close_position(current_price, timestamp, reason="take_profit")
            return True
        
        if stop_loss_hit:
            print(f"🛑 STOP LOSS triggered at {pnl_pct:.2f}%")
            self.close_position(current_price, timestamp, reason="stop_loss")
            return True
        
        # Check for partial take profit in scalping mode
        if self.use_scalping_mode and pnl_pct >= self.fixed_tp * self.scalping_tp_factor:
            self.execute_partial_take_profit(current_price, timestamp)
        
        return False
    
    def execute_partial_take_profit(self, current_price, timestamp):
        """
        Execute a partial take profit by closing part of the position
        
        Args:
            current_price: Current price of the trading symbol
            timestamp: Current timestamp
            
        Returns:
            bool: True if partial take profit was executed, False otherwise
        """
        if not self.has_open_position():
            return False
        
        position_info = self.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        amount = position_info["amount"]
        pnl_pct = position_info["pnl_pct"]
        
        # Close 50% of the position
        close_amount = amount * 0.5
        
        try:
            # Determine order side
            order_side = "SELL" if position_type == "LONG" else "BUY"
            
            # Get quantity precision
            quantity_precision = self.trader.account_manager.get_quantity_precision()
            close_amount = round(close_amount, quantity_precision)
            
            # Ensure minimum quantity
            min_qty = self.trader.account_manager.get_min_quantity()
            if close_amount < min_qty:
                print(f"Partial take profit amount {close_amount} is below minimum quantity {min_qty}")
                return False
            
            # Place market order to close partial position
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type="MARKET",
                quantity=close_amount,
                reduceOnly=True
            )
            
            print(f"🔄 PARTIAL TAKE PROFIT executed at {pnl_pct:.2f}%")
            print(f"Closed {close_amount} {self.symbol} at {current_price:.2f}")
            
            # Record the trade
            self.trader.trading_history.append({
                "timestamp": timestamp,
                "action": f"PARTIAL_CLOSE_{position_type}",
                "price": current_price,
                "quantity": close_amount,
                "pnl_pct": pnl_pct,
                "reason": "partial_take_profit"
            })
            
            # Send notification
            self.trader.send_notification(
                f"🔄 PARTIAL TAKE PROFIT on {self.symbol}\n"
                f"Closed {close_amount} at {current_price:.2f}\n"
                f"PNL: {pnl_pct:.2f}%"
            )
            
            return True
            
        except BinanceAPIException as e:
            print(f"Error executing partial take profit: {e}")
            return False
    
    def execute_pyramid_entry(self, current_price, timestamp):
        """
        Execute a pyramid entry by adding to an existing position
        
        Args:
            current_price: Current price of the trading symbol
            timestamp: Current timestamp
            
        Returns:
            bool: True if pyramid entry was executed, False otherwise
        """
        if not self.has_open_position() or not self.enable_pyramiding:
            return False
        
        if self.trader.pyramid_entries >= self.max_pyramid_entries:
            return False
        
        position_info = self.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        entry_price = position_info["entry_price"]
        
        # Check if price has moved in favorable direction
        price_moved_favorably = False
        
        if position_type == "LONG" and current_price > entry_price * (1 + self.pyramid_threshold_pct / 100):
            price_moved_favorably = True
            price_movement = (current_price - entry_price) / entry_price * 100
        elif position_type == "SHORT" and current_price < entry_price * (1 - self.pyramid_threshold_pct / 100):
            price_moved_favorably = True
            price_movement = (entry_price - current_price) / entry_price * 100
        
        if not price_moved_favorably:
            return False
        
        print(f"\n--- Pyramid Entry Opportunity ---")
        print(f"Position: {position_type}")
        print(f"Entry Price: {entry_price:.2f}")
        print(f"Current Price: {current_price:.2f}")
        print(f"Price Movement: {price_movement:.2f}%")
        print(f"Threshold: {self.pyramid_threshold_pct:.2f}%")
        
        # Calculate position size for pyramid entry (50% of original)
        position_size = self.trader.calculate_position_size(current_price) * 0.5
        
        try:
            # Determine order side
            order_side = "BUY" if position_type == "LONG" else "SELL"
            
            # Get quantity precision
            quantity_precision = self.trader.account_manager.get_quantity_precision()
            quantity = round(position_size / current_price, quantity_precision)
            
            # Ensure minimum quantity
            min_qty = self.trader.account_manager.get_min_quantity()
            if quantity < min_qty:
                print(f"Pyramid entry quantity {quantity} is below minimum quantity {min_qty}")
                return False
            
            # Place market order for pyramid entry
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type="MARKET",
                quantity=quantity
            )
            
            # Increment pyramid entries counter
            self.trader.pyramid_entries += 1
            
            print(f"🔺 PYRAMID ENTRY executed for {position_type}")
            print(f"Added {quantity} {self.symbol} at {current_price:.2f}")
            print(f"Pyramid entries: {self.trader.pyramid_entries}/{self.max_pyramid_entries}")
            
            # Record the trade
            self.trader.trading_history.append({
                "timestamp": timestamp,
                "action": f"PYRAMID_{position_type}",
                "price": current_price,
                "quantity": quantity,
                "reason": "pyramid_entry"
            })
            
            # Send notification
            self.trader.send_notification(
                f"🔺 PYRAMID ENTRY on {self.symbol}\n"
                f"Added {quantity} to {position_type} at {current_price:.2f}\n"
                f"Entry #{self.trader.pyramid_entries}/{self.max_pyramid_entries}"
            )
            
            return True
            
        except BinanceAPIException as e:
            print(f"Error executing pyramid entry: {e}")
            return False
    
    def close_position(self, current_price, timestamp, reason="manual"):
        """
        Close the current position
        
        Args:
            current_price: Current price of the trading symbol
            timestamp: Current timestamp
            reason: Reason for closing the position
            
        Returns:
            bool: True if position was closed, False otherwise
        """
        if not self.has_open_position():
            return False
        
        position_info = self.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        amount = position_info["amount"]
        entry_price = position_info["entry_price"]
        pnl_pct = position_info["pnl_pct"]
        
        try:
            # Determine order side
            order_side = "SELL" if position_type == "LONG" else "BUY"
            
            # Get quantity precision
            quantity_precision = self.trader.account_manager.get_quantity_precision()
            amount = round(amount, quantity_precision)
            
            # Place market order to close position
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type="MARKET",
                quantity=amount,
                reduceOnly=True
            )
            
            print(f"🔴 POSITION CLOSED: {position_type}")
            print(f"Closed {amount} {self.symbol} at {current_price:.2f}")
            print(f"Entry Price: {entry_price:.2f}")
            print(f"PNL: {pnl_pct:.2f}%")
            print(f"Reason: {reason}")
            
            # Update profit tracking
            self.trader.daily_profit += pnl_pct
            self.trader.total_profit += pnl_pct
            
            # Reset pyramid entries counter
            self.trader.pyramid_entries = 0
            
            # Record the trade
            self.trader.trading_history.append({
                "timestamp": timestamp,
                "action": f"CLOSE_{position_type}",
                "price": current_price,
                "quantity": amount,
                "pnl_pct": pnl_pct,
                "reason": reason
            })
            
            # Send notification
            emoji = "🎯" if reason == "take_profit" else "🛑" if reason == "stop_loss" else "🔴"
            self.trader.send_notification(
                f"{emoji} POSITION CLOSED on {self.symbol}\n"
                f"Type: {position_type}\n"
                f"Closed {amount} at {current_price:.2f}\n"
                f"PNL: {pnl_pct:.2f}%\n"
                f"Reason: {reason}"
            )
            
            return True
            
        except BinanceAPIException as e:
            print(f"Error closing position: {e}")
            return False
    
    def reassess_position(self, current_price, timestamp):
        """
        Reassess the current position based on market conditions
        
        Args:
            current_price: Current price of the trading symbol
            timestamp: Current timestamp
            
        Returns:
            bool: True if position was modified, False otherwise
        """
        if not self.has_open_position() or not self.trader.reassess_positions:
            return False
        
        position_info = self.get_position_info()
        
        if not position_info:
            return False
        
        position_type = position_info["type"]
        
        # Generate a new signal
        signal = self.trader.signal_generator.generate_trading_signal()
        
        # Check if signal contradicts current position
        if (position_type == "LONG" and signal == "SHORT") or (position_type == "SHORT" and signal == "LONG"):
            print(f"\n--- Position Reassessment ---")
            print(f"Current Position: {position_type}")
            print(f"New Signal: {signal}")
            print(f"Signal contradicts current position, closing position")
            
            # Close the position
            self.close_position(current_price, timestamp, reason="signal_reversal")
            
            # Execute new trade after a short delay
            time.sleep(2)
            self.trader.order_executor.execute_trade(signal, current_price, timestamp)
            
            return True
        
        return False
    
    def save_trading_results(self):
        """
        Save trading results to a CSV file
        
        Returns:
            bool: True if results were saved, False otherwise
        """
        if not self.trader.trading_history:
            print("No trading history to save")
            return False
        
        try:
            # Convert trading history to DataFrame
            df = pd.DataFrame(self.trader.trading_history)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_results_{self.symbol}_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            print(f"Trading results saved to {filename}")
            
            # Send notification
            self.trader.send_notification(
                f"📊 Trading session completed for {self.symbol}\n"
                f"Total profit: {self.trader.total_profit:.2f}%\n"
                f"Results saved to {filename}"
            )
            
            return True
            
        except Exception as e:
            print(f"Error saving trading results: {e}")
            return False
    
    def calculate_dynamic_take_profit(self, current_price):
        """
        Calculate dynamic take profit and stop loss levels based on market conditions
        
        Args:
            current_price: Current price of the trading symbol
            
        Returns:
            tuple: (take_profit_percentage, stop_loss_percentage)
        """
        # Get latest market data
        df = self.trader.account_manager.get_latest_data(lookback_candles=100)
        
        if df is None or len(df) < 50:
            # Default values if data is not available
            return (5.0, 3.0)
        
        # Calculate ATR if not already present
        if "atr" not in df.columns:
            df["atr"] = calculate_atr(df)
        
        # Get the latest ATR value
        atr_value = df["atr"].iloc[-1]
        
        # Calculate ATR as percentage of price
        atr_pct = (atr_value / current_price) * 100
        
        # Calculate volatility based on recent price action
        recent_volatility = df["close"].pct_change().rolling(window=20).std().iloc[-1] * 100
        
        # Adjust take profit and stop loss based on volatility
        if recent_volatility > 5.0:
            # High volatility - wider TP/SL
            tp_pct = max(7.0, atr_pct * 3.0)
            sl_pct = max(5.0, atr_pct * 2.0)
        elif recent_volatility > 2.0:
            # Medium volatility
            tp_pct = max(5.0, atr_pct * 2.5)
            sl_pct = max(3.0, atr_pct * 1.5)
        else:
            # Low volatility - tighter TP/SL
            tp_pct = max(3.0, atr_pct * 2.0)
            sl_pct = max(2.0, atr_pct * 1.0)
        
        # Adjust for scalping mode
        if self.use_scalping_mode:
            tp_pct *= self.scalping_tp_factor
            sl_pct *= self.scalping_sl_factor
        
        # Apply leverage factor
        tp_pct *= self.trader.leverage
        sl_pct *= self.trader.leverage
        
        # Cap at reasonable values
        tp_pct = min(tp_pct, 50.0)
        sl_pct = min(sl_pct, 30.0)
        
        return (tp_pct, sl_pct) 