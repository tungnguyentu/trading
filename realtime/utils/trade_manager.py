import re
from datetime import datetime

def check_take_profit_stop_loss(simulator, current_price, timestamp, initial_investment, max_daily_loss, daily_loss, save_results_callback, notification_callback):
    """Check for take profit or stop loss with risk management"""
    result = simulator.check_take_profit_stop_loss(current_price, timestamp)
    
    if result and ("STOP LOSS" in result or "TAKE PROFIT" in result):
        # Update daily loss tracking
        loss_match = re.search(r'(Profit|Loss): \$([0-9.-]+)', result)
        if loss_match:
            loss_amount = float(loss_match.group(2))
            
            # Update daily loss if it's a loss
            if "Loss" in result:
                daily_loss += loss_amount
                
                # Check if maximum daily loss is reached
                if daily_loss >= (initial_investment * max_daily_loss):
                    trading_disabled = True
                    message = (
                        "⚠️ Trading disabled for today\n"
                        f"Reached maximum daily loss: ${daily_loss:.2f}\n"
                        f"Current balance: ${simulator.current_balance:.2f}"
                    )
                    notification_callback(message)
        
        # Save results after each trade
        save_results_callback()
    
    return result

def execute_trade(simulator, signal, current_price, timestamp, initial_investment, max_position_size, risk_per_trade, get_data_callback):
    """Execute trade with risk management"""
    # Don't trade if balance is too low
    if simulator.current_balance < (initial_investment * 0.5):
        print(f"Balance too low (${simulator.current_balance:.2f}), trading paused")
        return None
        
    # Calculate maximum position size based on current balance
    max_position_value = simulator.current_balance * max_position_size
    
    # Calculate position size based on risk per trade
    risk_amount = simulator.current_balance * risk_per_trade
    
    # Get ATR for dynamic stop loss and take profit
    latest_df = get_data_callback(lookback_candles=20)
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
    simulator.stop_loss_pct = stop_loss_pct
    simulator.take_profit_pct = take_profit_pct
    
    # Execute the trade in the simulator
    if signal == 1:  # BUY signal
        # Calculate stop loss and take profit prices
        stop_loss_price = current_price * (1 - stop_loss_pct)
        take_profit_price = current_price * (1 + take_profit_pct)
        
        # Execute the trade
        result = simulator.open_long_position(
            current_price, 
            position_size, 
            stop_loss_price, 
            take_profit_price, 
            timestamp
        )
        
        if result:
            return f"BUY: Opened long position at ${current_price:.2f} with {position_size:.6f} units"
    
    elif signal == -1:  # SELL signal
        # Calculate stop loss and take profit prices
        stop_loss_price = current_price * (1 + stop_loss_pct)
        take_profit_price = current_price * (1 - take_profit_pct)
        
        # Execute the trade
        result = simulator.open_short_position(
            current_price, 
            position_size, 
            stop_loss_price, 
            take_profit_price, 
            timestamp
        )
        
        if result:
            return f"SELL: Opened short position at ${current_price:.2f} with {position_size:.6f} units"
    
    return None

def calculate_position_metrics(simulator, current_price):
    """Calculate current position metrics"""
    if not simulator.position:
        return None
        
    position_value = simulator.position_size * current_price
    profit_loss = 0
    profit_pct = 0
    
    if simulator.position == 'long':
        profit_loss = simulator.position_size * (current_price - simulator.entry_price)
        profit_pct = (current_price - simulator.entry_price) / simulator.entry_price * 100
    else:  # short
        profit_loss = simulator.position_size * (simulator.entry_price - current_price)
        profit_pct = (simulator.entry_price - current_price) / simulator.entry_price * 100
    
    return {
        'position': simulator.position,
        'entry_price': simulator.entry_price,
        'current_price': current_price,
        'position_size': simulator.position_size,
        'position_value': position_value,
        'profit_loss': profit_loss,
        'profit_pct': profit_pct,
        'stop_loss': simulator.stop_loss_price,
        'take_profit': simulator.take_profit_price,
        'entry_time': simulator.entry_time
    }

def close_all_positions(simulator, current_price, timestamp, reason='manual'):
    """Close all open positions"""
    if not simulator.position:
        return None
    
    result = simulator.close_position(current_price, timestamp, reason)
    
    if result:
        position_type = simulator.position
        profit = result.get('profit', 0)
        
        simulator.position = None
        simulator.entry_price = None
        simulator.position_size = 0
        simulator.stop_loss_price = None
        simulator.take_profit_price = None
        
        return {
            'position': position_type,
            'exit_price': current_price,
            'profit': profit,
            'reason': reason
        }
    
    return None