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
from .utils.notifications import send_telegram_notification
from .utils.data_fetcher import get_market_data, calculate_indicators
from .utils.trade_manager import execute_trade, check_take_profit_stop_loss
from .utils.reporting import save_results

# Load environment variables
load_dotenv()

class RealtimeSimulator:
    def __init__(self, symbol='BTCUSDT', initial_investment=50.0, daily_profit_target=15.0, leverage=5):
        """
        Initialize the real-time simulator
        
        Args:
            symbol: Trading pair to simulate
            initial_investment: Starting capital in USD
            daily_profit_target: Target profit per day in USD
            leverage: Margin trading leverage (1x-20x)
        """
        self.symbol = symbol
        self.initial_investment = initial_investment
        self.daily_profit_target = daily_profit_target
        
        # Set leverage (constrain between 1x and 20x)
        self.leverage = max(1, min(20, leverage))
        
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
        return get_market_data(
            self.client, 
            self.symbol, 
            self.timeframe, 
            lookback_candles,
            self.short_window,
            self.long_window,
            self.atr_period
        )
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for trading signals"""
        return calculate_indicators(df, self.short_window, self.long_window, self.atr_period)
    
    def send_notification(self, message):
        """Send Telegram notification"""
        if self.notifications_enabled:
            send_telegram_notification(self.telegram_bot_token, self.telegram_chat_id, message)
    
    def check_take_profit_stop_loss(self, current_price, timestamp):
        """Check for take profit or stop loss with risk management"""
        return check_take_profit_stop_loss(
            self.simulator, 
            current_price, 
            timestamp, 
            self.initial_investment,
            self.max_daily_loss,
            self.daily_loss,
            self.save_realtime_results,
            self.send_notification
        )
        
    def save_realtime_results(self):
        """Save real-time results to files"""
        save_results(
            self.simulator,
            self.results_dir,
            self.symbol,
            self.initial_investment
        )
        
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
            
        return execute_trade(
            self.simulator,
            signal,
            current_price,
            timestamp,
            self.initial_investment,
            self.max_position_size,
            self.risk_per_trade,
            self.get_latest_data
        )
        
    def run_realtime_simulation(self, duration_hours=24, update_interval_minutes=15):
        """
        Run real-time simulation
        
        Args:
            duration_hours: How long to run the simulation in hours
            update_interval_minutes: How often to update in minutes
        """
        from .simulation_runner import run_simulation
        
        return run_simulation(
            self,
            duration_hours,
            update_interval_minutes
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Trading Simulator')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair to simulate')
    parser.add_argument('--investment', type=float, default=50.0, help='Initial investment amount')
    parser.add_argument('--target', type=float, default=15.0, help='Daily profit target')
    parser.add_argument('--hours', type=int, default=24, help='Duration in hours')
    parser.add_argument('--interval', type=int, default=15, help='Update interval in minutes')
    parser.add_argument('--leverage', type=int, default=5, help='Margin trading leverage (1-20x)')
    
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