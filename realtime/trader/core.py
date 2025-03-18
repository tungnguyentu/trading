"""
Core RealtimeTrader class that orchestrates the trading operations.
"""

import time
import os
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

from .account import AccountManager
from .signals import SignalGenerator
from .position import PositionManager
from .orders import OrderExecutor
from .ml_integration import MLIntegration
from ..utils.notifications import send_telegram_notification

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
        """
        # Trading configuration
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
        
        # Take profit and stop loss settings
        self.use_dynamic_take_profit = use_dynamic_take_profit
        self.fixed_tp = fixed_tp
        self.fixed_sl = fixed_sl
        
        # Signal settings
        self.trend_following_mode = trend_following_mode
        self.use_enhanced_signals = use_enhanced_signals
        self.signal_confirmation_threshold = signal_confirmation_threshold
        self.signal_cooldown_minutes = signal_cooldown_minutes
        
        # Scalping settings
        self.use_scalping_mode = use_scalping_mode
        self.scalping_tp_factor = scalping_tp_factor
        self.scalping_sl_factor = scalping_sl_factor
        
        # ML settings
        self.use_ml_signals = use_ml_signals
        self.ml_confidence = ml_confidence
        self.train_ml = train_ml
        self.retrain_interval = retrain_interval
        self.reassess_positions = reassess_positions
        
        # Initialize trading client
        self.client = None
        self.initialize_trading_client()
        
        # Initialize managers
        self.account_manager = AccountManager(self)
        self.signal_generator = SignalGenerator(self)
        self.position_manager = PositionManager(self)
        self.order_executor = OrderExecutor(self)
        
        # Initialize ML integration if needed
        if self.use_ml_signals or self.train_ml:
            self.ml_integration = MLIntegration(self)
        else:
            self.ml_integration = None
        
        # Trading state
        self.last_signal_time = None
        self.trading_history = []
        self.daily_profit = 0.0
        self.total_profit = 0.0
        self.current_position = None
        self.pyramid_entries = 0
        
    def initialize_trading_client(self):
        """Initialize the Binance client with API keys"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in environment variables")
        
        # Initialize client with testnet if in test mode
        self.client = Client(api_key, api_secret, testnet=self.test_mode)
        
        # Test connection based on whether using testnet or not
        try:
            # Check if we can access futures account
            if self.test_mode:
                # For testnet
                print("Connecting to Binance Futures Testnet...")
                # Different endpoint for testnet
                self.client.futures_ping()
                account = self.client.futures_account()
                print("Successfully connected to Binance Futures Testnet")
            else:
                # For production
                print("Connecting to Binance Futures...")
                account = self.client.futures_account()
                print("Successfully connected to Binance Futures")
                
            # Validate account has USDT balance
            has_usdt = False
            if 'assets' in account:
                for asset in account['assets']:
                    if asset['asset'] == 'USDT':
                        has_usdt = True
                        print(f"USDT Balance: {float(asset['availableBalance'])}")
                        break
                        
            if not has_usdt:
                print("Warning: No USDT balance found in futures account")
                
        except BinanceAPIException as e:
            # Handle common error codes
            if e.code == -5000 and "Path /fapi/v1/account" in str(e):
                print("Error: Invalid futures API path. Make sure you have a Binance Futures account.")
                print("If using testnet, verify your testnet.binance.vision account is set up correctly.")
            elif e.code == -2015:
                print("Error: Invalid API key, secret or restrictions.")
            elif e.code == -1021:
                print("Error: Timestamp for this request was outside the recvWindow.")
            elif e.code == -1022:
                print("Error: Signature for this request was invalid.")
            elif e.code == -2014:
                print("Error: API-key format invalid.")
            else:
                print(f"Error connecting to Binance API: {e}")
                
            # Show detailed error information to help with debugging
            print(f"Error code: {e.code}, Message: {e.message}")
            print(f"Detailed error information: {str(e)}")
            
            # Create fallback test client if in test mode
            if self.test_mode:
                print("Switching to simulation mode without API connectivity...")
                # We'll continue in a simulation mode without real API access
            else:
                raise ValueError("Cannot continue trading without valid Binance Futures API connection")
    
    def send_notification(self, message):
        """Send notification via Telegram"""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN") 
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        send_telegram_notification(bot_token, chat_id, message)
    
    def run_real_trading(
        self, duration_hours=24, update_interval_minutes=0, update_interval_seconds=0
    ):
        """
        Run the trading bot for a specified duration with regular updates
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        update_interval = timedelta(
            minutes=update_interval_minutes, seconds=update_interval_seconds
        )
        
        print(f"Starting real-time trading for {duration_hours} hours")
        print(f"Update interval: {update_interval_minutes} minutes {update_interval_seconds} seconds")
        print(f"Trading will end at: {end_time}")
        
        self.send_notification(
            f"🚀 Starting real-time trading on {self.symbol}\n"
            f"Duration: {duration_hours} hours\n"
            f"Update interval: {update_interval_minutes}m {update_interval_seconds}s"
        )
        
        last_update = datetime.now()
        
        try:
            while datetime.now() < end_time:
                current_time = datetime.now()
                
                if current_time - last_update >= update_interval:
                    print(f"\n{'='*50}")
                    print(f"Update at {current_time}")
                    
                    # Get current market data
                    current_price = self.account_manager.get_current_price()
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Check existing positions
                    if self.position_manager.has_open_position():
                        # Manage existing position
                        self.position_manager.check_take_profit_stop_loss(current_price, timestamp)
                        
                        # Reassess position if enabled
                        if self.reassess_positions:
                            self.position_manager.reassess_position(current_price, timestamp)
                        
                        # Check for pyramid entry if enabled
                        if self.enable_pyramiding and self.pyramid_entries < self.max_pyramid_entries:
                            self.position_manager.execute_pyramid_entry(current_price, timestamp)
                    else:
                        # Generate trading signals
                        signal = self.signal_generator.generate_trading_signal()
                        
                        # Execute trade if signal is valid
                        if signal in ["LONG", "SHORT"]:
                            self.order_executor.execute_trade(signal, current_price, timestamp)
                    
                    # Train/retrain ML model if enabled
                    if self.ml_integration and self.train_ml:
                        if self.retrain_interval > 0:
                            # Check if it's time to retrain
                            hours_since_start = (current_time - start_time).total_seconds() / 3600
                            if hours_since_start % self.retrain_interval < update_interval.total_seconds() / 3600:
                                self.ml_integration.train_ml_model()
                    
                    last_update = current_time
                
                # Sleep to prevent excessive API calls
                time.sleep(10)
        
        except Exception as e:
            error_message = f"Error in trading loop: {str(e)}"
            print(error_message)
            self.send_notification(f"⚠️ ERROR: {error_message}")
        
        finally:
            # Save trading results
            self.position_manager.save_trading_results()
            
            # Send final notification
            self.send_notification(
                f"🏁 Trading session completed for {self.symbol}\n"
                f"Total profit: {self.total_profit:.2f}%"
            )