import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
import telegram
import asyncio
import logging
from ml_models import MLManager

logger = logging.getLogger(__name__)

load_dotenv()

class TradingBot:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        
        # Initialize Telegram bot
        self.telegram_bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # List of stable coins to trade against
        self.stable_coins = ['USDT']
        
        # List of base currencies to trade
        self.base_currencies = ['BTC', 'ETH', 'BNB', 'SOL', 'LTC']
        
        # Generate trading pairs
        self.trading_pairs = []
        for base in self.base_currencies:
            for stable in self.stable_coins:
                pair = f"{base}{stable}"
                # Verify if the pair exists on Binance
                try:
                    self.client.get_symbol_info(pair)
                    self.trading_pairs.append(pair)
                except:
                    continue
        
        # Trading parameters
        self.positions = {pair: None for pair in self.trading_pairs}
        self.entry_prices = {pair: None for pair in self.trading_pairs}
        self.short_window = 20
        self.long_window = 50
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.atr_period = 14
        self.loop = asyncio.get_event_loop()
        
        # Initialize ML Manager
        self.ml_manager = MLManager()
        
        # Timeframe
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE
        self.lookback_period = 500  # Number of candles to fetch for ML training

    async def send_telegram_message(self, message):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message
            )
            print(f"Telegram message sent: {message}")
        except Exception as e:
            print(f"Error sending telegram message: {e}")

    def send_message(self, message):
        """Synchronous wrapper for send_telegram_message"""
        try:
            future = self.send_telegram_message(message)
            self.loop.run_until_complete(future)
        except Exception as e:
            print(f"Error in send_message: {e}")

    def execute_trade(self, traditional_signal, ml_signal, ml_confidence, df, symbol):
        try:
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            atr = df['ATR'].iloc[-1]
            
            # Combine signals - only trade when both agree or ML has high confidence
            signal = 0
            if traditional_signal == ml_signal:
                signal = traditional_signal  # Both signals agree
            elif ml_confidence > 0.75:
                signal = ml_signal  # ML has high confidence
            
            if signal == 1 and self.positions[symbol] != 'long':
                self.positions[symbol] = 'long'
                self.entry_prices[symbol] = current_price
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
                
                message = (
                    f"🟢 BUY Signal for {symbol}\n"
                    f"Price: ${current_price:,.2f}\n"
                    f"Stop Loss: ${stop_loss:,.2f}\n"
                    f"Take Profit: ${take_profit:,.2f}\n"
                    f"ATR: ${atr:,.2f}\n"
                    f"ML Confidence: {ml_confidence:.2%}"
                )
                self.send_message(message)
                
            elif signal == -1 and self.positions[symbol] != 'short':
                self.positions[symbol] = 'short'  # Fixed: using positions dict
                self.entry_prices[symbol] = current_price  # Fixed: using entry_prices dict
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)
                
                message = (
                    f"🔴 SELL Signal for {symbol}\n"  # Fixed: using symbol parameter
                    f"Price: ${current_price:,.2f}\n"
                    f"Stop Loss: ${stop_loss:,.2f}\n"
                    f"Take Profit: ${take_profit:,.2f}\n"
                    f"ATR: ${atr:,.2f}"
                )
                self.send_message(message)  # Using new wrapper method
            
            # Check stop loss and take profit
            elif self.positions[symbol] and self.entry_prices[symbol]:
                pnl_pct = (current_price - self.entry_prices[symbol]) / self.entry_prices[symbol]
                
                if self.positions[symbol] == 'long':
                    if pnl_pct <= -self.stop_loss_pct:
                        message = f"🛑 Stop Loss triggered for {symbol} LONG position\nLoss: {pnl_pct:.2%}"
                        self.send_message(message)  # Using new wrapper method
                        self.positions[symbol] = None
                        self.entry_prices[symbol] = None
                    elif pnl_pct >= self.take_profit_pct:
                        message = f"💰 Take Profit reached for {symbol} LONG position\nProfit: {pnl_pct:.2%}"  # Fixed: added symbol
                        asyncio.run(self.send_telegram_message(message))
                        self.positions[symbol] = None  # Fixed: using positions dict
                        self.entry_prices[symbol] = None  # Fixed: using entry_prices dict
                
                elif self.positions[symbol] == 'short':
                    if pnl_pct >= self.stop_loss_pct:
                        message = f"🛑 Stop Loss triggered for {symbol} SHORT position\nLoss: {pnl_pct:.2%}"  # Fixed: added symbol
                        asyncio.run(self.send_telegram_message(message))
                        self.positions[symbol] = None  # Fixed: using positions dict
                        self.entry_prices[symbol] = None  # Fixed: using entry_prices dict
                    elif pnl_pct <= -self.take_profit_pct:
                        message = f"💰 Take Profit reached for {symbol} SHORT position\nProfit: {-pnl_pct:.2%}"  # Fixed: added symbol
                        asyncio.run(self.send_telegram_message(message))
                        self.positions[symbol] = None  # Fixed: using positions dict
                        self.entry_prices[symbol] = None  # Fixed: using entry_prices dict
                
        except BinanceAPIException as e:
            print(f"Binance API error while executing trade: {e}")
            error_message = f"❌ Error executing trade: {e}"
            asyncio.run(self.send_telegram_message(error_message))
            print(error_message)

    def calculate_signals(self, df):
        # Calculate moving averages
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate ATR
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=self.atr_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1
        
        return df

    def get_historical_data(self, symbol):
        try:
            print(f"Fetching historical data for {symbol}")
            # Fix the time parameter format
            klines = self.client.get_historical_klines(
                symbol,
                self.timeframe,
                f"{int(self.lookback_period * 15)} minutes ago UTC"  # Convert intervals to minutes for 15m timeframe
            )
            
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
            
            # Calculate traditional signals
            df = self.calculate_signals(df)
            
            print(f"Successfully retrieved historical data for {symbol}")
            return df
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return None

    def run(self):
        print("Starting trading bot with ML capabilities...")
        
        # Initial training of models if needed
        print("Checking ML models...")
        for symbol in self.trading_pairs:
            df = self.get_historical_data(symbol)
            if df is not None and symbol not in self.ml_manager.models:
                self.ml_manager.train_ml_model(symbol, df)
        
        while True:
            try:
                for symbol in self.trading_pairs:
                    print(f"Processing {symbol}")
                    df = self.get_historical_data(symbol)
                    if df is not None:
                        # Get traditional signal
                        traditional_signal = df['signal'].iloc[-1]
                        
                        # Get ML signal
                        ml_signal, ml_confidence = self.ml_manager.get_ml_signal(symbol, df)
                        
                        # Execute trade based on combined signals
                        self.execute_trade(traditional_signal, ml_signal, ml_confidence, df, symbol)
                
                print("Waiting for next iteration...")
                time.sleep(900)  # Check every 15 minutes
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()