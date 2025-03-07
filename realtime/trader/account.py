"""
Account management module for handling account-related operations.
"""

import pandas as pd
from binance.exceptions import BinanceAPIException

class AccountManager:
    def __init__(self, trader):
        """
        Initialize the AccountManager with a reference to the trader
        
        Args:
            trader: The RealtimeTrader instance
        """
        self.trader = trader
        self.client = trader.client
        self.symbol = trader.symbol
    
    def get_latest_data(self, lookback_candles=500):
        """
        Get the latest market data for the trading symbol
        
        Args:
            lookback_candles: Number of candles to retrieve
            
        Returns:
            DataFrame with market data
        """
        try:
            # Get klines data from Binance
            klines = self.client.futures_klines(
                symbol=self.symbol, interval="1m", limit=lookback_candles
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ]
            )
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except BinanceAPIException as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def get_account_balance(self):
        """
        Get the current account balance
        
        Returns:
            Float: Available balance in USDT
        """
        try:
            # If in test mode, use initial investment as balance
            if hasattr(self.trader, 'test_mode') and self.trader.test_mode:
                simulated_balance = self.trader.initial_investment
                print(f"Test mode: Using simulated balance of {simulated_balance} USDT")
                return simulated_balance
                
            # Get futures account information
            account_info = self.client.futures_account()
            
            # Find USDT balance
            for asset in account_info["assets"]:
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
            
            return 0.0
            
        except BinanceAPIException as e:
            print(f"Error fetching account balance: {e}")
            
            # If in test mode, provide a fallback simulated balance
            if hasattr(self.trader, 'test_mode') and self.trader.test_mode:
                simulated_balance = self.trader.initial_investment
                print(f"Test mode: Using fallback simulated balance of {simulated_balance} USDT")
                return simulated_balance
                
            return 0.0
    
    def get_current_price(self):
        """
        Get the current price of the trading symbol
        
        Returns:
            Float: Current price
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker["price"])
            
        except BinanceAPIException as e:
            print(f"Error fetching current price: {e}")
            return 0.0
    
    def get_symbol_info(self):
        """
        Get information about the trading symbol
        
        Returns:
            Dict: Symbol information
        """
        try:
            exchange_info = self.client.get_exchange_info()
            
            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == self.symbol:
                    return symbol_info
            
            return None
            
        except BinanceAPIException as e:
            print(f"Error fetching symbol info: {e}")
            return None
    
    def get_futures_symbol_info(self):
        """
        Get futures-specific information about the trading symbol
        
        Returns:
            Dict: Futures symbol information
        """
        try:
            exchange_info = self.client.futures_exchange_info()
            
            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == self.symbol:
                    return symbol_info
            
            return None
            
        except BinanceAPIException as e:
            print(f"Error fetching futures symbol info: {e}")
            return None
    
    def get_quantity_precision(self):
        """
        Get the quantity precision for the trading symbol
        
        Returns:
            Int: Quantity precision
        """
        symbol_info = self.get_futures_symbol_info()
        
        if symbol_info:
            for filter_item in symbol_info["filters"]:
                if filter_item["filterType"] == "LOT_SIZE":
                    step_size = float(filter_item["stepSize"])
                    precision = 0
                    
                    if step_size < 1:
                        precision = len(str(step_size).split(".")[1].rstrip("0"))
                    
                    return precision
        
        # Default precision if not found
        return 3
    
    def get_price_precision(self):
        """
        Get the price precision for the trading symbol
        
        Returns:
            Int: Price precision
        """
        symbol_info = self.get_futures_symbol_info()
        
        if symbol_info:
            for filter_item in symbol_info["filters"]:
                if filter_item["filterType"] == "PRICE_FILTER":
                    tick_size = float(filter_item["tickSize"])
                    precision = 0
                    
                    if tick_size < 1:
                        precision = len(str(tick_size).split(".")[1].rstrip("0"))
                    
                    return precision
        
        # Default precision if not found
        return 2
    
    def get_min_notional(self):
        """
        Get the minimum notional value for the trading symbol
        
        Returns:
            Float: Minimum notional value
        """
        symbol_info = self.get_futures_symbol_info()
        
        if symbol_info:
            for filter_item in symbol_info["filters"]:
                if filter_item["filterType"] == "MIN_NOTIONAL":
                    return float(filter_item["notional"])
        
        # Default value if not found
        return 5.0
    
    def get_min_quantity(self):
        """
        Get the minimum quantity for the trading symbol
        
        Returns:
            Float: Minimum quantity
        """
        symbol_info = self.get_futures_symbol_info()
        
        if symbol_info:
            for filter_item in symbol_info["filters"]:
                if filter_item["filterType"] == "LOT_SIZE":
                    return float(filter_item["minQty"])
        
        # Default value if not found
        return 0.001