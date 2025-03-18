"""
Utility functions for the RealtimeTrader.
"""

import os
import time
import urllib.request
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException

def test_binance_connection(api_key, api_secret, is_testnet=False):
    """
    Test connection to Binance API and provide detailed diagnostics
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        is_testnet: Whether to test connection to testnet
        
    Returns:
        dict: Connection test results
    """
    results = {
        'success': False,
        'errors': [],
        'warnings': [],
        'info': [],
        'account_data': None
    }
    
    # Check API key and secret
    if not api_key or not api_secret:
        results['errors'].append("API key or secret is missing or empty")
        return results
    
    # Initialize client
    client = Client(api_key, api_secret, testnet=is_testnet)
    
    # Test basic connectivity
    try:
        results['info'].append("Testing basic API connectivity...")
        server_time = client.get_server_time()
        time_diff = abs(int(time.time() * 1000) - server_time['serverTime'])
        
        results['info'].append(f"Server time: {server_time['serverTime']}")
        results['info'].append(f"Local time: {int(time.time() * 1000)}")
        results['info'].append(f"Time difference: {time_diff}ms")
        
        if time_diff > 1000:
            results['warnings'].append(f"Time difference is high ({time_diff}ms). This might cause issues with signed requests.")
    except BinanceAPIException as e:
        results['errors'].append(f"Basic connectivity test failed: {str(e)}")
        return results
    except Exception as e:
        results['errors'].append(f"Unexpected error during basic connectivity test: {str(e)}")
        return results
    
    # Test futures connectivity
    try:
        results['info'].append("Testing futures API connectivity...")
        
        # Test ping to futures API
        ping = client.futures_ping()
        results['info'].append("Futures ping successful")
        
        # Try to get exchange info
        exchange_info = client.futures_exchange_info()
        results['info'].append(f"Retrieved futures exchange info with {len(exchange_info.get('symbols', []))} symbols")
        
        # Try to get account data
        try:
            account = client.futures_account()
            results['account_data'] = account
            
            # Check if we have USDT balance
            has_usdt = False
            for asset in account.get('assets', []):
                if asset['asset'] == 'USDT':
                    has_usdt = True
                    results['info'].append(f"USDT balance: {float(asset['availableBalance'])}")
                    break
            
            if not has_usdt:
                results['warnings'].append("No USDT balance found in futures account")
                
            results['info'].append("Successfully retrieved futures account data")
            results['success'] = True
            
        except BinanceAPIException as e:
            results['errors'].append(f"Failed to get futures account data: {str(e)}")
            results['info'].append("Common causes: insufficient permissions on API key, or no futures account")
    except BinanceAPIException as e:
        results['errors'].append(f"Futures API test failed: {str(e)}")
    except Exception as e:
        results['errors'].append(f"Unexpected error during futures API test: {str(e)}")
    
    return results

def validate_trading_configuration(trader):
    """
    Validate the trading configuration and report potential issues
    
    Args:
        trader: RealtimeTrader instance
        
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Validate symbol
    try:
        symbol_info = trader.client.futures_exchange_info()
        valid_symbols = [s['symbol'] for s in symbol_info['symbols']]
        
        if trader.symbol not in valid_symbols:
            results['errors'].append(f"Symbol {trader.symbol} is not available for futures trading")
            results['valid'] = False
        else:
            results['info'].append(f"Symbol {trader.symbol} is valid")
    except Exception as e:
        results['warnings'].append(f"Could not validate symbol: {str(e)}")
    
    # Validate leverage
    try:
        max_leverage = 125  # Default max leverage
        for s in symbol_info['symbols']:
            if s['symbol'] == trader.symbol:
                if 'leverageBracket' in s:
                    max_leverage = s['leverageBracket'][0]['initialLeverage']
                break
        
        if trader.leverage > max_leverage:
            results['errors'].append(f"Leverage {trader.leverage} exceeds maximum allowed ({max_leverage})")
            results['valid'] = False
        else:
            results['info'].append(f"Leverage {trader.leverage} is valid")
    except Exception as e:
        results['warnings'].append(f"Could not validate leverage: {str(e)}")
    
    # Validate investment amount
    if trader.initial_investment <= 0:
        results['errors'].append(f"Initial investment must be greater than 0")
        results['valid'] = False
    else:
        results['info'].append(f"Initial investment of {trader.initial_investment} is valid")
    
    # Check for conflicting settings
    if trader.use_dynamic_take_profit and trader.fixed_tp > 0:
        results['warnings'].append(f"Both dynamic take profit and fixed take profit ({trader.fixed_tp}%) are enabled. Fixed take profit will be ignored.")
    
    # Check for ML settings
    if trader.use_ml_signals and not trader.ml_integration:
        results['errors'].append(f"ML signals are enabled but ML integration is not initialized")
        results['valid'] = False
    elif trader.use_ml_signals:
        results['info'].append(f"ML integration is properly initialized")
    
    return results
