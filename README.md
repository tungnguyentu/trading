# Crypto Trading Bot

A modular cryptocurrency trading bot for Binance Futures with advanced technical analysis, machine learning integration, and risk management features.

## Features

- Real-time trading on Binance Futures
- Technical analysis with multiple indicators
- Machine learning signal generation
- Dynamic position sizing based on market volatility
- Pyramiding strategy for adding to winning positions
- Dynamic take profit and stop loss levels
- Trend following mode
- Scalping mode for quick profits
- Compound interest option
- Detailed logging and notifications

## Project Structure

The project has been organized into a modular structure for better maintainability:

```
realtime/
├── __init__.py
├── utils/
│   ├── __init__.py
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── notifications.py
│   ├── reporting.py
│   └── trade_manager.py
├── trader/
│   ├── __init__.py
│   ├── account.py
│   ├── core.py
│   ├── ml_integration.py
│   ├── orders.py
│   ├── position.py
│   └── signals.py
└── real_trader.py (legacy file, use the modular structure instead)
```

### Module Descriptions

- **core.py**: Main RealtimeTrader class that orchestrates the trading operations
- **account.py**: Account management for handling account-related operations
- **signals.py**: Signal generation for analyzing market data and generating trading signals
- **position.py**: Position management for handling trading positions
- **orders.py**: Order execution for handling trade execution
- **ml_integration.py**: ML integration for using machine learning models in trading
- **indicators.py**: Technical indicators calculation with multiple library support

## Technical Indicators

The bot uses a flexible technical indicators system that supports multiple libraries:

1. **TA-Lib** (primary, if installed)
2. **pandas_ta** (fallback #1)
3. **ta** (fallback #2)
4. Pure Python implementations (final fallback)

This ensures the bot can run even if TA-Lib is not installed, which can be challenging on some platforms.

### Installing TA-Lib (Optional)

TA-Lib is the recommended library for technical indicators, but it requires a C/C++ library to be installed:

#### For macOS (using Homebrew):
```
brew install ta-lib
pip install TA-Lib
```

#### For Ubuntu/Debian:
```
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

#### For Windows:
Download the pre-built binary from: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip
Extract to `C:\ta-lib`
```
pip install --find-links=https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib TA-Lib
```

If you can't install TA-Lib, the bot will automatically use alternative libraries.

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Binance API keys:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token (optional)
   TELEGRAM_CHAT_ID=your_telegram_chat_id (optional)
   ```

## Usage

Run the trading bot with default settings:

```
python run_realtime_trading.py
```

### Command Line Arguments

The bot supports numerous command line arguments to customize its behavior:

#### Trading Parameters
- `--symbol`: Trading symbol (default: BTCUSDT)
- `--investment`: Initial investment amount in USDT (default: 50.0)
- `--profit-target`: Daily profit target percentage (default: 15.0)
- `--leverage`: Leverage for futures trading (default: 5)

#### Duration Parameters
- `--duration`: Trading duration in hours (default: 24)
- `--interval-minutes`: Update interval in minutes (default: 15)
- `--interval-seconds`: Additional update interval in seconds (default: 0)

#### Trading Modes
- `--test-mode`: Run in test mode without real trades
- `--full-investment`: Use full initial investment for each trade
- `--full-margin`: Use full available margin for each trade
- `--compound`: Enable compound interest

#### Pyramiding Settings
- `--pyramiding`: Enable pyramiding strategy
- `--max-pyramid`: Maximum pyramid entries (default: 2)
- `--pyramid-threshold`: Pyramid entry threshold percentage (default: 0.01)

#### Take Profit and Stop Loss Settings
- `--dynamic-tp`: Use dynamic take profit based on market conditions
- `--fixed-tp`: Fixed take profit percentage (0 to disable)
- `--fixed-sl`: Fixed stop loss percentage (0 to disable)

#### Signal Settings
- `--trend-following`: Enable trend following mode
- `--enhanced-signals`: Use enhanced signal generation
- `--signal-threshold`: Signal confirmation threshold (default: 2)
- `--signal-cooldown`: Signal cooldown in minutes (default: 15)

#### Scalping Settings
- `--scalping`: Enable scalping mode
- `--scalping-tp`: Scalping take profit factor (default: 0.5)
- `--scalping-sl`: Scalping stop loss factor (default: 0.8)

#### ML Settings
- `--ml-signals`: Use ML for signal generation
- `--ml-confidence`: ML confidence threshold (default: 0.6)
- `--train-ml`: Train ML model before trading
- `--retrain-interval`: ML model retraining interval in hours (0 to disable)

#### Position Management
- `--reassess`: Periodically reassess positions

### Example Commands

Basic trading with Bitcoin:
```
python run_realtime_trading.py --symbol BTCUSDT --investment 100 --leverage 3
```

Enhanced trading with technical analysis:
```
python run_realtime_trading.py --symbol ETHUSDT --enhanced-signals --signal-threshold 3 --trend-following
```

Trading with machine learning:
```
python run_realtime_trading.py --ml-signals --train-ml --ml-confidence 0.7 --retrain-interval 6
```

Scalping mode:
```
python run_realtime_trading.py --scalping --scalping-tp 0.3 --scalping-sl 0.5 --interval-minutes 5
```

## Warning

Trading cryptocurrencies involves significant risk and can result in the loss of your invested capital. This bot is provided for educational purposes only. Use at your own risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details.