# Crypto Trading Bot

This is a cryptocurrency trading bot that can run in both simulation mode and real trading mode using the Binance API.

## Features

- Real-time trading on Binance
- Simulation mode for testing strategies
- Test mode with fake balance for safe testing of real trading functionality
- Technical analysis indicators (SMA, ATR)
- Machine learning signal integration
- Risk management (stop loss, take profit)
- Telegram notifications
- Detailed trade history and reporting

## Requirements

- Python 3.7+
- Binance account with API keys
- Telegram bot (optional, for notifications)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Usage

### Simulation Mode

To run in simulation mode:

```
python run_simulation.py --symbol BTCUSDT --investment 50 --leverage 15 --hours 24 --interval 15
```

### Test Mode

To run in test mode with fake balance (no real trades executed):

```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 15 --hours 24 --interval 15 --test
```

### Real Trading Mode

To run in real trading mode:

```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 15 --hours 24 --interval 15
```

### Command Line Arguments

- `--symbol`: Trading pair (default: BTCUSDT)
- `--investment`: Initial investment amount in USD (default: 50.0)
- `--target`: Daily profit target in USD (default: 15.0)
- `--hours`: Duration in hours (default: 24)
- `--interval`: Update interval in minutes (default: 15)
- `--leverage`: Margin trading leverage (default: 15, range: 15-20)
- `--test`: Run in test mode with fake balance (no real trades)

## Risk Warning

**IMPORTANT**: This bot trades with real money when in real trading mode. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software.

## Safety Features

- Test mode with fake balance for safe testing
- Confirmation prompt before starting real trading
- Daily loss limit (10% of initial investment by default)
- Trading disabled automatically when daily loss limit is reached
- Position size limited to a percentage of account balance
- Risk per trade limited to a percentage of account balance

## Results and Reporting

Trading results are saved in the `realtime_results` directory:
- Trade history: `{symbol}_trade_history.csv` (or `test_{symbol}_trade_history.csv` for test mode)
- Current status: `{symbol}_status.csv` (or `test_{symbol}_status.csv` for test mode)

## License

This project is licensed under the MIT License - see the LICENSE file for details.